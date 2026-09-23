"""Day 18: a digest that collects itself.

A periodic task is three things kept apart on purpose:

* the **task** - what to collect and how often. Written once, when the tool
  creates it (``data/digest_tasks.json``).
* the **state** - every run that happened, the words they brought back and
  the counters over them (``data/digest_store.json``).
* the **runner** - the scheduler in the backend (see digest_scheduler), the
  only thing that writes the state.

They are kept apart because two processes are involved: the MCP server runs
as a short-lived subprocess (it creates the task and reads the digest), while
the runs happen in the long-lived backend. One writer per file, JSON written
atomically, so neither ever reads half of what the other wrote.

Both files are plain JSON on disk, so a restart loses nothing: the task is
still there, the runs already done are still counted, and the next run
happens on schedule as if nothing had stopped.

Like the rest of the project, this module stays clear of app.core.config -
the MCP subprocess imports it without GEMINI_API_KEY in its environment.
"""

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.services.jlpt_vocab_api import JlptVocabApiError, VocabWord, random_word

logger = logging.getLogger(__name__)

DEFAULT_TASKS_FILE = "data/digest_tasks.json"
DEFAULT_STORE_FILE = "data/digest_store.json"

# How many words one run collects.
WORDS_PER_RUN = 3
# Anything faster would hammer a free public API; anything slower than a day
# is not a digest.
MIN_INTERVAL_SECONDS = 10
MAX_INTERVAL_SECONDS = 86_400
# The counters are for ever; the lists are a window over the newest entries,
# so the file cannot grow without end.
RUNS_KEPT = 50
ITEMS_KEPT = 100
_LEVEL_IN_QUERY = re.compile(r"\bN([1-5])\b", re.IGNORECASE)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def as_iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def from_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def tasks_file() -> str:
    return os.getenv("DIGEST_TASKS_FILE_PATH", DEFAULT_TASKS_FILE)


def store_file() -> str:
    return os.getenv("DIGEST_STORE_FILE_PATH", DEFAULT_STORE_FILE)


@dataclass(frozen=True)
class DigestTask:
    """What to collect, and how often."""

    id: str
    query: str
    interval_seconds: int
    level: int | None = None
    created_at: str = ""
    active: bool = True

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "interval_seconds": self.interval_seconds,
            "level": self.level,
            "created_at": self.created_at,
            "active": self.active,
        }


@dataclass(frozen=True)
class CollectedItem:
    word: str
    reading: str = ""
    romaji: str = ""
    meaning: str = ""
    jlpt_level: str = ""
    collected_at: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "reading": self.reading,
            "romaji": self.romaji,
            "meaning": self.meaning,
            "jlpt_level": self.jlpt_level,
            "collected_at": self.collected_at,
        }


@dataclass(frozen=True)
class DigestRun:
    """One execution of the task: when, whether the API answered, how much it
    brought back."""

    started_at: str
    ok: bool = True
    items: int = 0
    error: str = ""

    def as_json(self) -> dict[str, Any]:
        return {"started_at": self.started_at, "ok": self.ok, "items": self.items, "error": self.error}


@dataclass
class DigestState:
    """Everything the runs of one task have produced.

    The counters are totals since the task was created; ``runs`` and
    ``items`` are the newest ones, kept for the summary.
    """

    total_runs: int = 0
    failed_runs: int = 0
    total_items: int = 0
    first_run_at: str = ""
    last_run_at: str = ""
    last_error: str = ""
    runs: list[DigestRun] = field(default_factory=list)
    items: list[CollectedItem] = field(default_factory=list)

    def record(self, run: DigestRun, items: list[CollectedItem]) -> None:
        self.total_runs += 1
        self.total_items += len(items)
        self.first_run_at = self.first_run_at or run.started_at
        self.last_run_at = run.started_at

        if run.ok:
            self.last_error = ""
        else:
            self.failed_runs += 1
            self.last_error = run.error

        self.runs = (self.runs + [run])[-RUNS_KEPT:]
        self.items = (self.items + items)[-ITEMS_KEPT:]

    def as_json(self) -> dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "failed_runs": self.failed_runs,
            "total_items": self.total_items,
            "first_run_at": self.first_run_at,
            "last_run_at": self.last_run_at,
            "last_error": self.last_error,
            "runs": [run.as_json() for run in self.runs],
            "items": [item.as_json() for item in self.items],
        }


def level_in(query: str) -> int | None:
    """A JLPT level named in the query ("collect N3 words") decides which
    level the run asks for; without one, any level is fine."""
    found = _LEVEL_IN_QUERY.search(query)

    return int(found.group(1)) if found else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write, then move into place: the other process never reads a file that
    is half written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    """Fail-safe like every other store here: a missing or broken file reads
    as empty rather than raising."""
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Could not read %s: %s", path, error)
        return {}

    return data if isinstance(data, dict) else {}


class DigestTaskStorage:
    """The periodic tasks. Written only when a task is created."""

    def __init__(self, file_path: str | None = None) -> None:
        self._path = Path(file_path or tasks_file())

    def load(self) -> list[DigestTask]:
        raw = _read_json(self._path).get("tasks")

        if not isinstance(raw, list):
            return []

        tasks = []

        for entry in raw:
            if not isinstance(entry, dict) or not _text(entry.get("id")):
                continue

            interval = entry.get("interval_seconds")

            tasks.append(
                DigestTask(
                    id=_text(entry.get("id")),
                    query=_text(entry.get("query")),
                    interval_seconds=int(interval) if isinstance(interval, int) else MIN_INTERVAL_SECONDS,
                    level=entry.get("level") if entry.get("level") in (1, 2, 3, 4, 5) else None,
                    created_at=_text(entry.get("created_at")),
                    active=entry.get("active", True) is not False,
                )
            )

        return tasks

    def save(self, tasks: list[DigestTask]) -> None:
        _write_json(self._path, {"tasks": [task.as_json() for task in tasks]})

    def create(self, query: str, interval_seconds: int) -> DigestTask:
        """Add a task and return it, with the id the caller will see."""
        tasks = self.load()
        task = DigestTask(
            id=f"digest-{len(tasks) + 1}",
            query=query.strip(),
            interval_seconds=max(MIN_INTERVAL_SECONDS, min(int(interval_seconds), MAX_INTERVAL_SECONDS)),
            level=level_in(query),
            created_at=as_iso(now_utc()),
        )
        self.save(tasks + [task])

        return task

    def latest(self) -> DigestTask | None:
        tasks = self.load()

        return tasks[-1] if tasks else None


class DigestStore:
    """What the runs produced, by task id. Written only by the scheduler."""

    def __init__(self, file_path: str | None = None) -> None:
        self._path = Path(file_path or store_file())

    def load(self) -> dict[str, DigestState]:
        raw = _read_json(self._path).get("digests")

        if not isinstance(raw, dict):
            return {}

        return {task_id: _state_from_json(entry) for task_id, entry in raw.items() if isinstance(entry, dict)}

    def save(self, states: dict[str, DigestState]) -> None:
        _write_json(self._path, {"digests": {task_id: state.as_json() for task_id, state in states.items()}})

    def state_of(self, task_id: str) -> DigestState:
        return self.load().get(task_id, DigestState())

    def record(self, task_id: str, run: DigestRun, items: list[CollectedItem]) -> DigestState:
        """Append one run. Read-modify-write on every run, so a restart in
        between loses at most the run that was in flight."""
        states = self.load()
        state = states.get(task_id, DigestState())
        state.record(run, items)
        states[task_id] = state
        self.save(states)

        return state


def _state_from_json(entry: dict[str, Any]) -> DigestState:
    runs = entry.get("runs")
    items = entry.get("items")

    return DigestState(
        total_runs=int(entry.get("total_runs") or 0),
        failed_runs=int(entry.get("failed_runs") or 0),
        total_items=int(entry.get("total_items") or 0),
        first_run_at=_text(entry.get("first_run_at")),
        last_run_at=_text(entry.get("last_run_at")),
        last_error=_text(entry.get("last_error")),
        runs=[
            DigestRun(
                started_at=_text(run.get("started_at")),
                ok=run.get("ok", True) is not False,
                items=int(run.get("items") or 0),
                error=_text(run.get("error")),
            )
            for run in (runs if isinstance(runs, list) else [])
            if isinstance(run, dict)
        ],
        items=[
            CollectedItem(
                word=_text(item.get("word")),
                reading=_text(item.get("reading")),
                romaji=_text(item.get("romaji")),
                meaning=_text(item.get("meaning")),
                jlpt_level=_text(item.get("jlpt_level")),
                collected_at=_text(item.get("collected_at")),
            )
            for item in (items if isinstance(items, list) else [])
            if isinstance(item, dict) and _text(item.get("word"))
        ],
    )


def _as_item(word: VocabWord, moment: str) -> CollectedItem:
    return CollectedItem(
        word=word.word,
        reading=word.furigana or "",
        romaji=word.romaji or "",
        meaning=word.meaning or "",
        jlpt_level=f"N{word.level}" if word.level in (1, 2, 3, 4, 5) else "",
        collected_at=moment,
    )


async def collect_once(task: DigestTask, store: DigestStore) -> DigestRun:
    """One run: ask the JLPT API for a few words, store them with a
    timestamp, update the counters.

    An API that fails does not end the task: the run is recorded as failed,
    with its reason, and whatever it managed to collect is kept. The next run
    happens on schedule as usual.
    """
    started = now_utc()
    moment = as_iso(started)
    items: list[CollectedItem] = []
    error = ""

    for _ in range(WORDS_PER_RUN):
        try:
            items.append(_as_item(await random_word(task.level), moment))
        except JlptVocabApiError as failure:
            error = str(failure)
            logger.warning("Digest %s: %s", task.id, error)
            break

    run = DigestRun(started_at=moment, ok=not error, items=len(items), error=error)
    store.record(task.id, run, items)
    logger.info(
        "Digest %s ran: %s, %d item(s), %d run(s) in total",
        task.id,
        "ok" if run.ok else f"failed - {error}",
        run.items,
        store.state_of(task.id).total_runs,
    )

    return run


def next_run_at(task: DigestTask, state: DigestState) -> datetime | None:
    """When this task is due. ``None`` means "now" - a task that has never
    run starts at once, and one whose time passed while the backend was down
    runs once on the way back up, not once per missed interval."""
    last = from_iso(state.last_run_at)

    return last + timedelta(seconds=task.interval_seconds) if last else None


def build_digest(task: DigestTask, state: DigestState) -> dict[str, Any]:
    """The aggregate, as the MCP tool returns it: how many runs, when the
    last one was, what was collected and one line of summary."""
    levels = Counter(item.jlpt_level for item in state.items if item.jlpt_level)
    unique = {item.word for item in state.items}
    due = next_run_at(task, state)

    return {
        "task_id": task.id,
        "query": task.query,
        "interval_seconds": task.interval_seconds,
        "active": task.active,
        "runs": state.total_runs,
        "failed_runs": state.failed_runs,
        "last_run": state.last_run_at,
        "next_run": as_iso(due) if due else "as soon as the scheduler ticks",
        "items_collected": state.total_items,
        "unique_words": len(unique),
        "levels": dict(sorted(levels.items())),
        "latest_items": [item.as_json() for item in state.items[-5:]][::-1],
        "last_error": state.last_error,
        "summary": summarise(task, state, levels),
    }


def summarise(task: DigestTask, state: DigestState, levels: Counter) -> str:
    """One sentence a model can read out loud."""
    if not state.total_runs:
        return (
            f"The task '{task.query}' is set up to run every {task.interval_seconds}s "
            "but has not run yet."
        )

    parts = [
        f"{state.total_items} word(s) collected for '{task.query}' in {state.total_runs} run(s) "
        f"every {task.interval_seconds}s, since {state.first_run_at}"
    ]

    if levels:
        parts.append("levels: " + ", ".join(f"{level} x{count}" for level, count in sorted(levels.items())))

    newest = state.items[-1] if state.items else None

    if newest:
        parts.append(f"newest: {newest.word} ({newest.reading}) - {newest.meaning}")

    if state.failed_runs:
        parts.append(f"{state.failed_runs} run(s) failed, last error: {state.last_error}")

    return "; ".join(parts) + "."
