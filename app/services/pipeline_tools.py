"""Day 19: what the three pipeline tools do.

search -> summarize -> save_to_file. The MCP server registers them; the work
lives here with the rest of the services, and none of it is new machinery:
search asks the same JLPT vocabulary client Day 17's lookup uses, summarize
is plain Python over what search returned, and saving goes through the same
atomic JSON write the digests use.

Each step works on the previous step's own output and nothing else. That is
the point of the chain rather than three separate lookups: summarize never
touches the network - it cannot, there is no client in it - so whatever it
says came from the search that ran a moment earlier, and the saved file
keeps both the summary and the findings it was made from, so the record on
disk is the whole chain and not just its last line.
"""

import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.jlpt_vocab_api import search_words
from app.services.json_store import write_json

SOURCE = "jlpt-vocab-api.vercel.app"
STAGES = ("search", "summarize", "save_to_file")

# The MCP server subprocess runs this code and is started with a minimal
# environment on purpose (no GEMINI_API_KEY), so the directory is read from
# the environment at call time rather than imported from app.core.config -
# the same reason the digest files are read that way.
DEFAULT_PIPELINE_DIR = "data/pipeline"


def _directory() -> Path:
    return Path(os.getenv("PIPELINE_DIR_PATH", DEFAULT_PIPELINE_DIR))

# Long enough to recognise the query in a file listing, short enough to keep
# the name readable next to the timestamp.
_MAX_SLUG = 24
_UNSAFE_IN_NAME = re.compile(r"[^\w-]+", re.UNICODE)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _jlpt(level: int | None) -> str | None:
    return f"N{level}" if level in (1, 2, 3, 4, 5) else None


# --- stage 1: search --------------------------------------------------------


async def search(query: str) -> dict[str, Any]:
    """Ask the JLPT vocabulary API about ``query`` and return the findings.

    Raises JlptVocabApiError when the API cannot be reached or answers with
    something that is not a word list - the caller turns that into a failed
    step, which is what stops the chain.
    """
    text = query.strip()
    entries = await search_words(text)

    matches = [
        {
            "word": entry.word,
            "reading": entry.furigana or "",
            "romaji": entry.romaji or "",
            "meaning": entry.meaning or "",
            "jlpt_level": _jlpt(entry.level) or "",
        }
        for entry in entries
    ]

    return {
        "query": text,
        "found": bool(matches),
        "count": len(matches),
        "matches": matches,
        "source": SOURCE,
        "searched_at": _stamp(_now()),
    }


# --- stage 2: summarize -----------------------------------------------------


def _entry_line(match: dict[str, Any]) -> str:
    """One dictionary entry as a person would read it out: word - reading
    (romaji) - meaning - level, with whatever the API left empty left out."""
    reading = str(match.get("reading") or "")
    romaji = str(match.get("romaji") or "")
    read_as = f"{reading} ({romaji})" if reading and romaji else reading or romaji

    parts = (str(match.get("word") or ""), read_as, str(match.get("meaning") or ""), str(match.get("jlpt_level") or ""))

    return " - ".join(part for part in parts if part)


def summarize(findings: dict[str, Any]) -> dict[str, Any]:
    """Turn what search found into a short summary. No lookup happens here."""
    query = str(findings.get("query", "")).strip()
    matches = [match for match in (findings.get("matches") or []) if isinstance(match, dict)]
    levels = Counter(str(match.get("jlpt_level") or "") for match in matches)
    levels.pop("", None)

    if not matches:
        headline = f"'{query}' is not in the JLPT vocabulary list."
        summary = (
            f"{headline} The list covers N5 to N1 vocabulary, so a word missing from it is "
            "not necessarily wrong - it is simply not one of the words the exam expects."
        )
    else:
        entries = "; ".join(_entry_line(match) for match in matches)
        level_note = ", ".join(f"{level} x{count}" for level, count in sorted(levels.items()))
        headline = (
            f"'{query}': {len(matches)} JLPT entr{'y' if len(matches) == 1 else 'ies'}"
            + (f", {level_note}" if level_note else "")
            + "."
        )
        summary = f"{headline} {entries}."

    return {
        "query": query,
        "headline": headline,
        "summary": summary,
        "based_on": len(matches),
        "levels": dict(sorted(levels.items())),
        "words": [str(match.get("word", "")) for match in matches],
        "source": str(findings.get("source") or SOURCE),
        "searched_at": str(findings.get("searched_at") or ""),
        "summarized_at": _stamp(_now()),
    }


# --- stage 3: save_to_file --------------------------------------------------


def _slug(query: str) -> str:
    cleaned = _UNSAFE_IN_NAME.sub("-", query).strip("-")

    return cleaned[:_MAX_SLUG] or "query"


def _free_path(directory: Path, stem: str) -> Path:
    """The first name nothing is using. Two runs within the same second are
    rare and would otherwise overwrite each other."""
    candidate = directory / f"{stem}.json"
    attempt = 2

    while candidate.exists():
        candidate = directory / f"{stem}-{attempt}.json"
        attempt += 1

    return candidate


def save_to_file(summary: dict[str, Any], findings: dict[str, Any]) -> dict[str, Any]:
    """Write the summary and the findings it came from to a timestamped JSON
    file, and report where it went."""
    moment = _now()
    query = str(summary.get("query") or findings.get("query") or "").strip()
    path = _free_path(_directory(), f"{moment.strftime('%Y%m%dT%H%M%S')}-{_slug(query)}")

    written = write_json(
        path,
        {
            "saved_at": _stamp(moment),
            "query": query,
            "pipeline": list(STAGES),
            "summary": summary,
            "findings": findings,
        },
    )

    return {
        "status": "saved",
        "file_name": path.name,
        "path": str(path),
        "bytes_written": written,
        "query": query,
        "saved_at": _stamp(moment),
    }
