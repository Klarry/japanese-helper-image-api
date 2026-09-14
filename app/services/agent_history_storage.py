"""The Japanese Learning Agent's conversation, and how it is persisted.

A single JSON file on disk, per the project's "no database, no new
dependencies" constraint for this feature.

A conversation is a set of named branches. There is always a ``main``
branch, and one of them is current. Each branch owns its messages, its
key-value ``facts``, and (for the summary strategy) its running ``summary``
- so nothing said on one branch can reach another. A checkpoint captures a
branch's contents as they are; a new branch starts as a copy of that
capture, which is what lets two branches fork from the same point and then
diverge without touching each other.

    {
      "strategy": "sticky_facts",
      "current_branch": "main",
      "branches": {
        "main": {"summary": "", "summary_tokens": null,
                 "facts": {"goal": "..."}, "messages": [{"role": ..., "content": ...}]}
      },
      "checkpoints": {"cp-1": {"branch": "main", "history": {...}}},
      "working_memory": {"goals": [...], "requirements": [...],
                         "constraints": [...], "decisions": [...]}
    }

A file written before branches existed holds a single conversation at the
top level, with no ``branches`` key; it loads as the ``main`` branch, so
nothing has to be migrated by hand.

This module owns the conversation's shape (and how it is rendered for a
prompt) plus load/save/clear. It has no opinion about what to send to Gemini
or when - JapaneseLearningAgent and the strategy modules own that.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from app.core.config import AGENT_HISTORY_FILE_PATH
from app.services.agent_memory import WorkingMemory, working_memory_as_json, working_memory_from_json

logger = logging.getLogger(__name__)

MAIN_BRANCH = "main"


class HistoryMessage(TypedDict):
    role: str
    content: str


@dataclass
class ConversationHistory:
    """Everything one branch remembers between turns.

    ``summary`` is only ever written by the summary strategy and stays empty
    under every other one. ``facts`` is only written by Sticky Facts. Both
    sit here rather than in the agent so a branch carries its whole state.
    """

    summary: str = ""
    messages: list[HistoryMessage] = field(default_factory=list)
    # How many tokens the stored summary is, as reported by Gemini when it
    # wrote it - kept so the size can be shown without re-counting it on
    # every turn. None means "not known", never an estimate.
    summary_tokens: int | None = None
    facts: dict[str, str] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """A branch's contents at the moment the checkpoint was taken.

    The whole conversation is copied rather than a position in it: a
    strategy may rewrite or shorten the branch's message list afterwards,
    and an index into that list would then point somewhere else entirely.
    """

    branch: str
    history: ConversationHistory


@dataclass
class ConversationState:
    """Everything persisted for the agent: the branches, the checkpoints
    taken on them, which branch is being talked on, which strategy was
    chosen (empty until a client chooses one), and the working memory of the
    task being worked on.

    ``working_memory`` sits beside the branches rather than inside them, and
    is never folded into a transcript: it is the current task's own layer,
    and mixing it into the messages is exactly what Day 11's memory model
    exists to avoid. Long-term memory is not here at all - it lives in its
    own file (see agent_memory), because it has to survive this whole object
    being thrown away.
    """

    strategy: str = ""
    current_branch: str = MAIN_BRANCH
    branches: dict[str, ConversationHistory] = field(
        default_factory=lambda: {MAIN_BRANCH: ConversationHistory()}
    )
    checkpoints: dict[str, Checkpoint] = field(default_factory=dict)
    working_memory: "WorkingMemory" = field(default_factory=lambda: WorkingMemory())

    def current(self) -> ConversationHistory:
        """The branch being talked on. Created empty if it somehow went
        missing, so a damaged file can never leave the agent without one."""
        return self.branches.setdefault(self.current_branch, ConversationHistory())


def format_transcript(messages: list[HistoryMessage]) -> str:
    """Render messages the way the model sees them.

    Shared by every strategy and by the summariser on purpose: what gets
    summarised or remembered is exactly the text the model would have read.
    """
    return "\n".join(
        f"{'Learner' if entry['role'] == 'user' else 'Assistant'}: {entry['content']}"
        for entry in messages
    )


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _as_token_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None

    return value


def _as_messages(value: Any) -> list[HistoryMessage]:
    if not isinstance(value, list):
        return []

    return [
        entry
        for entry in value
        if isinstance(entry, dict) and isinstance(entry.get("role"), str) and isinstance(entry.get("content"), str)
    ]


def _as_facts(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}

    return {key: item for key, item in value.items() if isinstance(key, str) and isinstance(item, str)}


def _as_history(value: Any) -> ConversationHistory:
    if not isinstance(value, dict):
        return ConversationHistory()

    return ConversationHistory(
        summary=_as_text(value.get("summary")),
        messages=_as_messages(value.get("messages")),
        summary_tokens=_as_token_count(value.get("summary_tokens")),
        facts=_as_facts(value.get("facts")),
    )


def _history_as_json(history: ConversationHistory) -> dict[str, Any]:
    return {
        "summary": history.summary,
        "summary_tokens": history.summary_tokens,
        "facts": history.facts,
        "messages": history.messages,
    }


class AgentHistoryStorage:
    """Loads, saves, and clears the persisted conversation."""

    def __init__(self, file_path: str = AGENT_HISTORY_FILE_PATH) -> None:
        self._path = Path(file_path)

    def load(self) -> ConversationState:
        """Return the persisted state, each branch's messages oldest first.

        A missing file means no conversation has happened yet, and an empty,
        corrupted, or unreadable file is treated the same way rather than
        raised as an error - failing safe to "no history" so a broken file
        never takes the whole agent down, per the project's existing
        error-handling convention of logging and returning a safe fallback.
        """
        if not self._path.exists():
            return ConversationState()

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read agent history from %s: %s", self._path, exc)
            return ConversationState()

        if not isinstance(data, dict):
            logger.warning("Agent history at %s is malformed (not an object)", self._path)
            return ConversationState()

        if "branches" not in data:
            return self._load_single_branch(data)

        return self._load_branches(data)

    @staticmethod
    def _load_single_branch(data: dict[str, Any]) -> ConversationState:
        """A file written before branches existed: one conversation at the
        top level, which is the main branch."""
        if not isinstance(data.get("messages"), list):
            return ConversationState()

        return ConversationState(branches={MAIN_BRANCH: _as_history(data)})

    @staticmethod
    def _load_branches(data: dict[str, Any]) -> ConversationState:
        raw_branches = data.get("branches")

        if not isinstance(raw_branches, dict):
            logger.warning("Agent history is malformed (branches is not an object)")
            return ConversationState()

        branches = {
            name: _as_history(history) for name, history in raw_branches.items() if isinstance(name, str)
        }

        if not branches:
            branches = {MAIN_BRANCH: ConversationHistory()}

        raw_checkpoints = data.get("checkpoints")
        checkpoints: dict[str, Checkpoint] = {}

        if isinstance(raw_checkpoints, dict):
            for name, saved in raw_checkpoints.items():
                if isinstance(name, str) and isinstance(saved, dict):
                    checkpoints[name] = Checkpoint(
                        branch=_as_text(saved.get("branch")) or MAIN_BRANCH,
                        history=_as_history(saved.get("history")),
                    )

        current_branch = _as_text(data.get("current_branch")) or MAIN_BRANCH

        if current_branch not in branches:
            logger.warning("Agent history points at a missing branch %r; using %r", current_branch, MAIN_BRANCH)
            current_branch = next(iter(branches))

        return ConversationState(
            strategy=_as_text(data.get("strategy")),
            current_branch=current_branch,
            branches=branches,
            checkpoints=checkpoints,
            working_memory=working_memory_from_json(data.get("working_memory")),
        )

    def save(self, state: ConversationState) -> None:
        """Persist the whole state, overwriting what was there."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                {
                    "strategy": state.strategy,
                    "current_branch": state.current_branch,
                    "branches": {
                        name: _history_as_json(history) for name, history in state.branches.items()
                    },
                    "checkpoints": {
                        name: {"branch": saved.branch, "history": _history_as_json(saved.history)}
                        for name, saved in state.checkpoints.items()
                    },
                    "working_memory": working_memory_as_json(state.working_memory),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def clear(self) -> None:
        """Remove the whole conversation - every branch and checkpoint."""
        self.save(ConversationState())


storage = AgentHistoryStorage()
