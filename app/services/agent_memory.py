"""Day 11: the agent's memory as three explicit layers.

Before this, everything the agent remembered was one list of messages (plus,
under one strategy, a key-value bag of facts). That works until two kinds of
knowledge with different lifetimes end up in the same place: "the learner is
working at N4 for this task" and "the learner's favourite word is 学習" are
both true, but only one of them should still be around next week.

So memory is split into three layers that are stored apart and sent apart:

* Short-term - the current conversation. The messages of this session, in
  order. It is not a copy of the transcript, it *is* the transcript: the
  branch's messages, which persistent history already keeps. Cleared when
  the dialogue is cleared.
* Working - the current task. Goals, requirements, constraints and the
  decisions already taken for the thing being worked on right now. Lives
  with the conversation, and goes away with it: a new dialogue is a new
  task.
* Long-term - the learner. Profile, preferences, important decisions and
  knowledge that stays true in the next conversation. Kept in its own file
  so that clearing the dialogue, or restarting the app, cannot take it.

Nothing here decides *what* belongs in which layer - agent_memory_router
does that, and the memory API lets a client write a layer directly. This
module owns the shapes, how each layer is rendered for the prompt, and how
long-term memory is persisted.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.config import AGENT_LONG_TERM_MEMORY_FILE_PATH, AGENT_MEMORY_ENTRY_LIMIT

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from app.services.agent_history_storage import HistoryMessage

logger = logging.getLogger(__name__)

_MAX_VALUE_LENGTH = 200


def _as_entries(value: Any, limit: int = AGENT_MEMORY_ENTRY_LIMIT) -> list[str]:
    """A layer's list field, from whatever was in the file or the request."""
    if not isinstance(value, list):
        return []

    entries = [str(entry)[:_MAX_VALUE_LENGTH] for entry in value if isinstance(entry, (str, int, float))]

    return entries[:limit]


def _as_profile(value: Any, limit: int = AGENT_MEMORY_ENTRY_LIMIT) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}

    profile = {
        key: str(item)[:_MAX_VALUE_LENGTH]
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, (str, int, float))
    }

    return dict(list(profile.items())[:limit])


def _bullets(caption: str, entries: list[str]) -> list[str]:
    return [f"{caption}: {entry}" for entry in entries]


@dataclass
class ShortTermMemory:
    """The current conversation - this session's messages, oldest first."""

    messages: "list[HistoryMessage]" = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.messages

    def as_section(self) -> str:
        # Imported here rather than at module level: agent_history_storage
        # persists WorkingMemory and so imports this module. Short-term
        # memory is rendered by the same function the strategies use, on
        # purpose - what the model reads as "the conversation" has exactly
        # one spelling.
        from app.services.agent_history_storage import format_transcript

        return "SHORT-TERM MEMORY (the current conversation):\n" + format_transcript(self.messages)


@dataclass
class WorkingMemory:
    """What the task at hand is: its goals, requirements, constraints, and
    the decisions already taken for it. Scoped to this conversation."""

    goals: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.goals or self.requirements or self.constraints or self.decisions)

    def as_section(self) -> str:
        lines = (
            _bullets("Goal", self.goals)
            + _bullets("Requirement", self.requirements)
            + _bullets("Constraint", self.constraints)
            + _bullets("Decision", self.decisions)
        )

        return (
            "WORKING MEMORY (the task being worked on right now - follow it for this task):\n"
            + "\n".join(f"- {line}" for line in lines)
        )


@dataclass
class LongTermMemory:
    """What stays true about the learner after this conversation ends."""

    profile: dict[str, str] = field(default_factory=dict)
    preferences: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    knowledge: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.profile or self.preferences or self.decisions or self.knowledge)

    def as_section(self) -> str:
        lines = (
            [f"{key}: {value}" for key, value in self.profile.items()]
            + _bullets("Preference", self.preferences)
            + _bullets("Decision", self.decisions)
            + _bullets("Knows", self.knowledge)
        )

        return (
            "LONG-TERM MEMORY (what the learner told you to remember about them, "
            "across conversations):\n" + "\n".join(f"- {line}" for line in lines)
        )


@dataclass(frozen=True)
class MemoryLayers:
    """All three layers as one request sees them."""

    short_term: ShortTermMemory = field(default_factory=ShortTermMemory)
    working: WorkingMemory = field(default_factory=WorkingMemory)
    long_term: LongTermMemory = field(default_factory=LongTermMemory)

    def as_sections(self) -> list[str]:
        """The layers a prompt sends, each labelled with the layer it came
        from and ordered from the most durable to the most immediate.

        Empty layers are left out rather than sent as empty headings: an
        empty section tells the model nothing and still costs tokens.
        """
        return [
            layer.as_section()
            for layer in (self.long_term, self.working, self.short_term)
            if not layer.is_empty()
        ]


def working_memory_from_json(value: Any) -> WorkingMemory:
    if not isinstance(value, dict):
        return WorkingMemory()

    return WorkingMemory(
        goals=_as_entries(value.get("goals")),
        requirements=_as_entries(value.get("requirements")),
        constraints=_as_entries(value.get("constraints")),
        decisions=_as_entries(value.get("decisions")),
    )


def working_memory_as_json(memory: WorkingMemory) -> dict[str, Any]:
    return {
        "goals": memory.goals,
        "requirements": memory.requirements,
        "constraints": memory.constraints,
        "decisions": memory.decisions,
    }


def long_term_memory_from_json(value: Any) -> LongTermMemory:
    if not isinstance(value, dict):
        return LongTermMemory()

    return LongTermMemory(
        profile=_as_profile(value.get("profile")),
        preferences=_as_entries(value.get("preferences")),
        decisions=_as_entries(value.get("decisions")),
        knowledge=_as_entries(value.get("knowledge")),
    )


def long_term_memory_as_json(memory: LongTermMemory) -> dict[str, Any]:
    return {
        "profile": memory.profile,
        "preferences": memory.preferences,
        "decisions": memory.decisions,
        "knowledge": memory.knowledge,
    }


class AgentLongTermMemoryStorage:
    """Long-term memory on disk, in its own file.

    Same fail-safe rules as the conversation storage: a missing, empty or
    broken file reads as "nothing remembered yet" rather than raising, so a
    damaged file can never take the agent down with it.
    """

    def __init__(self, file_path: str = AGENT_LONG_TERM_MEMORY_FILE_PATH) -> None:
        self._path = Path(file_path)

    def load(self) -> LongTermMemory:
        if not self._path.exists():
            return LongTermMemory()

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read long-term memory from %s: %s", self._path, exc)
            return LongTermMemory()

        return long_term_memory_from_json(data)

    def save(self, memory: LongTermMemory) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(long_term_memory_as_json(memory), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self.save(LongTermMemory())


long_term_storage = AgentLongTermMemoryStorage()
