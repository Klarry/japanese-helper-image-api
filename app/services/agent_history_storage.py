"""Persistence for the Japanese Learning Agent's conversation history.

A single JSON file on disk (see the layout in the task: {"messages": [...]}),
per the project's "no database, no new dependencies" constraint for this
feature. This class only knows how to load, save, and clear that file - it
has no opinion about prompts, roles, or Gemini. JapaneseLearningAgent owns
all of that; it just calls this to get messages to and from disk.
"""

import json
import logging
from pathlib import Path
from typing import TypedDict

from app.core.config import AGENT_HISTORY_FILE_PATH

logger = logging.getLogger(__name__)


class HistoryMessage(TypedDict):
    role: str
    content: str


class AgentHistoryStorage:
    """Loads, saves, and clears the persisted conversation history."""

    def __init__(self, file_path: str = AGENT_HISTORY_FILE_PATH) -> None:
        self._path = Path(file_path)

    def load(self) -> list[HistoryMessage]:
        """Return the persisted messages, oldest first.

        A missing file means no conversation has happened yet, and an empty,
        corrupted, or unreadable file is treated the same way rather than
        raised as an error - failing safe to "no history" so a broken file
        never takes the whole agent down, per the project's existing
        error-handling convention of logging and returning a safe fallback.
        """
        if not self._path.exists():
            return []

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            messages = data["messages"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Could not read agent history from %s: %s", self._path, exc)
            return []

        if not isinstance(messages, list):
            logger.warning("Agent history at %s is malformed (messages is not a list)", self._path)
            return []

        return messages

    def save(self, messages: list[HistoryMessage]) -> None:
        """Persist the full message list, overwriting whatever was there."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"messages": messages}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        """Remove all persisted history."""
        self.save([])


storage = AgentHistoryStorage()
