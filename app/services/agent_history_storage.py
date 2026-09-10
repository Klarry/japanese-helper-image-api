"""The Japanese Learning Agent's conversation, and how it is persisted.

A single JSON file on disk, per the project's "no database, no new
dependencies" constraint for this feature. The file holds two things, kept
separately: the running ``summary`` of the part of the conversation that is
no longer stored message by message, and the ``messages`` that are still
kept verbatim.

    {"summary": "...", "messages": [{"role": ..., "content": ...}, ...]}

Files written before compression existed have no ``summary`` key; they load
as a conversation with an empty summary, so nothing has to be migrated.

This module owns the conversation's shape (and how it is rendered for a
prompt) plus load/save/clear. It has no opinion about when to summarise or
what to ask Gemini - JapaneseLearningAgent and HistoryCompressor own that.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from app.core.config import AGENT_HISTORY_FILE_PATH

logger = logging.getLogger(__name__)


class HistoryMessage(TypedDict):
    role: str
    content: str


@dataclass
class ConversationHistory:
    """Everything remembered between turns.

    ``summary`` is empty until the first compression happens (and always
    empty while compression is off, which is what keeps the uncompressed
    mode byte-identical to how the agent worked before).
    """

    summary: str = ""
    messages: list[HistoryMessage] = field(default_factory=list)
    # How many tokens the stored summary is, as reported by Gemini when it
    # wrote it - kept so the size can be shown without re-counting it on
    # every turn. None means "not known", never an estimate.
    summary_tokens: int | None = None


def format_transcript(messages: list[HistoryMessage]) -> str:
    """Render messages the way the model sees them.

    Shared by the prompt builder and the summariser on purpose: the summary
    is written about exactly the text the model would otherwise have read.
    """
    return "\n".join(
        f"{'Learner' if entry['role'] == 'user' else 'Assistant'}: {entry['content']}"
        for entry in messages
    )


class AgentHistoryStorage:
    """Loads, saves, and clears the persisted conversation."""

    def __init__(self, file_path: str = AGENT_HISTORY_FILE_PATH) -> None:
        self._path = Path(file_path)

    def load(self) -> ConversationHistory:
        """Return the persisted conversation, messages oldest first.

        A missing file means no conversation has happened yet, and an empty,
        corrupted, or unreadable file is treated the same way rather than
        raised as an error - failing safe to "no history" so a broken file
        never takes the whole agent down, per the project's existing
        error-handling convention of logging and returning a safe fallback.
        """
        if not self._path.exists():
            return ConversationHistory()

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            messages = data["messages"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Could not read agent history from %s: %s", self._path, exc)
            return ConversationHistory()

        if not isinstance(messages, list):
            logger.warning("Agent history at %s is malformed (messages is not a list)", self._path)
            return ConversationHistory()

        summary = data.get("summary", "")

        if not isinstance(summary, str):
            logger.warning("Agent history at %s has a malformed summary; ignoring it", self._path)
            summary = ""

        summary_tokens = data.get("summary_tokens")

        if not isinstance(summary_tokens, int) or isinstance(summary_tokens, bool):
            summary_tokens = None

        return ConversationHistory(
            summary=summary,
            messages=messages,
            summary_tokens=summary_tokens,
        )

    def save(self, history: ConversationHistory) -> None:
        """Persist the summary and the messages, overwriting what was there."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                {
                    "summary": history.summary,
                    "summary_tokens": history.summary_tokens,
                    "messages": history.messages,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def clear(self) -> None:
        """Remove the whole conversation - summary included."""
        self.save(ConversationHistory())


storage = AgentHistoryStorage()
