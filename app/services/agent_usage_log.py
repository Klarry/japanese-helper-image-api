"""Token usage of every /agent/chat call, kept on disk.

The point of this file is the comparison the compression experiment is for:
run a dialogue with compression off, run it again with compression on, and
read back what each request actually cost. Each record carries the flag it
was produced under, so the two runs can be told apart afterwards.

Same shape and the same fail-safe rules as AgentHistoryStorage: a JSON file,
a missing or broken one reads as "nothing recorded yet" rather than an error.
This is instrumentation - it must never be the reason a chat turn fails.
"""

import json
import logging
from pathlib import Path
from typing import TypedDict

from app.core.config import AGENT_USAGE_LOG_FILE_PATH

logger = logging.getLogger(__name__)


class UsageRecord(TypedDict):
    """One /agent/chat call. Token counts are ``None`` exactly where Gemini
    did not report them, never an estimate standing in for a real value."""

    timestamp: str
    strategy: str
    compression_enabled: bool
    messages_sent: int
    summary_used: bool
    current_request_tokens: int | None
    history_tokens: int | None
    response_tokens: int | None
    total_tokens: int | None
    summarization_tokens: int
    facts_tokens: int
    memory_tokens: int
    task_tokens: int
    tool_tokens: int


class AgentUsageLog:
    """Appends and reads back the per-request usage records."""

    def __init__(self, file_path: str = AGENT_USAGE_LOG_FILE_PATH) -> None:
        self._path = Path(file_path)

    def load(self) -> list[UsageRecord]:
        if not self._path.exists():
            return []

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            entries = data["entries"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Could not read agent usage log from %s: %s", self._path, exc)
            return []

        if not isinstance(entries, list):
            logger.warning("Agent usage log at %s is malformed (entries is not a list)", self._path)
            return []

        return entries

    def append(self, entry: UsageRecord) -> None:
        entries = self.load()
        entries.append(entry)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"entries": entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


usage_log = AgentUsageLog()
