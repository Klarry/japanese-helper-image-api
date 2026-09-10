"""Keeps the agent's conversation from growing without bound.

The idea: the newest messages are worth sending to Gemini word for word,
while everything older is worth sending only for what it established - which
kanji were explained, what the learner is working on, what they asked to be
remembered. So the older messages are replaced by a running summary, and
only that summary plus the recent messages are sent.

Rewriting the summary costs a Gemini call, so it does not happen every turn:
it happens once ``summary_update_threshold`` messages have aged out past the
recent window. The summary is rolling - the previous one is fed back in - so
context from the very beginning of the conversation keeps surviving each
rewrite instead of being dropped one batch at a time.

The summary is produced through the same gemini_service every other feature
in this package uses; there is no second path to the LLM here.
"""

import logging
from typing import NamedTuple

from app.core.config import AGENT_RECENT_MESSAGES_KEPT, AGENT_SUMMARY_UPDATE_THRESHOLD
from app.services.agent_history_storage import HistoryMessage, format_transcript
from app.services.gemini_service import generate_text_with_usage

logger = logging.getLogger(__name__)

_SUMMARY_INSTRUCTIONS = (
    "You are maintaining the long-term memory of a Japanese-language tutoring "
    "conversation. The turns below are about to be dropped from the tutor's "
    "context window, so anything worth keeping has to survive in the summary "
    "you write.\n\n"
    "Rewrite the existing summary so that it also covers the new turns. Keep, "
    "in particular: every kanji, word and grammar point that was discussed and "
    "the essence of what was said about it; the learner's level, goals and "
    "interests; anything the learner said about themselves or asked you to "
    "remember; and mistakes worth revisiting. Drop greetings, filler and "
    "repetition, and do not add anything that was not said.\n\n"
    "Answer with the summary itself and nothing else: compact factual bullet "
    "points in Russian, at most 250 words."
)

_NO_SUMMARY_YET = "(none yet - this is the first summary of this conversation)"


class CompressionResult(NamedTuple):
    """The rewritten summary, the messages still kept verbatim, and what the
    rewrite itself cost - so compression's own token spend is visible in the
    comparison instead of hiding inside it."""

    summary: str
    messages: list[HistoryMessage]
    tokens_used: int
    # The generated summary's own size, straight from Gemini's output-token
    # count for the rewrite - a real number, and free, since the rewrite
    # reports it anyway.
    summary_tokens: int | None


class HistoryCompressor:
    """Decides when the summary is due, and rewrites it."""

    def __init__(
        self,
        recent_messages_kept: int = AGENT_RECENT_MESSAGES_KEPT,
        summary_update_threshold: int = AGENT_SUMMARY_UPDATE_THRESHOLD,
    ) -> None:
        self._recent_kept = recent_messages_kept
        self._threshold = summary_update_threshold

    def needs_compression(self, messages: list[HistoryMessage]) -> bool:
        """True once enough messages have aged out past the recent window."""
        return len(messages) - self._recent_kept >= self._threshold

    async def compress(
        self,
        previous_summary: str,
        messages: list[HistoryMessage],
    ) -> CompressionResult:
        """Fold everything older than the recent window into the summary.

        The recent messages are returned untouched - compression never
        rewrites, trims, or reorders them.
        """
        split_at = max(len(messages) - self._recent_kept, 0)
        older, recent = messages[:split_at], messages[split_at:]

        logger.info("Compressing %s older agent messages into the summary", len(older))
        generated = await generate_text_with_usage(self._build_summary_prompt(previous_summary, older))

        return CompressionResult(
            summary=generated.text,
            messages=recent,
            tokens_used=(generated.input_tokens or 0) + (generated.output_tokens or 0),
            summary_tokens=generated.output_tokens,
        )

    @staticmethod
    def _build_summary_prompt(previous_summary: str, older: list[HistoryMessage]) -> str:
        return (
            f"{_SUMMARY_INSTRUCTIONS}\n\n"
            f"Existing summary:\n{previous_summary or _NO_SUMMARY_YET}\n\n"
            f"Turns to fold into it:\n{format_transcript(older)}"
        )
