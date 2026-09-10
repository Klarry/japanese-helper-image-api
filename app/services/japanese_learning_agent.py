"""Day 6-7: a small, dedicated LLM agent for the Japanese-learning app.

Every other module in this package is a bare async function that calls
gemini_service directly. JapaneseLearningAgent is different on purpose: the
task asks for a real agent entity that owns the whole request -> history ->
prompt -> Gemini call -> history update -> response path for open-ended
learner questions, so /agent/chat never does more than `await agent.run(...)`
(and, for the other endpoints, `agent.get_history()`, `agent.clear_history()`
and `agent.get_usage()`).
"""

import logging
from datetime import datetime, timezone

from app.core.config import AGENT_COMPRESSION_ENABLED
from app.schemas.agent import AgentChatResponse, AgentCompressionStatus, AgentTokenUsage
from app.services.agent_history_compressor import HistoryCompressor
from app.services.agent_history_storage import (
    AgentHistoryStorage,
    ConversationHistory,
    format_transcript,
    storage,
)
from app.services.agent_usage_log import AgentUsageLog, UsageRecord, usage_log
from app.services.gemini_service import count_tokens, generate_text_with_usage

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "You are a Japanese language learning assistant. Help the learner with "
    "whatever they asked about - a kanji, a word, grammar, or an example "
    "sentence - with a clear, focused answer aimed at a JLPT learner. "
    "Respond in Russian, unless the learner's message explicitly asks for "
    "a different language. Earlier turns of the conversation are included "
    "below for context - use them to resolve references like \"it\" or "
    "\"that\" back to whatever was being discussed."
)


class JapaneseLearningAgent:
    """Owns the whole conversation, not just a single turn.

    `run` loads the persisted conversation, builds a Gemini prompt that
    includes it, asks Gemini, and - only once Gemini actually answers -
    appends both the learner's message and the answer to the history and
    saves it. A failed Gemini call raises straight through and nothing is
    persisted: that keeps the file free of orphaned user turns with no
    reply, which would otherwise misrepresent the conversation to the next
    prompt.

    Context sent to Gemini, in either of the two modes:

    * compression off (the default) - the entire conversation, every turn,
      exactly as this agent has always worked. Simple, and the baseline the
      experiment measures against; the prompt grows without bound.
    * compression on - the running summary plus the messages still kept
      verbatim. Messages age out of that window into the summary in batches
      (see HistoryCompressor), so what is sent stays bounded instead of
      growing with every turn.

    Token usage: Gemini's generateContent/Interactions response reports one
    combined prompt-token count for everything sent (instructions + history
    + the new message), not a breakdown. To report history tokens and
    current-request tokens separately, the history portion of the prompt is
    measured on its own via Gemini's :countTokens endpoint (a real count,
    not an estimate), and current-request tokens is the remainder of the
    real total. Because it measures what was actually sent, it is also the
    number the compression experiment turns on. Measuring is best-effort:
    if it fails, the chat answer is still returned - only the usage numbers
    that depend on it come back as null.
    """

    def __init__(
        self,
        history_storage: AgentHistoryStorage = storage,
        usage_log: AgentUsageLog = usage_log,
        compressor: HistoryCompressor | None = None,
        compression_enabled: bool = AGENT_COMPRESSION_ENABLED,
    ) -> None:
        self._history = history_storage
        self._usage_log = usage_log
        self._compressor = compressor or HistoryCompressor()
        self._compression_enabled = compression_enabled

    async def run(self, message: str, compression_enabled: bool | None = None) -> AgentChatResponse:
        """Answer one message. ``compression_enabled`` lets the caller pick
        the mode for this request - that is what makes running the same
        dialogue both ways possible from the client. Omitted, it falls back
        to how the server is configured."""
        use_compression = self._compression_enabled if compression_enabled is None else compression_enabled
        history = self._history.load()
        history_prefix = self._build_history_prefix(history)
        prompt = self._build_prompt(message, history_prefix)

        # Let a Gemini failure here (including "context window exceeded")
        # propagate as-is - gemini_service already turns a non-200 response
        # into an HTTPException, so FastAPI reports it as a real error
        # response instead of a crash, and nothing below runs, so nothing
        # is persisted.
        generated = await generate_text_with_usage(prompt)

        history_tokens = await self._count_history_tokens(history_prefix)
        usage = self._build_usage(generated.input_tokens, generated.output_tokens, history_tokens)
        # What this request actually sent, captured before the new turn is
        # appended and before any summarising rearranges it.
        messages_sent = len(history.messages)
        summary_used = bool(history.summary)

        history.messages.append({"role": "user", "content": message})
        history.messages.append({"role": "assistant", "content": generated.text})
        summarization_tokens = await self._compress_if_due(history, use_compression)
        self._history.save(history)

        self._record_usage(usage, use_compression, messages_sent, summary_used, summarization_tokens)

        return AgentChatResponse(
            response=generated.text,
            usage=usage,
            compression=self._build_compression_status(use_compression, history),
        )

    def get_history(self) -> ConversationHistory:
        return self._history.load()

    def clear_history(self) -> None:
        self._history.clear()

    def get_usage(self) -> list[UsageRecord]:
        return self._usage_log.load()

    async def _compress_if_due(self, history: ConversationHistory, use_compression: bool) -> int:
        """Fold aged-out messages into the summary once enough have piled up.

        Best-effort, like counting tokens: the learner's turn has already
        succeeded and is about to be saved, so a failed summarisation is
        logged and left for the next turn to retry rather than turned into
        an error for an answer that was perfectly fine. Nothing is dropped
        in the meantime - the messages simply stay verbatim a while longer.
        """
        if not use_compression or not self._compressor.needs_compression(history.messages):
            return 0

        try:
            result = await self._compressor.compress(history.summary, history.messages)
        except Exception as error:  # noqa: BLE001 - see docstring
            logger.warning("Could not compress agent history: %s", error)
            return 0

        history.summary = result.summary
        history.messages = result.messages
        history.summary_tokens = result.summary_tokens

        return result.tokens_used

    @staticmethod
    def _build_compression_status(
        use_compression: bool,
        history: ConversationHistory,
    ) -> AgentCompressionStatus:
        """The conversation as it now stands - what the next request will
        send, and what the numbers above it were produced with."""
        return AgentCompressionStatus(
            enabled=use_compression,
            summary_tokens=history.summary_tokens if history.summary else 0,
            recent_messages=len(history.messages),
        )

    def _record_usage(
        self,
        usage: AgentTokenUsage,
        use_compression: bool,
        messages_sent: int,
        summary_used: bool,
        summarization_tokens: int,
    ) -> None:
        """Persist what this request cost, so runs with and without
        compression can be compared afterwards. Instrumentation only - a
        failure here must never sink an answer the learner already has."""
        record: UsageRecord = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "compression_enabled": use_compression,
            "messages_sent": messages_sent,
            "summary_used": summary_used,
            "current_request_tokens": usage.current_request_tokens,
            "history_tokens": usage.history_tokens,
            "response_tokens": usage.response_tokens,
            "total_tokens": usage.total_tokens,
            "summarization_tokens": summarization_tokens,
        }

        try:
            self._usage_log.append(record)
        except Exception as error:  # noqa: BLE001 - see docstring
            logger.warning("Could not record agent token usage: %s", error)

    @staticmethod
    async def _count_history_tokens(history_prefix: str) -> int | None:
        if not history_prefix:
            return 0

        try:
            return await count_tokens(history_prefix)
        except Exception as error:  # noqa: BLE001 - a usage-tracking hiccup must not sink a successful answer
            logger.warning("Could not count history tokens: %s", error)
            return None

    @staticmethod
    def _build_usage(
        prompt_tokens: int | None,
        response_tokens: int | None,
        history_tokens: int | None,
    ) -> AgentTokenUsage:
        current_request_tokens = (
            max(prompt_tokens - history_tokens, 0)
            if prompt_tokens is not None and history_tokens is not None
            else None
        )
        total_tokens = (
            prompt_tokens + response_tokens
            if prompt_tokens is not None and response_tokens is not None
            else None
        )

        return AgentTokenUsage(
            current_request_tokens=current_request_tokens,
            history_tokens=history_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _build_history_prefix(history: ConversationHistory) -> str:
        """Everything that would come before the new message in the prompt -
        the exact text token-counted as "history". Empty when nothing has
        been said yet, so the very first turn reports zero history tokens.

        With compression off there is never a summary, so this builds
        precisely the prompt this agent has always built.
        """
        if not history.summary and not history.messages:
            return ""

        sections = [_INSTRUCTIONS]

        if history.summary:
            sections.append(f"Summary of the earlier part of the conversation:\n{history.summary}")

        if history.messages:
            sections.append(f"Conversation so far:\n{format_transcript(history.messages)}")

        return "\n\n".join(sections)

    @staticmethod
    def _build_prompt(message: str, history_prefix: str) -> str:
        if not history_prefix:
            return f"{_INSTRUCTIONS}\n\nLearner's request: {message}"

        return f"{history_prefix}\n\nLearner's new request: {message}"


agent = JapaneseLearningAgent()
