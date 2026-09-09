"""Day 6 (+ persistent context, + token usage): a small, dedicated LLM agent
for the Japanese-learning app.

Every other module in this package is a bare async function that calls
gemini_service directly. JapaneseLearningAgent is different on purpose: the
task asks for a real agent entity that owns the whole request -> history ->
prompt -> Gemini call -> history update -> response path for open-ended
learner questions, so /agent/chat never does more than `await agent.run(...)`
(and, for the history endpoints, `agent.get_history()` / `agent.clear_history()`).
"""

import logging

from app.schemas.agent import AgentChatResponse, AgentTokenUsage
from app.services.agent_history_storage import AgentHistoryStorage, HistoryMessage, storage
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

    `run` loads the persisted history, builds a Gemini prompt that includes
    it, asks Gemini, and - only once Gemini actually answers - appends both
    the learner's message and the answer to the history and saves it. A
    failed Gemini call raises straight through and nothing is persisted:
    that keeps the file free of orphaned user turns with no reply, which
    would otherwise misrepresent the conversation to the next prompt.

    Token usage: Gemini's generateContent/Interactions response reports one
    combined prompt-token count for everything sent (instructions + history
    + the new message), not a breakdown. To report history tokens and
    current-request tokens separately, the history portion of the prompt is
    measured on its own via Gemini's :countTokens endpoint (a real count,
    not an estimate), and current-request tokens is the remainder of the
    real total. Measuring history tokens is best-effort: if it fails, the
    chat answer is still returned - only the usage numbers that depend on it
    come back as null.
    """

    def __init__(self, history_storage: AgentHistoryStorage = storage) -> None:
        self._history = history_storage

    async def run(self, message: str) -> AgentChatResponse:
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

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": generated.text})
        self._history.save(history)

        return AgentChatResponse(
            response=generated.text,
            usage=self._build_usage(generated.input_tokens, generated.output_tokens, history_tokens),
        )

    def get_history(self) -> list[HistoryMessage]:
        return self._history.load()

    def clear_history(self) -> None:
        self._history.clear()

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
    def _build_history_prefix(history: list[HistoryMessage]) -> str:
        """Everything that would come before the new message in the prompt -
        the exact text token-counted as "history". Empty when there's no
        history yet, so the very first turn reports zero history tokens.
        """
        if not history:
            return ""

        transcript = "\n".join(
            f"{'Learner' if entry['role'] == 'user' else 'Assistant'}: {entry['content']}"
            for entry in history
        )
        return f"{_INSTRUCTIONS}\n\nConversation so far:\n{transcript}"

    @staticmethod
    def _build_prompt(message: str, history_prefix: str) -> str:
        if not history_prefix:
            return f"{_INSTRUCTIONS}\n\nLearner's request: {message}"

        return f"{history_prefix}\n\nLearner's new request: {message}"


agent = JapaneseLearningAgent()
