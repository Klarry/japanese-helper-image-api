"""Day 6 (+ persistent context): a small, dedicated LLM agent for the
Japanese-learning app.

Every other module in this package is a bare async function that calls
gemini_service directly. JapaneseLearningAgent is different on purpose: the
task asks for a real agent entity that owns the whole request -> history ->
prompt -> Gemini call -> history update -> response path for open-ended
learner questions, so /agent/chat never does more than `await agent.run(...)`
(and, for the history endpoints, `agent.get_history()` / `agent.clear_history()`).
"""

from app.services.agent_history_storage import AgentHistoryStorage, HistoryMessage, storage
from app.services.gemini_service import generate_text

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
    """

    def __init__(self, history_storage: AgentHistoryStorage = storage) -> None:
        self._history = history_storage

    async def run(self, message: str) -> str:
        history = self._history.load()
        prompt = self._build_prompt(message, history)

        response_text = await generate_text(prompt)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response_text})
        self._history.save(history)

        return response_text

    def get_history(self) -> list[HistoryMessage]:
        return self._history.load()

    def clear_history(self) -> None:
        self._history.clear()

    @staticmethod
    def _build_prompt(message: str, history: list[HistoryMessage]) -> str:
        if not history:
            return f"{_INSTRUCTIONS}\n\nLearner's request: {message}"

        transcript = "\n".join(
            f"{'Learner' if entry['role'] == 'user' else 'Assistant'}: {entry['content']}"
            for entry in history
        )
        return (
            f"{_INSTRUCTIONS}\n\n"
            f"Conversation so far:\n{transcript}\n\n"
            f"Learner's new request: {message}"
        )


agent = JapaneseLearningAgent()
