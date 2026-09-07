"""Day 6: a small, dedicated LLM agent for the Japanese-learning app.

Every other module in this package is a bare async function that calls
gemini_service directly. JapaneseLearningAgent is different on purpose: the
task asks for a real agent entity that owns the whole request -> prompt ->
Gemini call -> response path for open-ended learner questions, so
/agent/chat never does more than `await agent.run(message)`.
"""

from app.services.gemini_service import generate_text

_INSTRUCTIONS = (
    "You are a Japanese language learning assistant. Help the learner with "
    "whatever they asked about - a kanji, a word, grammar, or an example "
    "sentence - with a clear, focused answer aimed at a JLPT learner. "
    "Respond in Russian, unless the learner's message explicitly asks for "
    "a different language."
)


class JapaneseLearningAgent:
    """Builds the prompt for a learner's message, asks Gemini, and returns
    its answer. All prompt construction and Gemini interaction for
    /agent/chat lives here - the route never touches either.
    """

    async def run(self, message: str) -> str:
        prompt = self._build_prompt(message)
        return await generate_text(prompt)

    @staticmethod
    def _build_prompt(message: str) -> str:
        return f"{_INSTRUCTIONS}\n\nLearner's request: {message}"


agent = JapaneseLearningAgent()
