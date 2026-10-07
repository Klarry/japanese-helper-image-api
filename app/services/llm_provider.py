"""Day 28: which model writes the answer, behind one small interface.

The RAG pipeline has three model calls in it, not one, and that is the
whole reason this module exists. Swapping only the obvious one - the call
that writes the answer - would leave the query rewrite (Day 23) and the
claim-support check (Day 24) talking to the cloud while the endpoint
claimed to be local. So the provider is passed to all three, and a test
drives the whole pipeline with every Gemini entry point booby-trapped.

Nothing about retrieval changes. The FAISS index, the chunk metadata, the
similarity filter, the reranker, the relevance gate and the citation
validator are the ones Days 21-24 built and are used exactly as they are:
this module only decides who is handed the finished prompt.

Neither provider is a new client. ``GeminiProvider`` calls the same
``gemini_service`` every other caller does, and ``OllamaProvider`` calls
the ``ollama_service`` of Day 26 - there is one Ollama client in this
project, and a check that says so.
"""

import logging
from dataclasses import dataclass
from typing import Protocol

from app.services import ollama_service
from app.services.gemini_service import TEXT_MODEL, generate_text_with_usage

logger = logging.getLogger(__name__)

GEMINI = "gemini"
OLLAMA = "ollama"
PROVIDERS = (GEMINI, OLLAMA)


class ProviderUnavailable(Exception):
    """This provider cannot answer right now.

    Raised rather than handled, and deliberately never caught into a
    different provider: the point of naming one is that the answer came
    from it.
    """

    def __init__(self, provider: str, message: str, reason: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.message = message
        self.reason = reason


@dataclass(frozen=True)
class Generated:
    """What a provider produced.

    ``text`` is the ``generate(prompt) -> str`` the assignment asks for;
    the rest is what the provider reported about the run and is carried so
    the two can be compared on more than prose.
    """

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def tokens(self) -> int:
        return (self.input_tokens or 0) + (self.output_tokens or 0)


class LLMProvider(Protocol):
    """One prompt in, one answer out. Everything else is the caller's."""

    name: str
    model: str

    async def generate(self, prompt: str, temperature: float | None = None) -> Generated: ...


class GeminiProvider:
    """The hosted model, exactly as every other caller reaches it."""

    name = GEMINI

    def __init__(self, model: str = TEXT_MODEL) -> None:
        self.model = model

    async def generate(self, prompt: str, temperature: float | None = None) -> Generated:
        generated = await generate_text_with_usage(prompt, model=self.model, temperature=temperature)

        return Generated(
            text=generated.text,
            model=self.model,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
        )


class OllamaProvider:
    """The model on this machine, through the Day 26 service.

    ``temperature`` is accepted and ignored. Day 26 fixed the request shape
    at model/prompt/stream and a test pins it; the two places that pass a
    temperature (the query rewrite and the support check) both fall back
    safely on their own when the model wanders, so buying determinism here
    would cost more than it is worth.
    """

    name = OLLAMA

    def __init__(self, model: str = "") -> None:
        self.model = model or ollama_service.MODEL

    async def generate(self, prompt: str, temperature: float | None = None) -> Generated:
        try:
            answer = await ollama_service.generate(prompt, model=self.model)
        except ollama_service.OllamaUnavailable as error:
            raise ProviderUnavailable(OLLAMA, error.message, error.reason) from error

        return Generated(
            text=answer.response,
            model=answer.model,
            input_tokens=answer.prompt_tokens,
            output_tokens=answer.response_tokens,
        )


def provider_for(name: str) -> LLMProvider:
    """The provider a request named.

    An unknown name is refused rather than defaulted. Falling back to the
    cloud for a typo is the one failure this day is about.
    """
    chosen = (name or GEMINI).strip().lower()

    if chosen == GEMINI:
        return GeminiProvider()

    if chosen == OLLAMA:
        return OllamaProvider()

    raise ValueError(f"unknown provider: {name!r} (expected one of {', '.join(PROVIDERS)})")


async def text_from(provider: LLMProvider | None, prompt: str, temperature: float | None = None) -> str:
    """Generate with a provider, or return None-shaped nothing.

    Used by the two helpers that had a Gemini call of their own before this
    day existed. ``None`` means "behave exactly as you did", which is why
    the old call site is still there and still the default: 800 tests reach
    the model by patching those module-level names, and moving the call
    would have rewritten all of them to prove nothing.
    """
    if provider is None:
        raise ValueError("no provider")

    return (await provider.generate(prompt, temperature=temperature)).text
