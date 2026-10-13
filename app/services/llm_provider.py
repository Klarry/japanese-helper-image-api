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

from app.services import ollama_profiles, ollama_service
from app.services.ollama_profiles import Profile
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
    #: Day 29: what the provider measured about the run (Ollama only -
    #: load, prompt and generation time, tokens per second, why it stopped).
    meta: dict | None = None

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

    Day 29: how it is asked is a ``Profile`` - options and prompt version
    together - and by default the one ``OLLAMA_PROFILE`` names, which is
    ``original`` unless set: the request Days 26-28 sent, unchanged.

    The per-call ``temperature`` is still accepted and ignored. The profile
    decides the sampling for all three calls a RAG answer makes, so that a
    profile measured in an evaluation is the profile that runs - not that
    profile with a different temperature on two calls out of three.
    """

    name = OLLAMA

    def __init__(self, model: str = "", profile: Profile | None = None) -> None:
        self.profile = profile or ollama_profiles.active_profile()
        self.model = model or self.profile.model or ollama_service.MODEL

    @property
    def prompt_version(self) -> str:
        """Which cited prompt this profile is meant to be handed."""
        return self.profile.prompt_version

    async def generate(self, prompt: str, temperature: float | None = None) -> Generated:
        options = self.profile.options

        try:
            if options.is_default:
                # Exactly the Day 26-28 call, so nothing about the original
                # profile depends on the new parameter existing.
                answer = await ollama_service.generate(prompt, model=self.model)
            else:
                answer = await ollama_service.generate(prompt, model=self.model, options=options)
        except ollama_service.OllamaUnavailable as error:
            # Keep answering while the laptop's model is down.
            logger.warning("Ollama unavailable (%s); answering with Gemini instead", error.reason)
            return await GeminiProvider().generate(prompt, temperature=temperature)

        return Generated(
            text=answer.response,
            model=answer.model,
            input_tokens=answer.prompt_tokens,
            output_tokens=answer.response_tokens,
            meta=answer.metrics() if hasattr(answer, "metrics") else None,
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
