"""Day 29: named ways of driving the local model.

A profile is three things that only mean something together: which model,
how it is asked to generate, and which prompt it is handed for a cited RAG
answer. They are named so that an evaluation, a log line and a setting can
all point at the same one:

    original          what Days 26-28 sent - no options at all, so the
                      model's own defaults apply - with the Day 24 prompt
    baseline          the assignment's reference point: temperature 0.7,
                      num_predict 1024, num_ctx 4096, the Day 24 prompt
    optimized-params  the optimised numbers with the Day 24 prompt, so the
                      effect of the parameters can be told apart from the
                      effect of the prompt
    optimized         temperature 0.2, num_predict 512, num_ctx 4096,
                      thinking off, and the Day 29 prompt

None of these numbers is claimed to be best. They are candidates, and
``python -m app.evaluate_day29`` is what says how each one did.

The running service uses ``OLLAMA_PROFILE`` and defaults to ``original``:
the model that works keeps working exactly as it did until a measured
comparison says otherwise. ``OLLAMA_TEMPERATURE``, ``OLLAMA_NUM_PREDICT``,
``OLLAMA_NUM_CTX`` and ``OLLAMA_THINK`` override single fields of whichever
profile is active. Gemini has no profile and is not touched by any of this.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from app.services import ollama_service
from app.services.ollama_service import OllamaOptions
from app.services.rag_prompt import PROMPT_V1, PROMPT_V2, PROMPT_VERSIONS

ORIGINAL_NAME = "original"
BASELINE_NAME = "baseline"
OPTIMIZED_PARAMS_NAME = "optimized-params"
OPTIMIZED_NAME = "optimized"

#: Which profile the service runs when nothing says otherwise.
DEFAULT_PROFILE = ORIGINAL_NAME


@dataclass(frozen=True)
class Profile:
    """One complete way of asking the local model."""

    name: str
    options: OllamaOptions = field(default_factory=OllamaOptions)
    prompt_version: str = PROMPT_V1
    #: Empty means OLLAMA_MODEL - the model the project is configured with.
    model: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if self.prompt_version not in PROMPT_VERSIONS:
            raise ValueError(
                f"unknown prompt version {self.prompt_version!r} "
                f"(expected one of {', '.join(PROMPT_VERSIONS)})"
            )

    @property
    def resolved_model(self) -> str:
        return self.model or ollama_service.MODEL

    def on_model(self, model: str) -> "Profile":
        """The same profile on a different model - same options, same
        prompt - which is how a quantisation is compared fairly."""
        return replace(self, model=model, name=f"{self.name}@{model}")

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.resolved_model,
            "prompt_version": self.prompt_version,
            **self.options.as_dict(),
            "description": self.description,
        }


ORIGINAL = Profile(
    ORIGINAL_NAME,
    OllamaOptions(),
    PROMPT_V1,
    description="Days 26-28 as they ran: no options sent, the model's own defaults, Day 24 prompt",
)
BASELINE = Profile(
    BASELINE_NAME,
    OllamaOptions(temperature=0.7, num_predict=1024, num_ctx=4096),
    PROMPT_V1,
    description="the reference point: 0.7 / 1024 / 4096, Day 24 prompt",
)
OPTIMIZED_PARAMS = Profile(
    OPTIMIZED_PARAMS_NAME,
    OllamaOptions(temperature=0.2, num_predict=512, num_ctx=4096, think=False),
    PROMPT_V1,
    description="optimised parameters, Day 24 prompt - isolates what the parameters do",
)
OPTIMIZED = Profile(
    OPTIMIZED_NAME,
    OllamaOptions(temperature=0.2, num_predict=512, num_ctx=4096, think=False),
    PROMPT_V2,
    description="optimised parameters and the Day 29 prompt",
)

PROFILES: dict[str, Profile] = {
    profile.name: profile for profile in (ORIGINAL, BASELINE, OPTIMIZED_PARAMS, OPTIMIZED)
}


def profile_for(name: str) -> Profile:
    """A profile by name. Unknown names are refused, not defaulted."""
    chosen = (name or "").strip().lower()

    try:
        return PROFILES[chosen]
    except KeyError:
        raise ValueError(
            f"unknown Ollama profile: {name!r} (expected one of {', '.join(PROFILES)})"
        ) from None


def _number(environ: Mapping[str, str], key: str, cast):
    raw = (environ.get(key) or "").strip()

    if not raw:
        return None

    try:
        return cast(raw)
    except ValueError:
        raise ValueError(f"{key}={raw!r} is not a valid {cast.__name__}") from None


def _flag(environ: Mapping[str, str], key: str) -> bool | None:
    raw = (environ.get(key) or "").strip().lower()

    if not raw:
        return None

    if raw in {"1", "true", "yes", "on"}:
        return True

    if raw in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"{key}={raw!r} is not true or false")


def active_profile(environ: Mapping[str, str] | None = None) -> Profile:
    """The profile the service runs, from the environment.

    Read when asked rather than at import, so a changed setting takes effect
    on the next request without a restart of anything but the reader.
    """
    env = os.environ if environ is None else environ
    profile = profile_for(env.get("OLLAMA_PROFILE") or DEFAULT_PROFILE)
    overrides = {
        name: value
        for name, value in (
            ("temperature", _number(env, "OLLAMA_TEMPERATURE", float)),
            ("num_predict", _number(env, "OLLAMA_NUM_PREDICT", int)),
            ("num_ctx", _number(env, "OLLAMA_NUM_CTX", int)),
            ("think", _flag(env, "OLLAMA_THINK")),
        )
        if value is not None
    }

    if not overrides:
        return profile

    return replace(
        profile,
        name=f"{profile.name}+env",
        options=replace(profile.options, **overrides),
    )
