"""Day 26: the local LLM, reached over Ollama's HTTP API.

A second provider standing *beside* Gemini, not in front of it. Nothing in
the agent, the RAG pipeline, the MCP tools or the mini chat calls into this
module: it has one endpoint (``POST /local-llm``) and one CLI
(``python -m app.local_llm``), and that is the whole of its reach into the
project. Swapping the agent over to a local model is a later day's
decision, and keeping the two providers apart is what leaves that decision
open.

Settings are read from the environment here rather than imported from
``app.core.config``, for the reason ``rag_settings`` and
``embedding_service`` do the same: config requires GEMINI_API_KEY at import
time, and the point of a local model is that it runs with no cloud key at
all. The names and the defaults are the ones config declares.

Every failure mode gets its own message, written for someone reading a
terminal rather than a log: not running, model not pulled, timed out, an
HTTP error from the server, and a reply with nothing in it. None of them
are swallowed - the caller either gets text or gets told exactly what
happened.

Day 29 adds the knobs. ``generate`` takes an optional ``OllamaOptions`` -
temperature, num_predict, num_ctx and whether a thinking model may think -
and sends only the ones that are set. With none set the request is exactly
the one Days 26-28 sent, byte for byte, so "the original configuration" is
not a reconstruction: it is this function called the old way. The service
also reports what Ollama measured about each run (load, prompt and
generation time, tokens, why it stopped), because an optimisation claimed
without those numbers is a guess.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

#: Where Ollama listens. The default is its own: it binds to localhost, and
#: the backend talks to it over the loopback interface like any other local
#: service.
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
#: Which model answers. A name, not a code path - nothing here knows or
#: cares which family it belongs to.
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
#: Generous on purpose. A hosted API answers in seconds; a local model on a
#: laptop takes longer, and the first call after a restart also has to read
#: several gigabytes of weights off disk before the first token exists.
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))

#: The name of this mode, as the API reports it and the app displays it
#: (Day 27). Not configurable: it is not a setting, it is which half of the
#: project you are in. "gemini" is the other half, and nothing switches
#: between them on its own - there is no fallback in either direction.
PROVIDER = "ollama"

GENERATE_PATH = "/api/generate"
TAGS_PATH = "/api/tags"
SHOW_PATH = "/api/show"
PS_PATH = "/api/ps"


@dataclass(frozen=True)
class OllamaOptions:
    """How one request asks the model to generate (Day 29).

    Every field is optional, and a field left as None is not sent at all -
    the model's own default applies, which is exactly what happened before
    this class existed. Names are Ollama's: ``num_predict`` is the cap on
    generated tokens (thinking included, for a thinking model), ``num_ctx``
    is the context window the runner allocates, and ``think`` is a
    top-level request field rather than a sampling option.
    """

    temperature: float | None = None
    num_predict: int | None = None
    num_ctx: int | None = None
    think: bool | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")

        if self.num_predict is not None and self.num_predict != -1 and self.num_predict < 1:
            raise ValueError("num_predict must be positive, or -1 for no limit")

        if self.num_ctx is not None and self.num_ctx < 256:
            raise ValueError("num_ctx must be at least 256 tokens")

    @property
    def is_default(self) -> bool:
        """Nothing set: the request goes out exactly as Day 26 built it."""
        return self.request_fields() == {}

    def request_fields(self) -> dict:
        """The fields to add to a /api/generate body - and only those."""
        sampling = {
            name: value
            for name, value in (
                ("temperature", self.temperature),
                ("num_predict", self.num_predict),
                ("num_ctx", self.num_ctx),
            )
            if value is not None
        }
        fields: dict = {}

        if sampling:
            fields["options"] = sampling

        if self.think is not None:
            fields["think"] = self.think

        return fields

    def as_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
            "think": self.think,
        }


#: What Days 26-28 sent: nothing beyond model, prompt and stream.
DEFAULT_OPTIONS = OllamaOptions()

#: Thinking models (qwen3 among them) narrate before they answer. Newer
#: Ollama puts that narration in its own ``thinking`` field; older builds
#: leave it inline in these tags. Either way it is separated from the
#: answer rather than printed as part of it - and kept, not dropped, since
#: it is a real part of what the model produced.
_THINK_TAGS = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


class OllamaUnavailable(Exception):
    """Ollama could not be reached, or could not answer.

    One exception for every way this can go wrong, carrying a message that
    says which way it was and what to do about it. ``reason`` is the short
    machine-readable form, for a caller that has to map this onto a status
    code.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


def _not_running() -> OllamaUnavailable:
    return OllamaUnavailable(
        "not_running",
        f"Ollama is not running at {BASE_URL}.\n"
        "Start it with:\n\n"
        "    ollama serve\n\n"
        "or open the Ollama app.",
    )


def _timed_out(seconds: float) -> OllamaUnavailable:
    return OllamaUnavailable(
        "timeout",
        f"Ollama did not answer within {seconds:g}s.\n"
        "The first call after a restart also loads the model into memory, which\n"
        "can take a while. Try again, or raise OLLAMA_TIMEOUT.",
    )


def _model_missing(model: str) -> OllamaUnavailable:
    return OllamaUnavailable(
        "model_not_installed",
        f"The model '{model}' is not installed.\n"
        "Pull it with:\n\n"
        f"    ollama pull {model}\n\n"
        "or set OLLAMA_MODEL to one that is (ollama list).",
    )


def _http_error(status: int, body: str) -> OllamaUnavailable:
    return OllamaUnavailable(
        "http_error",
        f"Ollama returned HTTP {status}: {body.strip() or '(no body)'}",
    )


def _empty(model: str) -> OllamaUnavailable:
    return OllamaUnavailable(
        "empty_response",
        f"Ollama returned an empty response for model '{model}'.",
    )


def _truncated(model: str, limit: int | None) -> OllamaUnavailable:
    cap = f"num_predict={limit}" if limit is not None else "the token limit"
    return OllamaUnavailable(
        "truncated",
        f"'{model}' used all of {cap} before it produced an answer.\n"
        "A thinking model spends that budget on reasoning first - raise num_predict,\n"
        "or turn thinking off for this profile.",
    )


def _client() -> httpx.AsyncClient:
    """The HTTP client every call in this module uses.

    Its own function so a test can hand back a client on a mock transport:
    that way the tests exercise the real request this module builds, rather
    than a stand-in for it, without a socket ever being opened.
    """
    return httpx.AsyncClient(timeout=TIMEOUT)


def split_thinking(text: str) -> tuple[str, str]:
    """Separate a thinking model's narration from its answer.

    Returns ``(thinking, answer)``. Text with no ``<think>`` tags comes back
    unchanged as the answer, which is what a non-thinking model produces.
    """
    thinking = "\n".join(match.strip() for match in _THINK_TAGS.findall(text))
    answer = _THINK_TAGS.sub("", text).strip()

    # A thinking model stopped by num_predict halfway through its
    # reasoning leaves an opening tag and no closing one. What follows the
    # tag is still reasoning, and showing it as the answer would be worse
    # than showing nothing (Day 29).
    if answer.lower().startswith("<think>"):
        unfinished = answer[len("<think>") :].strip()
        thinking = "\n".join(part for part in (thinking, unfinished) if part)
        answer = ""

    return thinking, answer


@dataclass(frozen=True)
class LocalAnswer:
    """What the local model produced, and what it cost to produce it.

    ``response`` and ``model`` are what the assignment asks the service to
    return, and what the HTTP endpoint sends on. The rest is what Ollama
    reports about the run - kept because it is the only evidence that the
    generation happened here rather than in a datacentre.
    """

    response: str
    model: str
    thinking: str = ""
    seconds: float = 0.0
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    #: Day 29: what Ollama itself measured, in milliseconds. None when the
    #: server did not report it.
    load_ms: int | None = None
    prompt_eval_ms: int | None = None
    eval_ms: int | None = None
    ollama_total_ms: int | None = None
    #: "stop" when the model finished, "length" when num_predict cut it off.
    done_reason: str = ""
    options: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, str]:
        """Exactly the two fields the service contract promises."""
        return {"response": self.response, "model": self.model}

    @property
    def tokens_per_second(self) -> float | None:
        """Generation speed as Ollama measured it - generated tokens over
        the time spent generating them, loading and prompt excluded."""
        if not self.response_tokens or not self.eval_ms:
            return None

        return round(self.response_tokens / (self.eval_ms / 1000), 2)

    def metrics(self) -> dict:
        """Everything about the run except the text, for an evaluation."""
        return {
            "model": self.model,
            "seconds": self.seconds,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "load_ms": self.load_ms,
            "prompt_eval_ms": self.prompt_eval_ms,
            "eval_ms": self.eval_ms,
            "ollama_total_ms": self.ollama_total_ms,
            "tokens_per_second": self.tokens_per_second,
            "done_reason": self.done_reason,
            "thinking_chars": len(self.thinking),
            "options": self.options,
        }


async def check_ollama_health(model: str = "") -> dict:
    """Is Ollama up, and is the model we would ask for actually installed?

    Answers rather than raises: a health check that throws is no use to the
    thing asking whether it is safe to call. Unreachable comes back as
    ``available: false`` with the real error in it, not a swallowed one.
    """
    wanted = model or MODEL

    try:
        async with _client() as client:
            response = await client.get(f"{BASE_URL}{TAGS_PATH}")
    except httpx.TimeoutException:
        return {"available": False, "model": wanted, "error": _timed_out(TIMEOUT).message}
    except httpx.RequestError:
        return {"available": False, "model": wanted, "error": _not_running().message}

    if response.status_code != 200:
        return {
            "available": False,
            "model": wanted,
            "error": _http_error(response.status_code, response.text).message,
        }

    try:
        payload = response.json()
    except ValueError:
        return {
            "available": False,
            "model": wanted,
            "error": f"Ollama answered {TAGS_PATH} with something that is not JSON.",
        }

    installed = sorted(
        entry["name"]
        for entry in payload.get("models", [])
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    )

    return {
        "available": True,
        "model": wanted,
        "model_installed": wanted in installed,
        "models": installed,
    }


async def generate(
    prompt: str, model: str = "", options: OllamaOptions | None = None
) -> LocalAnswer:
    """Ask the local model one question and wait for the whole answer.

    ``stream`` is false because this is a request/response service, not a
    terminal: there is nobody to watch tokens arrive, and a caller that got
    half an answer could not tell it from a whole one.

    ``options`` (Day 29) adds only the fields that are set. Without it the
    body is the Day 26 one, unchanged.
    """
    wanted = model or MODEL
    chosen = options or DEFAULT_OPTIONS
    payload = {"model": wanted, "prompt": prompt, "stream": False, **chosen.request_fields()}
    # Labelled on purpose, and here rather than in a route: every caller -
    # the endpoint, the chat endpoint and the CLI - goes through this
    # function, so a log that says [LocalLLM] is proof the local path ran
    # and not something that had to be remembered at four call sites.
    logger.info("[LocalLLM] Provider: %s", PROVIDER)
    logger.info("[LocalLLM] Model: %s", wanted)

    if not chosen.is_default:
        logger.info("[LocalLLM] Options: %s", chosen.request_fields())

    logger.info("[LocalLLM] Request sent to %s%s", BASE_URL, GENERATE_PATH)
    started = time.monotonic()

    try:
        async with _client() as client:
            response = await client.post(f"{BASE_URL}{GENERATE_PATH}", json=payload)
    except httpx.TimeoutException as error:
        logger.error("Ollama timed out after %.1fs: %s", TIMEOUT, error)
        raise _timed_out(TIMEOUT) from error
    except httpx.RequestError as error:
        logger.error("Ollama is not reachable at %s: %s", BASE_URL, error)
        raise _not_running() from error

    seconds = time.monotonic() - started

    if response.status_code == 404:
        # Ollama answers a request for a model it does not have with a 404
        # and a body saying so. A 404 from this endpoint cannot mean
        # anything else - the path is fixed and we just built it.
        logger.error("Ollama does not have model %s: %s", wanted, response.text)
        raise _model_missing(wanted)

    if response.status_code != 200:
        logger.error("Ollama returned status %s: %s", response.status_code, response.text)
        raise _http_error(response.status_code, response.text)

    try:
        data = response.json()
    except ValueError as error:
        raise _http_error(response.status_code, response.text) from error

    inline_thinking, answer = split_thinking(str(data.get("response") or ""))
    thinking = str(data.get("thinking") or "") or inline_thinking
    done_reason = str(data.get("done_reason") or "")

    if not answer:
        # A model that thought and then said nothing is still an empty
        # answer: the caller asked for a response, and there isn't one.
        # When the token cap is why, the message says so (Day 29) - "empty"
        # would send someone looking for a bug that is really a setting.
        logger.warning(
            "Ollama returned an empty response for model %s (done_reason=%s)",
            wanted,
            done_reason or "-",
        )

        if done_reason == "length":
            raise _truncated(wanted, chosen.num_predict)

        raise _empty(wanted)

    logger.info("[LocalLLM] Response received in %.2fs", seconds)

    return LocalAnswer(
        response=answer,
        model=str(data.get("model") or wanted),
        thinking=thinking.strip(),
        seconds=round(seconds, 2),
        prompt_tokens=_count(data.get("prompt_eval_count")),
        response_tokens=_count(data.get("eval_count")),
        load_ms=_ms(data.get("load_duration")),
        prompt_eval_ms=_ms(data.get("prompt_eval_duration")),
        eval_ms=_ms(data.get("eval_duration")),
        ollama_total_ms=_ms(data.get("total_duration")),
        done_reason=done_reason,
        options=chosen.as_dict(),
    )


def _count(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None

    return value


def _ms(nanoseconds) -> int | None:
    """Ollama reports durations in nanoseconds."""
    count = _count(nanoseconds)

    return None if count is None else int(round(count / 1_000_000))


# --- Day 29: what the evaluation needs to know about the model -------------
#
# Read-only questions to the same server, through the same client. None of
# them is used by a request path; they exist so that the size, quantisation
# and memory of a model are read from Ollama rather than written down.


async def show_model(model: str = "") -> dict:
    """What Ollama knows about one installed model: family, size label,
    quantisation, its default parameters and its context length."""
    wanted = model or MODEL

    try:
        async with _client() as client:
            response = await client.post(f"{BASE_URL}{SHOW_PATH}", json={"model": wanted})
    except httpx.TimeoutException as error:
        raise _timed_out(TIMEOUT) from error
    except httpx.RequestError as error:
        raise _not_running() from error

    if response.status_code == 404:
        raise _model_missing(wanted)

    if response.status_code != 200:
        raise _http_error(response.status_code, response.text)

    data = response.json()
    info = data.get("model_info") or {}
    context = next(
        (value for key, value in info.items() if key.endswith(".context_length")),
        None,
    )

    return {
        "model": wanted,
        "details": data.get("details") or {},
        "parameters": data.get("parameters") or "",
        "capabilities": data.get("capabilities") or [],
        "context_length": context,
        "basename": info.get("general.basename") or "",
        "finetune": info.get("general.finetune") or "",
        "size_label": info.get("general.size_label") or "",
    }


async def running_models() -> list[dict]:
    """The models Ollama currently holds in memory, with how much each
    takes (``size``) and how much of that is on the GPU (``size_vram``)."""
    try:
        async with _client() as client:
            response = await client.get(f"{BASE_URL}{PS_PATH}")
    except httpx.TimeoutException as error:
        raise _timed_out(TIMEOUT) from error
    except httpx.RequestError as error:
        raise _not_running() from error

    if response.status_code != 200:
        raise _http_error(response.status_code, response.text)

    return [entry for entry in response.json().get("models", []) if isinstance(entry, dict)]


async def unload_model(model: str = "") -> None:
    """Ask Ollama to drop a model from memory now, so that the next request
    pays the full load - the only honest way to measure a cold start."""
    wanted = model or MODEL

    try:
        async with _client() as client:
            response = await client.post(
                f"{BASE_URL}{GENERATE_PATH}", json={"model": wanted, "keep_alive": 0}
            )
    except httpx.TimeoutException as error:
        raise _timed_out(TIMEOUT) from error
    except httpx.RequestError as error:
        raise _not_running() from error

    if response.status_code == 404:
        raise _model_missing(wanted)

    if response.status_code != 200:
        raise _http_error(response.status_code, response.text)


async def server_version() -> str:
    """The Ollama build answering - recorded beside every measurement,
    because the same model on a different build is a different runtime."""
    try:
        async with _client() as client:
            response = await client.get(f"{BASE_URL}/api/version")
    except httpx.TimeoutException as error:
        raise _timed_out(TIMEOUT) from error
    except httpx.RequestError as error:
        raise _not_running() from error

    if response.status_code != 200:
        return ""

    return str(response.json().get("version") or "")


async def list_models() -> list[dict]:
    """Every installed model as Ollama lists it, size on disk included."""
    try:
        async with _client() as client:
            response = await client.get(f"{BASE_URL}{TAGS_PATH}")
    except httpx.TimeoutException as error:
        raise _timed_out(TIMEOUT) from error
    except httpx.RequestError as error:
        raise _not_running() from error

    if response.status_code != 200:
        raise _http_error(response.status_code, response.text)

    return [entry for entry in response.json().get("models", []) if isinstance(entry, dict)]
