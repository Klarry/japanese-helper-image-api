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
"""

import logging
import os
import re
import time
from dataclasses import dataclass

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

GENERATE_PATH = "/api/generate"
TAGS_PATH = "/api/tags"

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

    def as_dict(self) -> dict[str, str]:
        """Exactly the two fields the service contract promises."""
        return {"response": self.response, "model": self.model}


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


async def generate(prompt: str, model: str = "") -> LocalAnswer:
    """Ask the local model one question and wait for the whole answer.

    ``stream`` is false because this is a request/response service, not a
    terminal: there is nobody to watch tokens arrive, and a caller that got
    half an answer could not tell it from a whole one.
    """
    wanted = model or MODEL
    payload = {"model": wanted, "prompt": prompt, "stream": False}
    logger.info("Calling Ollama model=%s at %s", wanted, BASE_URL)
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

    if not answer:
        # A model that thought and then said nothing is still an empty
        # answer: the caller asked for a response, and there isn't one.
        logger.warning("Ollama returned an empty response for model %s", wanted)
        raise _empty(wanted)

    return LocalAnswer(
        response=answer,
        model=str(data.get("model") or wanted),
        thinking=thinking.strip(),
        seconds=round(seconds, 2),
        prompt_tokens=_count(data.get("prompt_eval_count")),
        response_tokens=_count(data.get("eval_count")),
    )


def _count(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None

    return value
