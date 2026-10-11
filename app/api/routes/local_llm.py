"""Days 26-27: the endpoints that reach the model running on this machine.

``POST /local-llm`` (Day 26) is the demo: one prompt in, one answer out.
``POST /local-chat`` (Day 27) is what the Android app talks to, and the
difference is only who is asking - both go through the same
``ollama_service`` and neither touches Gemini.

Mounted beside the others and wired to nothing else: no agent, no RAG, no
MCP. In this mode the local model answers or nothing does. There is no
fallback to the cloud anywhere in this file, and a test fails if one appears.

Day 30 makes ``/local-chat`` a service other machines can use: a request id
and measured timings on every answer, an optional conversation kept by the
Day 7 history storage (one file per conversation), and the guards in
``local_chat_guard`` in front of the model - rate limit, input and context
limits, a cap on generations at once, a timeout. Every refusal uses the same
failure shape the app already reads, with its own reason and status.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import replace

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from app.schemas.local_llm import (
    LocalChatRequest,
    LocalChatResponse,
    LocalChatUnavailable,
    LocalLlmHealthResponse,
    LocalLlmRequest,
    LocalLlmResponse,
)
from app.services import ollama_profiles, ollama_service
from app.services.agent_history_storage import AgentHistoryStorage, format_transcript
from app.services.local_chat_guard import (
    ContextTooLarge,
    LocalChatGuard,
    LocalChatLimits,
    client_address,
    plan_prompt,
)

router = APIRouter()
logger = logging.getLogger(__name__)

#: The guards for this process. Built from the environment once; a test (or
#: an operator, through ``reset_guard``) can swap in a fresh set.
guard = LocalChatGuard.from_env()


def reset_guard(limits: LocalChatLimits | None = None) -> LocalChatGuard:
    """Fresh counters and, optionally, different limits."""
    global guard
    guard = LocalChatGuard(limits or LocalChatLimits.from_env())

    return guard

#: Which failure is the local server's fault (it is not running, the model
#: was never pulled, it took too long) and which is a bad answer from a
#: server that did reply. 503 says "try again once it is up"; 502 says "it
#: answered, and the answer was no use".
_STATUS = {
    "not_running": 503,
    "timeout": 503,
    "model_not_installed": 503,
    "http_error": 502,
    "empty_response": 502,
    # Day 29: the token cap ran out before an answer - a setting, not an outage.
    "truncated": 502,
}


@router.post("/local-llm")
async def local_llm(request: LocalLlmRequest) -> LocalLlmResponse:
    try:
        answer = await ollama_service.generate(request.prompt)
    except ollama_service.OllamaUnavailable as error:
        # The message is passed through exactly as the service wrote it,
        # newlines and shell commands included: whoever gets this response
        # is the person who can act on it.
        raise HTTPException(
            status_code=_STATUS.get(error.reason, 502),
            detail=error.message,
        ) from error

    return LocalLlmResponse(model=answer.model, response=answer.response)


def _service_options(limits: LocalChatLimits) -> tuple[str, ollama_service.OllamaOptions, int]:
    """The model, the options and the answer reserve the chat runs with.

    The Day 29 profile decides temperature, num_predict and thinking, and
    the context window too when it sets one; when it does not (``original``
    does not), the service sets it explicitly rather than inheriting
    whatever the server's default happens to be. The reserve is the
    profile's num_predict when there is one, so the budget and the cap on
    the answer are the same number.
    """
    profile = ollama_profiles.active_profile()
    options = profile.options

    if options.num_ctx is None:
        options = replace(options, num_ctx=limits.default_num_ctx)

    reserve = options.num_predict if options.num_predict and options.num_predict > 0 else limits.output_reserve_tokens

    return profile.resolved_model, options, reserve


@router.get("/local-llm/health")
async def local_llm_health(strict: bool = False):
    """200 with the verdict in the body - or, with ``?strict=true``, 503
    when the model cannot answer.

    The default stays as Day 27 made it, because the app reads the body: a
    health check that fails its own request for "the thing I check is down"
    looks the same as one that is broken. ``strict`` is for everything that
    reads only the status line - ``curl -f``, a monitor, a load balancer -
    and for those "FastAPI is up" must never pass for "the model is up".
    """
    model, options, _ = _service_options(guard.limits)
    report = await ollama_service.check_ollama_health(model)
    ready = bool(report.get("available") and report.get("model_installed"))
    body = LocalLlmHealthResponse(
        **report,
        ready=ready,
        service={**guard.status(), "options": options.request_fields()},
    )

    if strict and not ready:
        return JSONResponse(status_code=503, content=body.model_dump())

    return body


#: The one line the app shows when the local model cannot answer. Fixed
#: wording, so the screen does not have to parse a cause out of prose - the
#: cause travels beside it in ``reason`` and ``detail``.
UNAVAILABLE = "Local LLM is unavailable"


def _unavailable(error: ollama_service.OllamaUnavailable, request_id: str = "") -> JSONResponse:
    """Say no, and say why.

    Deliberately not a fallback. The whole point of this mode is that the
    answer came from this machine; an answer quietly fetched from Gemini
    instead would be a different claim wearing the same response shape.
    """
    body = LocalChatUnavailable(
        error=UNAVAILABLE,
        reason=error.reason,
        detail=error.message,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=_STATUS.get(error.reason, 502),
        content=body.model_dump(),
        headers={"X-Request-ID": request_id} if request_id else None,
    )


def _refuse(
    status: int, reason: str, error: str, detail: str, request_id: str, retry_after: int | None = None
) -> JSONResponse:
    """The service's own "no": same shape as an Ollama failure, own reason."""
    body = LocalChatUnavailable(
        error=error, reason=reason, detail=detail, request_id=request_id, retry_after=retry_after
    )
    headers = {"X-Request-ID": request_id}

    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)

    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def _log(request_id: str, client: str, model: str, outcome: str, status: int, started: float, **extra) -> None:
    """One line per request. Ids, sizes and times - never what was said."""
    details = " ".join(f"{key}={value}" for key, value in extra.items() if value is not None)
    logger.info(
        "[LocalService] request_id=%s endpoint=/local-chat client=%s model=%s outcome=%s status=%d total_ms=%d %s",
        request_id,
        client,
        model,
        outcome,
        status,
        _ms(time.perf_counter() - started),
        details,
    )


@router.post("/local-chat")
async def local_chat(payload: LocalChatRequest, request: Request, response: Response):
    """The app's chat (Day 27), answered by the model on this machine - and
    since Day 30, by a service other machines on the network can use.

    The checks run cheapest first and before the model is touched: who is
    asking and how often, how long the message is, whether it fits the
    context, whether this conversation already has a request running,
    whether there is a free generation slot. Only then is Ollama called,
    under a timeout, through the same service the CLI uses. Nothing here
    knows about Gemini, and nothing here would know how to call it.
    """
    started = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    current = guard
    limits = current.limits
    client = client_address(request, limits.trusted_proxies)
    model, options, reserve = _service_options(limits)
    conversation = payload.conversation_id

    decision = current.limiter.check(client)

    if not decision.allowed:
        _log(request_id, client, model, "rate_limited", 429, started)
        return _refuse(
            429,
            "rate_limited",
            "Too many requests",
            f"At most {limits.rate_limit} chat requests per {limits.rate_window_seconds:g}s from one address. "
            f"Try again in {decision.retry_after}s.",
            request_id,
            retry_after=decision.retry_after,
        )

    if len(payload.message) > limits.max_message_chars:
        _log(request_id, client, model, "input_too_large", 413, started, chars=len(payload.message))
        return _refuse(
            413,
            "input_too_large",
            "Message is too long",
            f"The message is {len(payload.message)} characters; the limit is {limits.max_message_chars}.",
            request_id,
        )

    storage = None
    history: list = []

    if conversation:
        storage = AgentHistoryStorage(str(limits.history_path / f"{conversation}.json"))
        state = storage.load()
        history = list(state.current().messages)

    try:
        plan = plan_prompt(
            payload.message,
            history,
            num_ctx=options.num_ctx,
            reserve=reserve,
            recent=limits.recent_messages,
            render=format_transcript,
        )
    except ContextTooLarge as error:
        _log(request_id, client, model, "context_too_large", 413, started, est_tokens=error.needed, budget=error.budget)
        return _refuse(
            413,
            "context_too_large",
            "Message does not fit the context window",
            f"The message alone is about {error.needed} tokens (estimated); with num_ctx={options.num_ctx} "
            f"and {reserve} kept for the answer, the prompt may use {error.budget}. "
            "It was refused rather than cut: the model would not have read all of it.",
            request_id,
        )

    if conversation and conversation in current.busy_conversations:
        _log(request_id, client, model, "conversation_busy", 409, started)
        return _refuse(
            409,
            "conversation_busy",
            "This conversation is already being answered",
            "Wait for the previous message in this conversation to be answered, then send the next.",
            request_id,
        )

    if not current.gate.try_enter():
        _log(request_id, client, model, "busy", 503, started, active=current.gate.active)
        return _refuse(
            503,
            "busy",
            "Local LLM is busy",
            f"{current.gate.active} of {current.gate.limit} generation slots are in use. Try again shortly.",
            request_id,
        )

    if conversation:
        current.busy_conversations.add(conversation)

    generation_started = time.perf_counter()

    try:
        answer = await asyncio.wait_for(
            ollama_service.generate(plan.prompt, model=model, options=options),
            timeout=limits.timeout_seconds,
        )
    except asyncio.TimeoutError:
        _log(request_id, client, model, "timeout", 503, started)
        return _refuse(
            503,
            "timeout",
            UNAVAILABLE,
            f"Generation did not finish within {limits.timeout_seconds:g}s "
            "(LOCAL_CHAT_TIMEOUT_SECONDS). The request was cancelled.",
            request_id,
        )
    except ollama_service.OllamaUnavailable as error:
        _log(request_id, client, model, error.reason, _STATUS.get(error.reason, 502), started)
        return _unavailable(error, request_id)
    finally:
        # Always, whatever happened above - a slot or a conversation that
        # leaked here would be a service that slowly stops answering.
        current.gate.leave()
        current.busy_conversations.discard(conversation or "")

    generation_ms = _ms(time.perf_counter() - generation_started)

    if storage is not None:
        moment = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        state.current().messages.extend(
            [
                {"role": "user", "content": payload.message, "timestamp": moment},
                {"role": "assistant", "content": answer.response, "timestamp": moment},
            ]
        )
        storage.save(state)

    total_ms = _ms(time.perf_counter() - started)
    response.headers["X-Request-ID"] = request_id
    _log(
        request_id,
        client,
        answer.model,
        "ok",
        200,
        started,
        generation_ms=generation_ms,
        prompt_tokens=answer.prompt_tokens,
        response_tokens=answer.response_tokens,
        history=plan.history_used,
    )

    return LocalChatResponse(
        response=answer.response,
        model=answer.model,
        provider=ollama_service.PROVIDER,
        seconds=answer.seconds,
        request_id=request_id,
        conversation_id=conversation,
        timing={"generation_ms": generation_ms, "total_ms": total_ms},
        context=plan.as_dict(),
        usage={
            "prompt_tokens": answer.prompt_tokens,
            "response_tokens": answer.response_tokens,
            "tokens_per_second": answer.tokens_per_second,
            "options": options.request_fields(),
        },
    )
