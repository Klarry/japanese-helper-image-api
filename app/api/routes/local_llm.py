"""Days 26-27: the endpoints that reach the model running on this machine.

``POST /local-llm`` (Day 26) is the demo: one prompt in, one answer out.
``POST /local-chat`` (Day 27) is what the Android app talks to, and the
difference is only who is asking - both go through the same
``ollama_service`` and neither touches Gemini.

Mounted beside the others and wired to nothing else: no agent, no RAG, no
MCP, no conversation history. In this mode the local model answers or
nothing does. There is no fallback to the cloud anywhere in this file, and
a test fails if one appears.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.local_llm import (
    LocalChatRequest,
    LocalChatResponse,
    LocalChatUnavailable,
    LocalLlmHealthResponse,
    LocalLlmRequest,
    LocalLlmResponse,
)
from app.services import ollama_service

router = APIRouter()

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


@router.get("/local-llm/health")
async def local_llm_health() -> LocalLlmHealthResponse:
    """Always 200, even when Ollama is down.

    A health check that returns an error status for "the thing I check is
    unhealthy" is indistinguishable from one that is broken itself. The
    verdict is in the body.
    """
    return LocalLlmHealthResponse(**await ollama_service.check_ollama_health())


#: The one line the app shows when the local model cannot answer. Fixed
#: wording, so the screen does not have to parse a cause out of prose - the
#: cause travels beside it in ``reason`` and ``detail``.
UNAVAILABLE = "Local LLM is unavailable"


def _unavailable(error: ollama_service.OllamaUnavailable) -> JSONResponse:
    """Say no, and say why.

    Deliberately not a fallback. The whole point of this mode is that the
    answer came from this machine; an answer quietly fetched from Gemini
    instead would be a different claim wearing the same response shape.
    """
    body = LocalChatUnavailable(
        error=UNAVAILABLE,
        reason=error.reason,
        detail=error.message,
    )

    return JSONResponse(status_code=_STATUS.get(error.reason, 502), content=body.model_dump())


@router.post("/local-chat")
async def local_chat(request: LocalChatRequest) -> LocalChatResponse:
    """Day 27: the app's chat, answered by the model on this machine.

    The same service the CLI uses, called the same way. Nothing here knows
    about Gemini, and nothing here would know how to call it.
    """
    try:
        answer = await ollama_service.generate(request.message)
    except ollama_service.OllamaUnavailable as error:
        return _unavailable(error)

    return LocalChatResponse(
        response=answer.response,
        model=answer.model,
        provider=ollama_service.PROVIDER,
        seconds=answer.seconds,
    )
