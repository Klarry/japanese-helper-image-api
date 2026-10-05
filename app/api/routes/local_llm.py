"""Day 26: the demo endpoint for the local model.

Mounted beside the others and wired to nothing: no agent, no RAG, no MCP,
no conversation history. Its whole job is to prove that a model running on
this machine can be reached over HTTP.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.local_llm import (
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
