"""Day 31: ``POST /dev/chat`` - the developer assistant over HTTP.

    {"message": "/help How does RAG work in this project?", "provider": "ollama"}

The command is in the message, as it would be typed in a chat. Anything that
is not /help is a 400 that says what is understood. A provider that cannot
answer is a 503, never an answer from the other one.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas.dev_assistant import DevChatError, DevChatRequest, DevChatResponse
from app.services import llm_provider
from app.services.dev_assistant import DevAssistant, NotAHelpCommand

router = APIRouter()
logger = logging.getLogger(__name__)


def assistant_for(provider_name: str) -> DevAssistant:
    """One per request, as /rag/chat builds its agent: the provider is per
    request, and the index behind the retriever is loaded once and shared."""
    return DevAssistant(provider_name)


@router.post(
    "/dev/chat",
    response_model=DevChatResponse,
    responses={400: {"model": DevChatError}, 503: {"model": DevChatError}},
)
async def dev_chat(request: DevChatRequest):
    """Answer a /help command from the project's documentation (RAG) and its
    live Git state (MCP get_git_branch)."""
    assistant = assistant_for(request.provider)

    try:
        answer = await assistant.ask(request.message)
    except NotAHelpCommand as error:
        body = DevChatError(error="Unknown command", detail=f"{error}. Try: /help")
        return JSONResponse(status_code=400, content=body.model_dump())
    except llm_provider.ProviderUnavailable as error:
        logger.warning("[DevAssistant] provider=%s unavailable (%s) - no fallback", error.provider, error.reason)
        body = DevChatError(
            error="The selected model is unavailable",
            detail=error.message,
            provider=error.provider,
            reason=error.reason,
        )
        return JSONResponse(status_code=503, content=body.model_dump())

    return DevChatResponse(**answer.as_dict())
