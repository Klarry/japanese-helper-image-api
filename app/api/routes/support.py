"""Day 33: ``POST /support/chat`` - the support assistant over HTTP.

    {"message": "Почему не работает авторизация?", "ticket_id": "TICKET-1042", "provider": "ollama"}

Answers 404 for a ticket the CRM does not have, 409 for a ticket whose user
cannot be verified, 422 for a malformed request, 503 when the CRM, the
help-centre index or the named model is unavailable - and never an answer
from the other provider, or about a ticket that was not found.

Access: this is a local demonstration on synthetic data. There is no
authentication model for end users in this backend; the ticket id is the only
key, the user is resolved from the ticket, and the server is meant for a
trusted network (Day 30). A production version would check that the caller
owns the ticket before anything is read.
"""

import logging
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas.support import SupportChatError, SupportChatRequest, SupportChatResponse
from app.services.llm_provider import ProviderUnavailable
from app.services.support_assistant import SupportAssistant, SupportError

router = APIRouter()
logger = logging.getLogger(__name__)


def assistant_for(provider_name: str) -> SupportAssistant:
    return SupportAssistant(provider_name)


def _error(status: int, body: SupportChatError) -> JSONResponse:
    headers = {"X-Request-ID": body.request_id} if body.request_id else None
    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)


@router.post(
    "/support/chat",
    response_model=SupportChatResponse,
    responses={code: {"model": SupportChatError} for code in (404, 409, 502, 503)},
)
async def support_chat(request: SupportChatRequest):
    """Answer a user's question about the product, with their ticket (via MCP)
    and the help centre (via RAG) as context."""
    request_id = uuid.uuid4().hex[:12]

    try:
        answer = await assistant_for(request.provider).ask(request.message, request.ticket_id, request_id)
    except SupportError as error:
        logger.info("[Support] request_id=%s ticket=%s refused: %s", request_id, request.ticket_id, error.kind)
        return _error(
            error.status or 502,
            SupportChatError(error=error.kind, detail=error.message, ticket_id=request.ticket_id, request_id=request_id),
        )
    except ProviderUnavailable as error:
        logger.warning("[Support] request_id=%s provider=%s unavailable (%s) - no fallback", request_id, error.provider, error.reason)
        return _error(
            503,
            SupportChatError(
                error="provider_unavailable",
                detail=error.message,
                ticket_id=request.ticket_id,
                request_id=request_id,
                provider=error.provider,
                reason=error.reason,
            ),
        )

    checked = SupportChatResponse(**answer.as_dict())

    return JSONResponse(content=checked.model_dump(), headers={"X-Request-ID": request_id})
