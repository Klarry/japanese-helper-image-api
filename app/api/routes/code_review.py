"""Day 32: ``POST /code-review`` - an AI review of one pull request's diff.

Called by the GitHub Actions job (``app/github_review.py``) or by hand. The
request carries the diff; the service never reaches GitHub or the disk.

Who may call it: with ``CODE_REVIEW_TOKEN`` set, anyone presenting it as
``Authorization: Bearer <token>``; without it, only this machine (loopback).
The server listens on the network since Day 30, so an unset token must not
mean an open door to the model and to the code under review.

One review at a time per process: a review is the longest call this backend
makes, and a second one would only slow both down. The other answers are
413 (diff too large), 422 (not a reviewable request), 502 (no usable JSON
from the model after one repair) and 503 (busy, or the named model is
unavailable - never a review from the other provider).
"""

import hmac
import logging
import os
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas.code_review import CodeReviewError, CodeReviewRequest, CodeReviewResponse
from app.services import code_review
from app.services.code_review import (
    CodeReviewer,
    InvalidReviewOutput,
    PullRequest,
    ReviewRequestError,
    ReviewTooLarge,
)
from app.services.llm_provider import ProviderUnavailable
from app.services.local_chat_guard import ConcurrencyGate

router = APIRouter()
logger = logging.getLogger(__name__)

LOOPBACK = frozenset({"127.0.0.1", "::1"})
gate = ConcurrencyGate(1)


def reviewer_for(provider_name: str) -> CodeReviewer:
    return CodeReviewer(provider_name)


def _error(status: int, error: str, detail: str, request_id: str, **extra) -> JSONResponse:
    body = CodeReviewError(error=error, detail=detail, request_id=request_id, **extra)
    return JSONResponse(status_code=status, content=body.model_dump(), headers={"X-Request-ID": request_id})


def _refusal(request: Request, request_id: str) -> JSONResponse | None:
    """401/403, or None when the caller may review. The token is compared in
    constant time and never logged or echoed."""
    token = os.getenv("CODE_REVIEW_TOKEN", "").strip()

    if token:
        given = request.headers.get("authorization", "")

        if not hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
            return _error(401, "unauthorized", "a valid Authorization: Bearer token is required", request_id)

        return None

    host = request.client.host if request.client else ""

    if host not in LOOPBACK:
        return _error(
            403,
            "forbidden",
            "without CODE_REVIEW_TOKEN the review endpoint only accepts requests from this machine",
            request_id,
        )

    return None


@router.post(
    "/code-review",
    response_model=CodeReviewResponse,
    responses={code: {"model": CodeReviewError} for code in (401, 403, 413, 422, 502, 503)},
)
async def review_pull_request(body: CodeReviewRequest, request: Request):
    """Review a pull request's diff with RAG context from the project index.
    Returns validated findings (bugs, security, architecture, tests, ...),
    notes, recommendations and the limitations of the review."""
    request_id = uuid.uuid4().hex[:12]
    refused = _refusal(request, request_id)

    if refused is not None:
        return refused

    if len(body.diff) > code_review.MAX_DIFF_CHARS:
        return _error(
            413,
            "diff_too_large",
            f"the diff has {len(body.diff)} characters; the limit is {code_review.MAX_DIFF_CHARS}",
            request_id,
        )

    if not gate.try_enter():
        return _error(503, "busy", "a review is already running; try again when it finishes", request_id)

    try:
        reviewer = reviewer_for(body.provider)
        result = await reviewer.review(
            PullRequest(
                repository=body.repository,
                number=body.pull_request_number,
                base_branch=body.base_branch,
                head_branch=body.head_branch,
                title=body.title,
                diff=body.diff,
                changed_files=[item.model_dump() for item in body.changed_files],
            ),
            request_id=request_id,
        )
    except ReviewRequestError as error:
        return _error(422, "invalid_diff", str(error), request_id)
    except ReviewTooLarge as error:
        return _error(413, "diff_too_large", str(error), request_id)
    except InvalidReviewOutput as error:
        return _error(502, "invalid_model_output", str(error), request_id, provider=error.provider)
    except ProviderUnavailable as error:
        logger.warning("[CodeReview] request_id=%s provider=%s unavailable (%s) - no fallback", request_id, error.provider, error.reason)
        return _error(503, "provider_unavailable", error.message, request_id, provider=error.provider, reason=error.reason)
    finally:
        gate.leave()

    # Through the response model on purpose: the shape GitHub Actions reads
    # is checked here, not assumed.
    checked = CodeReviewResponse(**result.as_dict())

    return JSONResponse(content=checked.model_dump(), headers={"X-Request-ID": request_id})
