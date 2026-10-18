"""Day 35: ``POST /release/prepare`` - prepare a release *draft* for a repository.

    {"base_ref": "main", "provider": "ollama"}

The assistant inspects the repository through the Git and file MCP servers,
reviews the change set with the Day 32 code review (and its RAG context),
optionally runs the checks a person configured, drafts a changelog and release
notes the change set can back up, and saves ``release_report.md``,
``CHANGELOG.md``, ``release_notes.md``, ``review_findings.json`` and ``run.json``
into ``output/release/<run_id>/``. It never commits, tags, builds a release or
publishes - there is no such step to call.

Who may call it: the same rule as ``/code-review`` (``CODE_REVIEW_TOKEN`` as a
bearer token, or this machine only) - the endpoint reads source diffs and, when
asked, runs the project's tests.

The answers: 200 with the run record (``completed``, ``partial``, ``no_changes``);
401/403 (not allowed); 422 (an unknown repository, base reference or check, an
output folder the rules refuse); 503 (busy, an MCP server that will not start, or
the named model unavailable - never a draft from the other provider; a run whose
model was down is still recorded and returned with its errors).
"""

import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.routes.code_review import _refusal
from app.schemas.release import ReleaseError, ReleasePrepareRequest, ReleaseResponse
from app.services.local_chat_guard import ConcurrencyGate
from app.services.release_assistant import FAILED, ReleaseAssistant, ReleaseError as RefusedRequest, ReleaseRequest

router = APIRouter()
logger = logging.getLogger(__name__)

gate = ConcurrencyGate(1)

_UNAVAILABLE = {"mcp_unavailable"}


def assistant_for(provider_name: str) -> ReleaseAssistant:
    return ReleaseAssistant(provider_name)


def _error(status: int, error: str, detail: str, request_id: str) -> JSONResponse:
    body = ReleaseError(error=error, detail=detail, request_id=request_id)
    return JSONResponse(status_code=status, content=body.model_dump(), headers={"X-Request-ID": request_id})


@router.post(
    "/release/prepare",
    response_model=ReleaseResponse,
    responses={code: {"model": ReleaseError} for code in (401, 403, 422, 503)},
)
async def prepare_release(body: ReleasePrepareRequest, request: Request):
    """Analyze the repository's Git changes, review them, compare them with the documentation and
    draft a changelog, release notes and a release report. Read-only: nothing is committed, built or published."""
    request_id = uuid.uuid4().hex[:12]
    refused = _refusal(request, request_id)

    if refused is not None:
        return refused

    if not gate.try_enter():
        return _error(503, "busy", "a release preparation is already running; try again when it finishes", request_id)

    try:
        run = await assistant_for(body.provider).prepare(
            ReleaseRequest(
                base_ref=body.base_ref,
                provider=body.provider,
                repository=body.repository,
                include_uncommitted=body.include_uncommitted,
                run_checks=body.run_checks,
                checks=tuple(body.checks),
            )
        )
    except RefusedRequest as error:
        status = 503 if error.kind in _UNAVAILABLE else 422
        logger.info("[ReleaseAssistant] request %s refused: %s", request_id, error.kind)
        return _error(status, error.kind, error.message, request_id)
    finally:
        gate.leave()

    checked = ReleaseResponse(**run.as_dict())
    unavailable = run.status == FAILED and any(item["kind"] == "provider_unavailable" for item in run.errors)

    return JSONResponse(status_code=503 if unavailable else 200, content=checked.model_dump(), headers={"X-Request-ID": request_id})
