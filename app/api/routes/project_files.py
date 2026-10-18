"""Day 34: the project file assistant over HTTP.

    POST /project-files/task   {"task": "Find all usages of ... and explain the flow", "provider": "ollama"}
    POST /project-files/apply  {"path": "README.md", "proposal": "output/proposals/README.md",
                                "expected_sha256": "...", "confirm": false}

``/task`` answers with what the run did - files inspected, tool calls, the
report or proposal it saved, the diff - and a ``status`` that says how far it
got. The HTTP status says whether the service could run at all: 503 when the
project-files MCP server or the named model was unavailable and nothing was
produced, 200 otherwise (``partial`` and ``failed`` included). A model that
is down is never replaced by the other provider.

``/apply`` is the only way a project file outside the output folder changes,
and it changes nothing without ``confirm: true`` and the sha256 the proposal
was made against: 409 when the file changed since, 403 for a path the rules
protect.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas.project_files import (
    ApplyRequest,
    ApplyResponse,
    ProjectFilesError,
    ProjectTaskRequest,
    ProjectTaskResponse,
)
from app.services.file_assistant import FAILED, ApplyRefused, FileAssistant, apply_proposal
from app.services.llm_provider import ProviderUnavailable

router = APIRouter()
logger = logging.getLogger(__name__)

_UNAVAILABLE = {"mcp_unavailable", "provider_unavailable"}


def assistant_for(provider_name: str) -> FileAssistant:
    return FileAssistant(provider_name)


@router.post(
    "/project-files/task",
    response_model=ProjectTaskResponse,
    responses={503: {"model": ProjectFilesError}},
)
async def project_files_task(request: ProjectTaskRequest):
    """Carry out a task on the project's files: find usages of a component or
    API and explain the flow, or compare documentation with the code and
    propose a patch. The assistant chooses the files."""
    try:
        result = await assistant_for(request.provider).run(request.task)
    except ProviderUnavailable as error:
        logger.warning("[FileAssistant] provider=%s unavailable (%s) - no fallback", error.provider, error.reason)
        body = ProjectFilesError(error="provider_unavailable", detail=error.message, provider=error.provider, reason=error.reason)
        return JSONResponse(status_code=503, content=body.model_dump())

    checked = ProjectTaskResponse(**result.as_dict())
    unavailable = checked.status == FAILED and any(error.kind in _UNAVAILABLE for error in checked.errors)

    return JSONResponse(status_code=503 if unavailable else 200, content=checked.model_dump())


@router.post(
    "/project-files/apply",
    response_model=ApplyResponse,
    responses={code: {"model": ProjectFilesError} for code in (403, 404, 409, 422, 503)},
)
async def project_files_apply(request: ApplyRequest):
    """Show the change a saved proposal makes; with confirm, make it - only if
    the file still has the sha256 the proposal was made against."""
    try:
        result = await apply_proposal(request.path, request.proposal, request.expected_sha256, request.confirm)
    except ApplyRefused as error:
        logger.info("[FileAssistant] apply %s refused: %s", request.path, error.kind)
        return JSONResponse(status_code=error.status, content=ProjectFilesError(error=error.kind, detail=error.message).model_dump())

    return ApplyResponse(**result)
