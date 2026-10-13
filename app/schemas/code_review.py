"""Day 32: the request and reply of ``POST /code-review``."""

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.code_review_diff import DiffError, safe_path
from app.services.llm_provider import OLLAMA, PROVIDERS

_REPOSITORY = r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$"
_REF = re.compile(r"^[A-Za-z0-9._/\-]{1,255}$")


class ChangedFile(BaseModel):
    """One entry of the PR's file list. Only a label: nothing is read from disk."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=300)
    status: str = "modified"
    additions: int | None = Field(default=None, ge=0)
    deletions: int | None = Field(default=None, ge=0)

    @field_validator("path")
    @classmethod
    def path_is_relative(cls, value: str) -> str:
        try:
            return safe_path(value)
        except DiffError as error:
            raise ValueError(str(error)) from error

    @field_validator("status")
    @classmethod
    def status_is_known(cls, value: str) -> str:
        known = ("added", "modified", "removed", "renamed", "copied", "changed", "unchanged")

        if value not in known:
            raise ValueError(f"status must be one of {', '.join(known)}")

        return value


class CodeReviewRequest(BaseModel):
    """A pull request as GitHub Actions sends it: metadata plus the unified
    diff. The diff is the whole input - the service never fetches anything
    from GitHub or the filesystem."""

    model_config = ConfigDict(extra="forbid")

    repository: str = Field(pattern=_REPOSITORY)
    pull_request_number: int = Field(ge=1, le=100_000_000)
    base_branch: str
    head_branch: str
    title: str = Field(default="", max_length=300)
    diff: str = Field(min_length=1)
    changed_files: list[ChangedFile] = Field(default_factory=list, max_length=300)
    provider: str = OLLAMA

    @field_validator("base_branch", "head_branch")
    @classmethod
    def ref_is_plain(cls, value: str) -> str:
        if not _REF.match(value or "") or ".." in value or value.startswith(("/", "-")):
            raise ValueError("not a plain branch name")

        return value

    @field_validator("provider")
    @classmethod
    def provider_must_be_known(cls, value: str) -> str:
        chosen = (value or "").strip().lower()

        if chosen not in PROVIDERS:
            raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}, got {value!r}")

        return chosen


class CodeReviewSource(BaseModel):
    id: str
    file: str
    section: str
    chunk_id: str


class CodeReviewFinding(BaseModel):
    category: str
    severity: str
    kind: str
    file: str
    line: int | None = None
    title: str
    description: str
    recommendation: str = ""
    evidence: str
    sources: list[CodeReviewSource] = Field(default_factory=list)


class CodeReviewResponse(BaseModel):
    """``findings`` are validated against the diff; ``validation`` says what
    the model proposed and what was rejected or adjusted, and why."""

    request_id: str
    repository: str
    pull_request_number: int
    base_branch: str
    head_branch: str
    summary: str
    findings: list[CodeReviewFinding]
    architecture_notes: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provider: str
    model: str
    files: list[dict] = Field(default_factory=list)
    context: dict = Field(default_factory=dict)
    validation: dict = Field(default_factory=dict)
    generation: dict = Field(default_factory=dict)
    timing: dict = Field(default_factory=dict)


class CodeReviewError(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
    provider: str | None = None
    reason: str | None = None
