"""Day 35: the request and reply of ``POST /release/prepare``.

The reply is the run record - the same document the assistant writes as
``run.json`` - so what the API says and what is on disk cannot differ.
"""

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.llm_provider import OLLAMA, PROVIDERS

_REF = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./@^~{}+-]{0,199}$")
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class ReleasePrepareRequest(BaseModel):
    """What to prepare. Nothing here can publish: the assistant has no such step."""

    model_config = ConfigDict(extra="forbid")

    #: The start of the release range (a branch, tag or commit). Empty: no range is assumed.
    base_ref: str = Field(default="", max_length=200)
    provider: str = OLLAMA
    #: A repository named in data/release/release_config.json.
    repository: str = "backend"
    #: With a base_ref: also analyze the working tree's uncommitted changes.
    include_uncommitted: bool = False
    #: Run the configured checks (tests, static analysis). Off unless asked for.
    run_checks: bool = False
    #: Which checks, by id; empty means every check configured for the repository.
    checks: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("provider")
    @classmethod
    def provider_must_be_known(cls, value: str) -> str:
        chosen = (value or "").strip().lower()

        if chosen not in PROVIDERS:
            raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}, got {value!r}")

        return chosen

    @field_validator("base_ref")
    @classmethod
    def base_ref_is_a_reference(cls, value: str) -> str:
        text = value.strip()

        if text and (not _REF.match(text) or ".." in text or text.endswith(("/", ".lock"))):
            raise ValueError("base_ref must be a branch, tag or commit name - no spaces, ranges or options")

        return text

    @field_validator("repository")
    @classmethod
    def repository_is_a_name(cls, value: str) -> str:
        if not _NAME.match(value.strip()):
            raise ValueError("repository is the name of a configured repository, for example 'backend' or 'android'")

        return value.strip()

    @field_validator("checks")
    @classmethod
    def checks_are_ids(cls, value: list[str]) -> list[str]:
        if any(not _NAME.match(item) for item in value):
            raise ValueError("checks are ids such as 'backend-tests'")

        return value


class ReleaseError(BaseModel):
    error: str
    detail: str
    request_id: str = ""


class Readiness(BaseModel):
    verdict: str
    summary: str
    blockers: list[str]
    warnings: list[str]
    unverified: list[str]


class FileChange(BaseModel):
    path: str
    status: str
    additions: int
    deletions: int
    origin: str
    also_edited_uncommitted: bool = False
    user_facing: bool
    analysis: str
    withheld: str
    review: str


class Finding(BaseModel):
    id: str
    category: str
    severity: str
    kind: str
    file: str | None = None
    line: int | None = None
    title: str
    description: str
    recommendation: str = ""
    evidence: str = ""
    sources: list = Field(default_factory=list)
    source: str
    validation_status: str
    batch: int | None = None


class CheckOutcome(BaseModel):
    id: str
    title: str
    status: str
    command: str
    documented_in: str = ""
    exit_code: int | None = None
    seconds: float = 0.0
    reason: str = ""
    output_tail: str = ""
    errors: list[str] = Field(default_factory=list)


class Artifact(BaseModel):
    name: str
    path: str
    sha256: str
    bytes: int
    created: bool = True


class ReleaseResponse(BaseModel):
    """The whole run record. ``status``: completed, partial (something could not be done - see
    errors), failed (nothing was reviewed: the model was unavailable) or no_changes."""

    run_id: str
    status: str
    started_at: str
    finished_at: str
    seconds: float
    request: dict
    readiness: Readiness
    repository: dict
    git_status_at_start: dict
    scope: dict
    files: list[FileChange]
    other_changes: dict
    commits: dict
    inspection: dict
    documentation: dict
    review: dict
    findings: list[Finding]
    checks: list[CheckOutcome]
    tree_after_checks: dict
    draft: dict
    artifacts: list[Artifact]
    output_dir: str
    run_dir: str
    limitations: list[str]
    unresolved: list[str]
    errors: list[dict]
    tool_calls: list[dict]
    llm: dict
