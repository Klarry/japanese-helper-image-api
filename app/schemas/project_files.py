"""Day 34: the request and reply of ``POST /project-files/task`` and
``POST /project-files/apply``."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.llm_provider import OLLAMA, PROVIDERS


class ProjectTaskRequest(BaseModel):
    """A goal, in words. No file names are needed: the assistant finds them."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=2000)
    provider: str = OLLAMA

    @field_validator("task")
    @classmethod
    def task_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task is empty")

        return value

    @field_validator("provider")
    @classmethod
    def provider_must_be_known(cls, value: str) -> str:
        chosen = (value or "").strip().lower()

        if chosen not in PROVIDERS:
            raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}, got {value!r}")

        return chosen


class FileInspected(BaseModel):
    path: str
    sha256: str
    lines: int
    chars: int
    #: "model" when the model asked for the file, "agent" when a rule did.
    read_by: str


class TaskArtifact(BaseModel):
    path: str
    kind: str
    sha256: str
    bytes: int
    created: bool


class ProposedDiff(BaseModel):
    path: str
    proposal: str
    patch: str
    base_sha256: str
    added: int
    removed: int
    diff: str
    changes: int
    apply: str


class TaskToolCall(BaseModel):
    tool: str
    server: str
    arguments: dict
    planned_by: str
    ok: bool
    ms: int = 0
    summary: str = ""
    error: str = ""


class TaskModelCall(BaseModel):
    stage: str
    ok: bool
    ms: int = 0
    prompt_chars: int = 0
    reply_chars: int = 0
    error: str = ""


class TaskError(BaseModel):
    stage: str
    kind: str
    message: str


class ProjectTaskResponse(BaseModel):
    """``status``: completed, partial (something was produced, a limit or an
    error stopped part of the work - see ``errors``), failed (nothing was
    produced) or unsupported (not one of the two workflows)."""

    task: str
    status: str
    workflow: str
    subject: str
    summary: str
    provider: str
    model: str
    run_id: str
    plan: dict = Field(default_factory=dict)
    files_inspected: list[FileInspected] = Field(default_factory=list)
    artifacts: list[TaskArtifact] = Field(default_factory=list)
    diffs: list[ProposedDiff] = Field(default_factory=list)
    tool_calls: list[TaskToolCall] = Field(default_factory=list)
    model_calls: list[TaskModelCall] = Field(default_factory=list)
    findings: dict = Field(default_factory=dict)
    errors: list[TaskError] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    limits: dict = Field(default_factory=dict)
    timing: dict = Field(default_factory=dict)
    run_record: str = ""


class ApplyRequest(BaseModel):
    """Apply a saved proposal to the file it was made for. Without
    ``confirm`` nothing is written and the reply is the diff."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=500)
    proposal: str = Field(min_length=1, max_length=500)
    expected_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    confirm: bool = False


class ApplyResponse(BaseModel):
    path: str
    proposal: str
    applied: bool
    diff: str
    added: int
    removed: int
    base_sha256: str | None = None
    sha256: str | None = None
    message: str


class ProjectFilesError(BaseModel):
    error: str
    detail: str
    provider: str | None = None
    reason: str | None = None
