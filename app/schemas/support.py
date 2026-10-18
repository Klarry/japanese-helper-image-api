"""Day 33: the request and reply of ``POST /support/chat``."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.rag_chat import RagChatCitation, RagChatSource
from app.services.llm_provider import OLLAMA, PROVIDERS


class SupportChatRequest(BaseModel):
    """A question and the ticket it is about. There is no user_id: the user is
    the one the ticket belongs to, read from the CRM - a caller cannot name
    someone else's account."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000)
    ticket_id: str = Field(pattern=r"^[Tt][Ii][Cc][Kk][Ee][Tt]-\d{1,8}$")
    provider: str = OLLAMA

    @field_validator("message")
    @classmethod
    def message_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message is empty")

        return value

    @field_validator("ticket_id")
    @classmethod
    def ticket_id_upper(cls, value: str) -> str:
        return value.upper()

    @field_validator("provider")
    @classmethod
    def provider_must_be_known(cls, value: str) -> str:
        chosen = (value or "").strip().lower()

        if chosen not in PROVIDERS:
            raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}, got {value!r}")

        return chosen


class SupportToolUse(BaseModel):
    name: str
    server: str
    ok: bool
    ms: int = 0
    error: str = ""


class SupportTicketContext(BaseModel):
    """The verified case the answer was written with."""

    ticket_id: str
    status: str | None = None
    category: str | None = None
    subject: str | None = None
    description: str | None = None
    platform: str | None = None
    app_version: str | None = None
    previous_attempts: list[str] = Field(default_factory=list)
    user_language: str | None = None
    user_subscription: str | None = None


class SupportChatResponse(BaseModel):
    """``sources``/``citations`` are documentation, validated against the
    help-centre chunks; ``ticket`` is the CRM record the model was shown, and
    ``ticket_facts_used`` the fields it says it relied on. ``status`` is
    "answered" when at least one citation held, "insufficient_documentation"
    when none did."""

    request_id: str
    answer: str
    ticket_id: str
    status: str
    sources: list[RagChatSource] = Field(default_factory=list)
    citations: list[RagChatCitation] = Field(default_factory=list)
    rejected_citations: int = 0
    ticket: SupportTicketContext
    ticket_facts_used: list[str] = Field(default_factory=list)
    tools_used: list[SupportToolUse] = Field(default_factory=list)
    provider: str
    model: str
    retrieval: dict = Field(default_factory=dict)
    timing: dict = Field(default_factory=dict)


class SupportChatError(BaseModel):
    error: str
    detail: str
    ticket_id: str | None = None
    request_id: str | None = None
    provider: str | None = None
    reason: str | None = None
