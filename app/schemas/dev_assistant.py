"""Day 31: the request and reply of the developer assistant (/help)."""

from pydantic import BaseModel, Field, field_validator

from app.schemas.rag_chat import RagChatCitation, RagChatSource
from app.services.llm_provider import GEMINI, PROVIDERS


class DevChatRequest(BaseModel):
    """``message`` is the command itself - "/help" or "/help <question>"."""

    message: str
    provider: str = GEMINI

    @field_validator("provider")
    @classmethod
    def provider_must_be_known(cls, value: str) -> str:
        chosen = (value or "").strip().lower()

        if chosen not in PROVIDERS:
            raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}, got {value!r}")

        return chosen


class DevChatToolUse(BaseModel):
    name: str
    server: str
    ok: bool
    result: dict | None = None
    error: str = ""
    ms: int = 0


class DevChatResponse(BaseModel):
    """The /rag/chat shape (Day 28) where it overlaps - sources, citations,
    rag_status, provider, model - plus the tools that were called.

    ``rag_status`` is "disabled" when no documentation was needed (plain
    /help, or a question about the branch alone); ``model`` is null when no
    model was asked anything."""

    command: str
    question: str
    answer: str
    sources: list[RagChatSource] = Field(default_factory=list)
    citations: list[RagChatCitation] = Field(default_factory=list)
    rag_status: str
    confidence: str = ""
    provider: str
    model: str | None = None
    tools_used: list[DevChatToolUse] = Field(default_factory=list)
    timing: dict = Field(default_factory=dict)
    best_relevance: float = 0.0
    #: How the documentation was searched (search query, best similarity
    #: against the filter, whether the question itself was searched again).
    #: Null when no documentation was needed.
    retrieval: dict | None = None


class DevChatError(BaseModel):
    error: str
    detail: str
    provider: str | None = None
    reason: str | None = None
