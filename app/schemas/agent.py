from pydantic import BaseModel, field_validator


class AgentChatRequest(BaseModel):
    message: str
    # Which mode to answer in. Omitted means "whatever the server is
    # configured for" (AGENT_COMPRESSION_ENABLED), so existing clients that
    # know nothing about compression keep working unchanged.
    compression_enabled: bool | None = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("message must not be empty")

        return stripped


class AgentTokenUsage(BaseModel):
    """Real token counts from Gemini for one /agent/chat call. A field is
    ``None`` only when Gemini didn't report the underlying number for that
    call - never an estimate standing in for a missing real value.
    """

    current_request_tokens: int | None = None
    history_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None


class AgentCompressionStatus(BaseModel):
    """What the agent is now keeping for this conversation, so a client can
    show the mode next to the token usage it produced. ``summary_tokens`` is
    the real size of the stored summary (zero when there is none, null when
    Gemini did not report it), and ``recent_messages`` is how many messages
    are still kept word for word."""

    enabled: bool = False
    summary_tokens: int | None = None
    recent_messages: int = 0


class AgentChatResponse(BaseModel):
    response: str
    usage: AgentTokenUsage
    compression: AgentCompressionStatus


class AgentHistoryMessage(BaseModel):
    role: str
    content: str


class AgentHistoryResponse(BaseModel):
    """The stored conversation. ``summary`` covers the older messages that
    compression has already folded away; it is always empty while
    compression is off, and clients that only read ``messages`` are
    unaffected either way."""

    messages: list[AgentHistoryMessage]
    summary: str = ""


class AgentUsageEntry(BaseModel):
    """One recorded /agent/chat call. Every field has a default so a record
    written by an older version of the app still reads back cleanly."""

    timestamp: str = ""
    compression_enabled: bool = False
    messages_sent: int = 0
    summary_used: bool = False
    current_request_tokens: int | None = None
    history_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    summarization_tokens: int = 0


class AgentUsageResponse(BaseModel):
    entries: list[AgentUsageEntry]
