from enum import Enum

from pydantic import BaseModel, field_validator


class ContextStrategy(str, Enum):
    """How much of the conversation a request puts in front of the model.

    Part of the API contract rather than an implementation detail: the client
    picks one. FULL and SUMMARY are what the agent already did (whole history,
    and the running summary plus the messages kept verbatim); the other three
    are the strategies that work without any summary at all.
    """

    FULL = "full"
    SUMMARY = "summary"
    SLIDING_WINDOW = "sliding_window"
    STICKY_FACTS = "sticky_facts"
    BRANCHING = "branching"


class AgentChatRequest(BaseModel):
    message: str
    # Which strategy to answer with, for this request only. Omitted means the
    # strategy chosen through PUT /agent/strategy, and failing that the older
    # compression flag below - so clients that know nothing about strategies
    # keep working exactly as before.
    strategy: ContextStrategy | None = None
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
    """What this request actually sent, so a client can show it next to the
    token usage the same request produced. ``summary_tokens`` is the real
    size of the summary that went with it (zero when none did, which is
    always the case for the summary-free strategies), and ``messages_sent``
    is how many messages went along word for word.

    It describes the request, not the conversation as it stands afterwards:
    on the turn where older messages are folded away, the tokens above were
    still spent on sending them, and a status claiming otherwise would
    contradict its own numbers."""

    enabled: bool = False
    summary_tokens: int | None = None
    messages_sent: int = 0


class AgentChatResponse(BaseModel):
    response: str
    usage: AgentTokenUsage
    compression: AgentCompressionStatus
    strategy: str = ContextStrategy.FULL.value


class AgentHistoryMessage(BaseModel):
    role: str
    content: str


class AgentHistoryResponse(BaseModel):
    """The stored conversation on the current branch. ``summary`` covers the
    older messages that compression has already folded away; it is always
    empty under the summary-free strategies, and clients that only read
    ``messages`` are unaffected either way."""

    messages: list[AgentHistoryMessage]
    summary: str = ""


class AgentStrategyRequest(BaseModel):
    strategy: ContextStrategy


class AgentStrategyResponse(BaseModel):
    strategy: str


class AgentContextResponse(BaseModel):
    """Exactly what the next request would put in front of the model, plus
    the state that decides it: the strategy, the branch being talked on, and
    the branches and checkpoints available to switch to."""

    strategy: str
    branch: str
    branches: list[str]
    checkpoints: list[str]
    facts: dict[str, str] = {}
    messages: list[AgentHistoryMessage] = []
    context: str = ""


class AgentCheckpointRequest(BaseModel):
    name: str | None = None


class AgentCheckpointResponse(BaseModel):
    name: str
    branch: str
    messages: int


class AgentBranchRequest(BaseModel):
    name: str
    checkpoint: str


class AgentBranchSwitchRequest(BaseModel):
    name: str


class AgentBranchResponse(BaseModel):
    branch: str
    branches: list[str]


class AgentUsageEntry(BaseModel):
    """One recorded /agent/chat call. Every field has a default so a record
    written by an older version of the app still reads back cleanly."""

    timestamp: str = ""
    strategy: str = ""
    compression_enabled: bool = False
    messages_sent: int = 0
    summary_used: bool = False
    current_request_tokens: int | None = None
    history_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    summarization_tokens: int = 0
    facts_tokens: int = 0


class AgentUsageResponse(BaseModel):
    entries: list[AgentUsageEntry]
