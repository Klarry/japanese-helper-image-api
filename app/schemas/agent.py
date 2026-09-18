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
    LAYERED_MEMORY = "layered_memory"


class InvariantCategory(str, Enum):
    """The four kinds of rule the invariants layer is made of. A category is
    part of the contract rather than a label: it says what kind of thing is
    being constrained, which is what makes a conflict explainable instead of
    merely refused."""

    ARCHITECTURE = "architecture"
    TECHNOLOGY_STACK = "technology_stack"
    TECHNICAL_DECISIONS = "technical_decisions"
    BUSINESS_RULES = "business_rules"


class MemoryLayer(str, Enum):
    """The three layers the agent's memory is made of, kept apart because
    they have different lifetimes: short-term lasts a conversation, working
    memory lasts a task, long-term outlives both."""

    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"


class TaskStage(str, Enum):
    """The stages a task moves through, in the only order they may be taken.

    ``IDLE`` is the absence of a task - not a stage the learner is ever in the
    middle of, which is why it is the only stage a task can be started from
    and the one clearing returns to.
    """

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"


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


class AgentShortTermMemory(BaseModel):
    """The current conversation. Not a copy of the transcript - it is the
    stored messages of this session, reported as a memory layer."""

    messages: list[AgentHistoryMessage] = []


class AgentWorkingMemory(BaseModel):
    """The task being worked on right now."""

    goals: list[str] = []
    requirements: list[str] = []
    constraints: list[str] = []
    decisions: list[str] = []


class AgentLongTermMemory(BaseModel):
    """What stays true about the learner between conversations."""

    profile: dict[str, str] = {}
    preferences: list[str] = []
    decisions: list[str] = []
    knowledge: list[str] = []


class AgentMemoryResponse(BaseModel):
    """All three layers, each under its own key. Deliberately not merged
    into one blob: which layer something is in is the whole point."""

    short_term: AgentShortTermMemory = AgentShortTermMemory()
    working: AgentWorkingMemory = AgentWorkingMemory()
    long_term: AgentLongTermMemory = AgentLongTermMemory()


class AgentShortTermMemoryRequest(BaseModel):
    """Replace the current conversation with these messages."""

    messages: list[AgentHistoryMessage] = []


class AgentWorkingMemoryRequest(BaseModel):
    """Update working memory. A field left out stays as it is; send an empty
    list to empty that field, or DELETE the layer to clear all of it."""

    goals: list[str] | None = None
    requirements: list[str] | None = None
    constraints: list[str] | None = None
    decisions: list[str] | None = None


class AgentLongTermMemoryRequest(BaseModel):
    """Update long-term memory, with the same leave-out-to-keep rule."""

    profile: dict[str, str] | None = None
    preferences: list[str] | None = None
    decisions: list[str] | None = None
    knowledge: list[str] | None = None


class AgentUserProfile(BaseModel):
    """How the learner wants to be answered. Settings, not conversation - so
    this is not one of the memory layers and is never mixed into them."""

    preferred_language: str = ""
    japanese_level: str = ""
    explanation_style: str = ""
    answer_format: str = ""
    translation_language: str = ""
    preferences: list[str] = []


class AgentUserProfileRequest(BaseModel):
    """Update the profile. A field left out stays as it is; send an empty
    string (or an empty list) to unset one, or DELETE the profile to unset
    everything."""

    preferred_language: str | None = None
    japanese_level: str | None = None
    explanation_style: str | None = None
    answer_format: str | None = None
    translation_language: str | None = None
    preferences: list[str] | None = None


class AgentInvariant(BaseModel):
    """One rule the agent may never break. ``id`` is what makes a single
    rule changeable or removable without touching the rest."""

    id: str
    category: str
    rule: str


class AgentInvariantsResponse(BaseModel):
    invariants: list[AgentInvariant] = []


class AgentInvariantRequest(BaseModel):
    """A rule to add or change. The category is one of the four the layer
    is made of; an unknown one is rejected rather than stored."""

    category: InvariantCategory
    rule: str

    @field_validator("rule")
    @classmethod
    def rule_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("rule must not be empty")

        return stripped


class AgentTaskTransitionError(BaseModel):
    """Why a move between stages was refused.

    Everything the refusal has to say, in named fields rather than one
    sentence: which stage the task is in, which stage was asked for, which
    stage may actually come next, and the condition that is not met yet.
    ``message`` is those four put into a sentence, for a client that only
    wants to show something.
    """

    message: str = ""
    current_stage: str = ""
    requested_stage: str | None = None
    required_next: list[str] = []
    unmet_condition: str = ""


class AgentTaskStateResponse(BaseModel):
    """Where the task in progress has got to, and where it may go next.

    ``allowed_next`` is the state machine itself, reported rather than
    documented: a client never has to guess which move is legal, and an
    empty list means the task is finished (or there is none).
    ``next_requirement`` is the other half of that answer - a move can be
    legal in shape and still blocked because the stage's own work is not
    finished, and this says what is missing.
    """

    task_stage: str = "idle"
    current_step: str = ""
    expected_action: str = ""
    allowed_next: list[str] = []
    plan: str = ""
    validation_passed: bool = False
    validation_note: str = ""
    next_requirement: str = ""
    # The last refused move, kept so the refusal survives the response that
    # reported it: the screen can still show why the task did not advance,
    # and the agent is told about it on the next message.
    blocked: AgentTaskTransitionError | None = None


class AgentTaskTransitionRequest(BaseModel):
    """Ask for a specific move. Rejected as a whole when the move is not
    allowed - the step fields describe the stage being asked for, so storing
    them without the stage would leave the task describing itself wrongly."""

    task_stage: TaskStage
    current_step: str = ""
    expected_action: str = ""


class AgentTaskPlanRequest(BaseModel):
    """Approve the plan the task will be executed by. Only meaningful while
    the task is planning: the plan is what planning produces."""

    plan: str

    @field_validator("plan")
    @classmethod
    def plan_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("plan must not be empty")

        return stripped


class AgentTaskValidationRequest(BaseModel):
    """Record how validation went. A failed validation is recorded too - it
    is the difference between "not checked yet" and "checked, not right yet",
    and neither of them opens the way to done."""

    passed: bool
    notes: str = ""


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
    memory_tokens: int = 0
    task_tokens: int = 0


class AgentUsageResponse(BaseModel):
    entries: list[AgentUsageEntry]
