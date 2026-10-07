"""Day 28: asking the index a question, and saying who answered it.

The response is the Day 24 schema with two fields added, not a new shape:
whatever else changes, an answer still arrives with its sources, its
citations, its confidence and its status, and a caller written for Day 24
can read this one without knowing Day 28 happened.
"""

from pydantic import BaseModel, Field, field_validator

from app.services.llm_provider import GEMINI, PROVIDERS


class RagChatRequest(BaseModel):
    message: str
    #: Which model writes the answer. Retrieval is local either way - the
    #: index, the filter and the reranker are the same ones in both modes.
    provider: str = GEMINI

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("message must not be empty")

        return stripped

    @field_validator("provider")
    @classmethod
    def provider_must_be_known(cls, value: str) -> str:
        chosen = (value or GEMINI).strip().lower()

        if chosen not in PROVIDERS:
            # Refused rather than defaulted: silently answering a typo with
            # the cloud is the one failure this mode exists to prevent.
            raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}, got {value!r}")

        return chosen


class RagChatTiming(BaseModel):
    """Where the time went, so the two providers can be compared on more
    than prose. Retrieval is the local half and should look the same in
    both; generation is the half that differs."""

    retrieval_ms: int
    generation_ms: int
    total_ms: int


class RagChatSource(BaseModel):
    source: str
    file: str
    section: str
    chunk_id: str


class RagChatCitation(BaseModel):
    """An exact fragment of one retrieved chunk, as Day 24 validated it."""

    quote: str
    source: str
    section: str
    chunk_id: str


class RagChatResponse(BaseModel):
    answer: str
    sources: list[RagChatSource] = Field(default_factory=list)
    citations: list[RagChatCitation] = Field(default_factory=list)
    confidence: str
    rag_status: str
    provider: str
    model: str
    retrieved_count: int = 0
    final_count: int = 0
    best_relevance: float = 0.0
    answer_threshold: float = 0.0
    timing: RagChatTiming


class RagChatUnavailable(BaseModel):
    """The local model could not answer.

    No field for an answer, because there is no fallback: the mode's whole
    claim is that the answer came from the model it names.
    """

    error: str
    provider: str
    reason: str
    detail: str
