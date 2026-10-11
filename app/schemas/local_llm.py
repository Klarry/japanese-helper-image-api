"""Day 26: the request and reply shapes of the local-LLM demo endpoint.

Deliberately small. The endpoint exists to show that a model running on
this machine answers over HTTP - it is not a second agent, so it takes a
prompt and returns an answer, and nothing about memory, retrieval or
tools appears here.
"""

import re

from pydantic import BaseModel, field_validator

#: Day 30. A conversation id becomes a file name, so it is held to the
#: characters a file name can safely be built of - and nothing else.
_CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class LocalLlmRequest(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("prompt must not be empty")

        return stripped


class LocalLlmResponse(BaseModel):
    """What the assignment asks for, and nothing else: which model
    answered, and what it said."""

    model: str
    response: str


class LocalLlmHealthResponse(BaseModel):
    """Whether the local model can be called at all.

    ``available`` and ``error`` are the two the assignment names. The two
    optional fields below exist because "Ollama is up" and "the model you
    are about to ask for is installed" are different facts, and finding out
    the second one by failing a generation is a worse way to learn it.
    """

    available: bool
    model: str
    model_installed: bool | None = None
    models: list[str] | None = None
    error: str | None = None
    #: Day 30: both of the above at once - Ollama answers *and* has the
    #: model. Not "FastAPI is running": that is true even when this is false.
    ready: bool | None = None
    #: Day 30: the limits the chat endpoint is enforcing right now.
    service: dict | None = None


class LocalChatRequest(BaseModel):
    """What the app sends (Day 27). One field.

    Named ``message`` rather than ``prompt`` because this is the chat the
    app talks to, not the Day 26 demo - and a device that could also set
    the model or the provider could also contradict them.

    Day 30 adds ``conversation_id``, optional. Without it every message
    stands alone, exactly as on Day 27; with it, earlier turns of that
    conversation go into the prompt, as many as the context budget allows.
    Its length is not limited here on purpose: an oversized message is
    answered with 413 by the endpoint, not with a validation error.
    """

    message: str
    conversation_id: str | None = None

    @field_validator("conversation_id")
    @classmethod
    def conversation_id_must_be_a_safe_name(cls, value: str | None) -> str | None:
        if value is None:
            return None

        if not _CONVERSATION_ID.match(value):
            raise ValueError("conversation_id may only use letters, digits, '-' and '_' (1-64 characters)")

        return value

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("message must not be empty")

        return stripped


class LocalChatResponse(BaseModel):
    """The answer, and who produced it.

    ``provider`` is here so the screen can say *which* model answered
    without inferring it from the model name, and so a reply that somehow
    came from the cloud could not be mistaken for a local one.
    """

    response: str
    model: str
    provider: str
    seconds: float
    #: Day 30. Everything below is additive: a client written for Day 27
    #: reads the four fields above and is unaffected by the rest.
    request_id: str = ""
    conversation_id: str | None = None
    #: Measured, in milliseconds: the model call, and the whole request.
    timing: dict | None = None
    #: What went into the prompt, and what the context budget allowed.
    context: dict | None = None
    #: What Ollama reported about the run - real counts, not estimates.
    usage: dict | None = None


class LocalChatUnavailable(BaseModel):
    """What comes back when the local model cannot answer.

    ``error`` is the one line the app shows. ``reason`` and ``detail`` are
    the Day 26 rule kept: the real cause is not swallowed, and the detail
    still says what to do about it. There is deliberately no field for a
    fallback answer, because there is deliberately no fallback.
    """

    error: str
    reason: str
    detail: str
    #: Day 30: the same failure shape now also covers the service's own
    #: refusals (rate limit, size, capacity), so every one carries the id
    #: that finds it in the log - and the provider, so no one wonders
    #: whether some other model was tried instead.
    provider: str = "ollama"
    request_id: str = ""
    retry_after: int | None = None
