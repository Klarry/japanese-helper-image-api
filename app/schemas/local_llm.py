"""Day 26: the request and reply shapes of the local-LLM demo endpoint.

Deliberately small. The endpoint exists to show that a model running on
this machine answers over HTTP - it is not a second agent, so it takes a
prompt and returns an answer, and nothing about memory, retrieval or
tools appears here.
"""

from pydantic import BaseModel, field_validator


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


class LocalChatRequest(BaseModel):
    """What the app sends (Day 27). One field.

    Named ``message`` rather than ``prompt`` because this is the chat the
    app talks to, not the Day 26 demo - and a device that could also set
    the model or the provider could also contradict them.
    """

    message: str

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
