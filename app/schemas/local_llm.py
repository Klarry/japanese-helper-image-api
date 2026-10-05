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
