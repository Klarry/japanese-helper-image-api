from pydantic import BaseModel, field_validator


class AgentChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("message must not be empty")

        return stripped


class AgentChatResponse(BaseModel):
    response: str


class AgentHistoryMessage(BaseModel):
    role: str
    content: str


class AgentHistoryResponse(BaseModel):
    messages: list[AgentHistoryMessage]
