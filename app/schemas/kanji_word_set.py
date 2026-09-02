from enum import Enum

from pydantic import BaseModel, field_validator


class ExperimentType(str, Enum):
    DIRECT = "DIRECT"
    STEP_BY_STEP = "STEP_BY_STEP"
    PROMPT = "PROMPT"
    EXPERTS = "EXPERTS"


class KanjiWordSetRequest(BaseModel):
    kanji: str
    experimentType: ExperimentType

    @field_validator("kanji")
    @classmethod
    def kanji_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("kanji must not be empty")

        return stripped


class KanjiWordSetResponse(BaseModel):
    prompt: str
    words: list[str]
    cost: int
    value: int
