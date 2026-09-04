from pydantic import BaseModel, field_validator

# Fan-out bound: every listed model costs one Gemini call per request.
MAX_COMPARED_MODELS = 5


class ModelComparisonRequest(BaseModel):
    kanji: str
    prompt: str
    models: list[str]

    @field_validator("kanji", "prompt")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("value must not be empty")

        return stripped

    @field_validator("models")
    @classmethod
    def models_must_be_usable(cls, value: list[str]) -> list[str]:
        models = [model.strip() for model in value]

        if not models or any(not model for model in models):
            raise ValueError("models must contain at least one non-empty model id")

        if len(models) > MAX_COMPARED_MODELS:
            raise ValueError(f"models must contain at most {MAX_COMPARED_MODELS} entries")

        return models


class ModelComparisonResult(BaseModel):
    model: str
    text: str | None = None
    response_time_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class ModelComparisonResponse(BaseModel):
    prompt: str
    results: list[ModelComparisonResult]
