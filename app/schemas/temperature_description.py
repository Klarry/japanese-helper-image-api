from pydantic import BaseModel, field_validator

# The three temperature values this experiment compares. Sampling is
# non-deterministic, so we don't try to match floats with a tolerance -
# these exact literals round-trip identically through JSON and Python.
SUPPORTED_TEMPERATURES = (0.0, 0.7, 1.2)


class TemperatureDescriptionRequest(BaseModel):
    kanji: str
    temperature: float

    @field_validator("kanji")
    @classmethod
    def kanji_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("kanji must not be empty")

        return stripped

    @field_validator("temperature")
    @classmethod
    def temperature_must_be_supported(cls, value: float) -> float:
        if value not in SUPPORTED_TEMPERATURES:
            raise ValueError(
                f"temperature must be one of {SUPPORTED_TEMPERATURES}, got {value}"
            )

        return value


class TemperatureDescriptionResponse(BaseModel):
    sentence: str
    translation: str
    temperature: float
