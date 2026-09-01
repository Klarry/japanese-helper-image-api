from pydantic import BaseModel


class DescriptionRequest(BaseModel):
    meaning: str


class DescriptionResponse(BaseModel):
    uncontrolled: str
    controlled: str
