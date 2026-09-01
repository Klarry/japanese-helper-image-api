from pydantic import BaseModel


class ImageSearchRequest(BaseModel):
    query: str
