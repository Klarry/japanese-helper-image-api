from fastapi import APIRouter

from app.schemas.description import DescriptionRequest, DescriptionResponse
from app.services.description_service import describe

router = APIRouter()


@router.post("/description")
async def description(request: DescriptionRequest) -> DescriptionResponse:
    return await describe(request.meaning)
