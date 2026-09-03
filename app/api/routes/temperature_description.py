from fastapi import APIRouter

from app.schemas.temperature_description import (
    TemperatureDescriptionRequest,
    TemperatureDescriptionResponse,
)
from app.services.temperature_description_service import build_temperature_description

router = APIRouter()


@router.post("/temperature-description")
async def temperature_description(
    request: TemperatureDescriptionRequest,
) -> TemperatureDescriptionResponse:
    return await build_temperature_description(request.kanji, request.temperature)
