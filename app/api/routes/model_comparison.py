from fastapi import APIRouter

from app.schemas.model_comparison import (
    ModelComparisonRequest,
    ModelComparisonResponse,
)
from app.services.model_comparison_service import compare_models

router = APIRouter()


@router.post("/model-comparison")
async def model_comparison(
    request: ModelComparisonRequest,
) -> ModelComparisonResponse:
    return await compare_models(request.kanji, request.prompt, request.models)
