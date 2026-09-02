from fastapi import APIRouter

from app.schemas.kanji_word_set import KanjiWordSetRequest, KanjiWordSetResponse
from app.services.kanji_word_set_service import build_kanji_word_set

router = APIRouter()


@router.post("/kanji-word-set")
async def kanji_word_set(request: KanjiWordSetRequest) -> KanjiWordSetResponse:
    return await build_kanji_word_set(request.experimentType, request.kanji)
