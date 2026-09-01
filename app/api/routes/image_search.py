from fastapi import APIRouter
from fastapi.responses import Response

from app.schemas.image_search import ImageSearchRequest
from app.services.image_search_service import find_image_bytes

router = APIRouter()


@router.post("/image-search")
async def image_search(request: ImageSearchRequest) -> Response:
    image_bytes = await find_image_bytes(request.query)

    return Response(
        content=image_bytes,
        media_type="image/jpeg",
    )
