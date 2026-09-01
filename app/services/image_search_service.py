import base64
import io
import logging

from fastapi import HTTPException
from PIL import Image

from app.core.config import JPEG_QUALITY, MAX_IMAGE_WIDTH
from app.services.gemini_service import search_image

logger = logging.getLogger(__name__)


def _optimize_image(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")

        if image.width > MAX_IMAGE_WIDTH:
            height = round(image.height * MAX_IMAGE_WIDTH / image.width)
            image = image.resize(
                (MAX_IMAGE_WIDTH, height),
                Image.Resampling.LANCZOS,
            )

        output = io.BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
        )

        return output.getvalue()


def decode_and_optimize(base64_data: str) -> bytes:
    try:
        image_bytes = base64.b64decode(base64_data)
        return _optimize_image(image_bytes)
    except Exception as e:
        logger.exception("Failed to process image returned by Gemini")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {e}",
        )


async def find_image_bytes(query: str) -> bytes:
    """Search for an image and return optimized JPEG bytes."""
    base64_data = await search_image(query)
    image_bytes = decode_and_optimize(base64_data)
    logger.info("Returning image of %d bytes", len(image_bytes))
    return image_bytes
