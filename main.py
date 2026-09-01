import os
import base64
import io
import httpx

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image

load_dotenv()

app = FastAPI()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

MAX_IMAGE_WIDTH = 800
JPEG_QUALITY = 75


class ImageSearchRequest(BaseModel):
    query: str


def find_image(obj):
    if isinstance(obj, dict):
        if obj.get("type") == "image":
            data = obj.get("data")

            if data:
                return data, obj.get("mime_type", "image/jpeg")

        for value in obj.values():
            result = find_image(value)

            if result:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = find_image(item)

            if result:
                return result

    return None


def optimize_image(image_bytes: bytes) -> bytes:
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


@app.post("/image-search")
async def image_search(request: ImageSearchRequest):
    payload = {
        "model": "gemini-3.1-flash-image",
        "input": f"Find a suitable image of {request.query}",
        "tools": [
            {
                "type": "google_search",
                "search_types": "image_search",
            }
        ],
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            GEMINI_URL,
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text,
        )

    data = response.json()

    image = find_image(data)

    if not image:
        raise HTTPException(
            status_code=404,
            detail="No image found in Gemini response",
        )

    image_data, _ = image

    try:
        image_bytes = base64.b64decode(image_data)
        image_bytes = optimize_image(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {e}",
        )

    return Response(
        content=image_bytes,
        media_type="image/jpeg",
    )
