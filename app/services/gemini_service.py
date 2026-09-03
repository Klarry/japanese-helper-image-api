import logging
from typing import Any

import httpx
from fastapi import HTTPException

from app.core.config import (
    GEMINI_API_KEY,
    GEMINI_URL,
    HTTP_TIMEOUT,
    IMAGE_SEARCH_MODEL,
    TEXT_MODEL,
)

logger = logging.getLogger(__name__)


async def _post_to_gemini(payload: dict[str, Any]) -> dict[str, Any]:
    model = payload.get("model")
    logger.info("Calling Gemini model=%s", model)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(
            GEMINI_URL,
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if response.status_code != 200:
        logger.error("Gemini model=%s returned status %s", model, response.status_code)
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text,
        )

    return response.json()


def _find_image(obj: Any) -> str | None:
    if isinstance(obj, dict):
        if obj.get("type") == "image":
            data = obj.get("data")

            if data:
                return data

        for value in obj.values():
            result = _find_image(value)

            if result:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = _find_image(item)

            if result:
                return result

    return None


def _find_text(obj: Any) -> str | None:
    if isinstance(obj, dict):
        if obj.get("type") == "text":
            data = obj.get("data") or obj.get("text")

            if data:
                return data

        if "candidates" in obj:
            try:
                return obj["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                pass

        for value in obj.values():
            result = _find_text(value)

            if result:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = _find_text(item)

            if result:
                return result

    return None


async def search_image(query: str) -> str:
    """Run a Gemini image search and return the base64 payload of the image."""
    payload = {
        "model": IMAGE_SEARCH_MODEL,
        "input": f"Find a suitable image of {query}",
        "tools": [
            {
                "type": "google_search",
                "search_types": "image_search",
            }
        ],
    }

    data = await _post_to_gemini(payload)
    image_data = _find_image(data)

    if not image_data:
        logger.warning("No image found in Gemini response")
        raise HTTPException(
            status_code=404,
            detail="No image found in Gemini response",
        )

    return image_data


async def generate_text(prompt: str, temperature: float | None = None) -> str:
    """Run a text generation prompt through Gemini.

    ``temperature`` is omitted from the payload when not given, so existing
    callers that don't pass it keep getting the exact same request they
    always have.
    """
    payload: dict[str, Any] = {
        "model": TEXT_MODEL,
        "input": prompt,
    }

    if temperature is not None:
        payload["temperature"] = temperature

    data = await _post_to_gemini(payload)
    text = _find_text(data)

    if not text:
        logger.warning("No text found in Gemini response")
        raise HTTPException(
            status_code=502,
            detail="No text found in Gemini response",
        )

    return text.strip()
