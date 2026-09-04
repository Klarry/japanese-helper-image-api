import logging
from typing import Any, NamedTuple

import httpx
from fastapi import HTTPException

from app.core.config import (
    GEMINI_API_KEY,
    GEMINI_GENERATE_CONTENT_URL_TEMPLATE,
    GEMINI_URL,
    HTTP_TIMEOUT,
    IMAGE_SEARCH_MODEL,
    TEXT_MODEL,
)

logger = logging.getLogger(__name__)


async def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(
            url,
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if response.status_code != 200:
        # Log the response body too, not just the status - it's the only place
        # this ever surfaces server-side, and it's what actually explains a
        # pass-through error like a 400 (e.g. an unsupported payload field).
        logger.error(
            "Gemini POST %s returned status %s: %s",
            url,
            response.status_code,
            response.text,
        )
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text,
        )

    return response.json()


async def _post_to_gemini(payload: dict[str, Any]) -> dict[str, Any]:
    logger.info("Calling Gemini model=%s", payload.get("model"))
    return await _post(GEMINI_URL, payload)


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


# Where each API surface reports token usage. Only these documented fields are
# read - anything else stays None rather than being guessed at.
_USAGE_FIELDS = (
    # Interactions API
    ("usage", "total_input_tokens", "total_output_tokens"),
    # generateContent API
    ("usageMetadata", "promptTokenCount", "candidatesTokenCount"),
)


def _as_token_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    return int(value)


def _find_token_usage(obj: Any) -> tuple[int | None, int | None]:
    if isinstance(obj, dict):
        for container, input_field, output_field in _USAGE_FIELDS:
            block = obj.get(container)

            if isinstance(block, dict):
                return (
                    _as_token_count(block.get(input_field)),
                    _as_token_count(block.get(output_field)),
                )

        for value in obj.values():
            found = _find_token_usage(value)

            if found != (None, None):
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = _find_token_usage(item)

            if found != (None, None):
                return found

    return None, None


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


class GeneratedText(NamedTuple):
    """A generation result. Token counts are ``None`` when Gemini didn't report them."""

    text: str
    input_tokens: int | None
    output_tokens: int | None


async def _generate(prompt: str, temperature: float | None, model: str) -> GeneratedText:
    """Run a text generation prompt through Gemini.

    When ``temperature`` is omitted, this is the exact same Interactions API
    request every existing caller always got. When it's given, the request
    goes to the classic generateContent endpoint instead: the Interactions
    API has no temperature parameter at all (confirmed by Google's docs and
    by the API itself - "Unknown parameter 'temperature'"), while
    generateContent supports it via ``generationConfig.temperature``.
    """
    if temperature is None:
        data = await _post_to_gemini({"model": model, "input": prompt})
    else:
        url = GEMINI_GENERATE_CONTENT_URL_TEMPLATE.format(model=model)
        logger.info("Calling Gemini model=%s temperature=%s", model, temperature)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        data = await _post(url, payload)

    text = _find_text(data)

    if not text:
        logger.warning("No text found in Gemini response")
        raise HTTPException(
            status_code=502,
            detail="No text found in Gemini response",
        )

    input_tokens, output_tokens = _find_token_usage(data)

    return GeneratedText(text.strip(), input_tokens, output_tokens)


async def generate_text(
    prompt: str,
    temperature: float | None = None,
    model: str = TEXT_MODEL,
) -> str:
    """Generate text with Gemini. Defaults to [TEXT_MODEL], so existing
    callers keep sending exactly the request they always have.
    """
    return (await _generate(prompt, temperature, model)).text


async def generate_text_with_usage(
    prompt: str,
    model: str = TEXT_MODEL,
    temperature: float | None = None,
) -> GeneratedText:
    """Same call as [generate_text], but also reports Gemini's token usage."""
    return await _generate(prompt, temperature, model)
