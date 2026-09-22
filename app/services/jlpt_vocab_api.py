"""The JLPT vocabulary API - the Japanese dictionary the app already uses.

The Android app reads random words from https://jlpt-vocab-api.vercel.app
(``GET api/words/random``, see VocabApi.kt). This module talks to the same
API from the backend: ``GET api/words?word=...`` looks a word up by how it is
written and returns every entry spelled exactly that way, with its reading,
romaji, English meaning and JLPT level:

    {"total": 1, "offset": 0, "limit": 10,
     "words": [{"word": "学習", "meaning": "study, learning",
                "furigana": "がくしゅう", "romaji": "gakushū", "level": 3}]}

``level`` is the number of the JLPT level (3 means N3). The fields are the
ones the app's RandomWordDto already maps.

Deliberately free of app.core.config: the MCP server that uses this runs as
a subprocess without GEMINI_API_KEY in its environment, and config.py reads
that key at import time. The base URL can still be overridden, through
JLPT_VOCAB_API_URL.
"""

import logging
import os

import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://jlpt-vocab-api.vercel.app/"
_SEARCH_PATH = "api/words"
_TIMEOUT_SECONDS = 10.0


class VocabWord(BaseModel):
    """One entry, with the API's own field names."""

    word: str
    meaning: str | None = None
    furigana: str | None = None
    romaji: str | None = None
    level: int | None = None


class _SearchPage(BaseModel):
    total: int = 0
    words: list[VocabWord] = []


class JlptVocabApiError(RuntimeError):
    """The API could not be reached, or answered with something that is not a
    word list. Never raised for "no such word" - that is an empty result."""


def api_url() -> str:
    """Read at call time, so a test (or a subprocess) can point it elsewhere."""
    return os.getenv("JLPT_VOCAB_API_URL", DEFAULT_API_URL)


async def search_words(word: str, transport: httpx.AsyncBaseTransport | None = None) -> list[VocabWord]:
    """Every JLPT entry written exactly as ``word``. Empty when there is none.

    Raises JlptVocabApiError when the API is down, answers with an error
    status, or returns something that does not parse - the three ways a
    caller could otherwise mistake a broken dictionary for a missing word.
    """
    url = api_url().rstrip("/") + "/" + _SEARCH_PATH

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS, transport=transport) as client:
            response = await client.get(url, params={"word": word})
    except httpx.HTTPError as error:
        logger.error("JLPT vocabulary API GET %s failed: %r", url, error)
        raise JlptVocabApiError(f"the JLPT vocabulary API is unreachable ({error.__class__.__name__})") from error

    if response.status_code != 200:
        logger.error("JLPT vocabulary API GET %s returned %s: %s", url, response.status_code, response.text[:300])
        raise JlptVocabApiError(f"the JLPT vocabulary API answered with status {response.status_code}")

    try:
        return _SearchPage.model_validate(response.json()).words
    except (ValueError, ValidationError) as error:
        logger.error("JLPT vocabulary API GET %s returned an unreadable body: %s", url, response.text[:300])
        raise JlptVocabApiError("the JLPT vocabulary API returned something that is not a word list") from error
