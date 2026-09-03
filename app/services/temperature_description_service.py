"""Temperature Description: runs one fixed prompt through Gemini at a
caller-selected temperature and parses the resulting Japanese sentence and
its Russian translation.

The prompt is identical on every call - only ``temperature`` varies between
the 0.0 / 0.7 / 1.2 requests the client compares.
"""

import logging
import re

from fastapi import HTTPException

from app.schemas.temperature_description import TemperatureDescriptionResponse
from app.services.gemini_service import generate_text

logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = """Compose one natural Japanese sentence using the kanji {KANJI}.
Use common vocabulary and a language level around N4-N3.
Add a Russian translation.
Respond in exactly this format, with no extra commentary:
sentence: <the Japanese sentence>
translation: <the Russian translation>"""

# Matches "sentence: ... translation: ..." (case-insensitive, tolerant of
# newlines/extra whitespace and of leading reasoning the model was told not
# to include but might anyway).
_RESULT_PATTERN = re.compile(
    r"sentence\s*:\s*(?P<sentence>.+?)\s*translation\s*:\s*(?P<translation>.+)",
    re.IGNORECASE | re.DOTALL,
)


def _build_prompt(kanji: str) -> str:
    return _PROMPT_TEMPLATE.replace("{KANJI}", kanji)


def _parse_result(text: str) -> tuple[str, str]:
    match = _RESULT_PATTERN.search(text)

    if not match:
        logger.warning("Could not parse temperature description result: %r", text)
        raise HTTPException(
            status_code=502,
            detail="Failed to parse AI response into sentence/translation",
        )

    sentence = match.group("sentence").strip()
    translation = match.group("translation").strip()

    if not sentence or not translation:
        logger.warning("Parsed empty sentence/translation from result: %r", text)
        raise HTTPException(
            status_code=502,
            detail="AI response did not contain both a sentence and a translation",
        )

    return sentence, translation


async def build_temperature_description(
    kanji: str, temperature: float
) -> TemperatureDescriptionResponse:
    prompt = _build_prompt(kanji)
    result_text = await generate_text(prompt, temperature=temperature)
    sentence, translation = _parse_result(result_text)

    return TemperatureDescriptionResponse(
        sentence=sentence,
        translation=translation,
        temperature=temperature,
    )
