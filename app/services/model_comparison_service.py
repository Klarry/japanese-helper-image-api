"""Model Comparison: runs one prompt for one kanji against several Gemini
text models and reports each model's answer, latency and token usage.

Every model gets the identical prompt - the model id is the only thing that
differs between the calls being compared.
"""

import asyncio
import logging
import time

from app.schemas.model_comparison import (
    ModelComparisonResponse,
    ModelComparisonResult,
)
from app.services.gemini_service import generate_text_with_usage

logger = logging.getLogger(__name__)

_KANJI_PLACEHOLDER = "{KANJI}"


def _build_prompt(kanji: str, prompt: str) -> str:
    if _KANJI_PLACEHOLDER in prompt:
        return prompt.replace(_KANJI_PLACEHOLDER, kanji)

    return f"{prompt}\n\nKanji: {kanji}"


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


async def _run_model(prompt: str, model: str) -> ModelComparisonResult:
    started = time.perf_counter()

    try:
        generated = await generate_text_with_usage(prompt, model=model)
    except Exception as error:  # noqa: BLE001 - one model must not sink the rest
        logger.warning("Model %s failed during comparison: %s", model, error)

        return ModelComparisonResult(
            model=model,
            response_time_ms=_elapsed_ms(started),
            error=getattr(error, "detail", None) or str(error),
        )

    return ModelComparisonResult(
        model=model,
        text=generated.text,
        response_time_ms=_elapsed_ms(started),
        input_tokens=generated.input_tokens,
        output_tokens=generated.output_tokens,
    )


async def compare_models(
    kanji: str, prompt: str, models: list[str]
) -> ModelComparisonResponse:
    final_prompt = _build_prompt(kanji, prompt)
    results = await asyncio.gather(
        *(_run_model(final_prompt, model) for model in models)
    )

    return ModelComparisonResponse(prompt=final_prompt, results=list(results))
