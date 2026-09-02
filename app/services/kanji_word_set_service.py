"""Kanji Word Set: builds a prompt for one of four prompting experiments,
sends it to Gemini, and parses the resulting words/cost/value into a
structured response.

All four prompt templates are defined server-side (never sent by the
client) and share the same fixed 0/1 Knapsack formulation: cost =
kanji_count, value = frequency * 2 + 3, budget = 10. Only the kanji is
substituted dynamically via the {KANJI} placeholder.
"""

import logging
import re

from fastapi import HTTPException

from app.schemas.kanji_word_set import ExperimentType, KanjiWordSetResponse
from app.services.gemini_service import generate_text

logger = logging.getLogger(__name__)


_DIRECT_TEMPLATE = """Ты получаешь один японский кандзи.

Кандзи: {KANJI}

Сформируй набор распространённых японских слов, содержащих этот кандзи.

Для каждого слова определи:
- слово;
- чтение;
- перевод;
- frequency - частотность от 1 до 5, где 5 означает наиболее распространённое слово;
- kanji_count - количество кандзи в слове.

Рассчитай для каждого слова:
- cost = kanji_count;
- value = frequency × 2 + 3.

Затем примени классический алгоритм 0/1 Knapsack с бюджетом 10 единиц. Каждое слово можно выбрать не более одного раза.

Цель - максимизировать суммарную value при условии, что суммарная cost ≤ 10.

В итоговом ответе должны быть только выбранные слова и итоговые cost и value. Используй следующий формат: words: слово1, слово2, слово3 cost: число value: число."""


_STEP_BY_STEP_TEMPLATE = """Ты получаешь один японский кандзи.

Кандзи: {KANJI}

Сформируй набор распространённых японских слов, содержащих этот кандзи.

Для каждого слова определи:
- слово;
- чтение;
- перевод;
- frequency - частотность от 1 до 5, где 5 означает наиболее распространённое слово;
- kanji_count - количество кандзи в слове.

Рассчитай для каждого слова:
- cost = kanji_count;
- value = frequency × 2 + 3.

Затем примени классический алгоритм 0/1 Knapsack с бюджетом 10 единиц. Каждое слово можно выбрать не более одного раза.

Цель - максимизировать суммарную value при условии, что суммарная cost ≤ 10.

Решай задачу пошагово:
1. сформируй кандидатов;
2. определи их параметры;
3. рассчитай cost и value;
4. примени 0/1 Knapsack;
5. проверь полученное решение.

В итоговом ответе должны быть только выбранные слова и итоговые cost и value. Используй следующий формат: words: слово1, слово2, слово3 cost: число value: число."""


_PROMPT_TEMPLATE = """Ты - эксперт по prompt engineering.

Составь оптимальный prompt для решения следующей задачи.

На вход подаётся один японский кандзи.

Кандзи: {KANJI}

Нужно сформировать набор распространённых японских слов, содержащих этот кандзи.

Для каждого слова определить:
- слово;
- чтение;
- перевод;
- frequency от 1 до 5, где 5 означает наиболее распространённое слово;
- kanji_count — количество кандзи в слове.

Затем рассчитать:
- cost = kanji_count;
- value = frequency × 2 + 3.

После этого применить классический алгоритм 0/1 Knapsack с бюджетом 10 единиц. Каждое слово можно выбрать не более одного раза.

Цель - максимизировать суммарную value при условии, что суммарная cost не превышает 10.

Составь точный и однозначный prompt, который заставит модель выполнить всю задачу и вернуть итоговый выбранный набор слов с суммарными cost и value. А затем выполни этот prompt.

В итоговом ответе должны быть только выбранные слова и итоговые cost и value. Используй следующий формат: words: слово1, слово2, слово3 cost: число value: число Без объяснений."""


_EXPERTS_TEMPLATE = """Ты решаешь задачу с помощью группы экспертов.

Кандзи: {KANJI}

Задача:

Сформировать набор распространённых японских слов, содержащих данный кандзи.

Для каждого слова определить:
- слово;
- чтение;
- перевод;
- frequency - частотность от 1 до 5, где 5 означает наиболее распространённое слово;
- kanji_count - количество кандзи в слове.

Затем рассчитать:
- cost = kanji_count;
- value = frequency × 2 + 3.

После этого применить классический алгоритм 0/1 Knapsack с бюджетом 10 единиц.

Каждое слово можно выбрать не более одного раза.

Цель - максимизировать суммарную value при условии:

суммарная cost ≤ 10.

Создай трёх экспертов:

1. Аналитик - формирует список кандидатов и проверяет их параметры.
2. Инженер - применяет алгоритм 0/1 Knapsack и находит оптимальный набор.
3. Критик - проверяет слова, расчёты, ограничения и итоговое решение эксперта-инженера.

Каждый эксперт должен предоставить своё решение.

После этого сравни результаты экспертов и сформируй итоговое решение.

В итоговом ответе должны быть только выбранные слова и итоговые cost и value. Используй следующий формат: words: слово1, слово2, слово3 cost: число value: число Без объяснений.

Не меняй заданные формулы cost и value и бюджет 10."""


_PROMPT_TEMPLATES: dict[ExperimentType, str] = {
    ExperimentType.DIRECT: _DIRECT_TEMPLATE,
    ExperimentType.STEP_BY_STEP: _STEP_BY_STEP_TEMPLATE,
    ExperimentType.PROMPT: _PROMPT_TEMPLATE,
    ExperimentType.EXPERTS: _EXPERTS_TEMPLATE,
}

# Matches "words: a, b, c cost: N value: N" (case-insensitive, tolerant of
# newlines/extra whitespace, and of leading reasoning the model was told not
# to include but might anyway - we only ever surface the final match).
_RESULT_PATTERN = re.compile(
    r"words\s*:\s*(?P<words>.+?)\s*cost\s*:\s*(?P<cost>-?\d+)\s*value\s*:\s*(?P<value>-?\d+)",
    re.IGNORECASE | re.DOTALL,
)


def _build_prompt(experiment_type: ExperimentType, kanji: str) -> str:
    template = _PROMPT_TEMPLATES[experiment_type]
    return template.replace("{KANJI}", kanji)


def _parse_result(text: str) -> tuple[list[str], int, int]:
    match = _RESULT_PATTERN.search(text)

    if not match:
        logger.warning("Could not parse kanji word set result: %r", text)
        raise HTTPException(
            status_code=502,
            detail="Failed to parse AI response into words/cost/value",
        )

    words = [word.strip() for word in match.group("words").split(",") if word.strip()]

    if not words:
        logger.warning("Parsed empty word list from result: %r", text)
        raise HTTPException(
            status_code=502,
            detail="AI response did not contain any words",
        )

    return words, int(match.group("cost")), int(match.group("value"))


async def build_kanji_word_set(experiment_type: ExperimentType, kanji: str) -> KanjiWordSetResponse:
    prompt = _build_prompt(experiment_type, kanji)
    result_text = await generate_text(prompt)
    words, cost, value = _parse_result(result_text)

    return KanjiWordSetResponse(
        prompt=prompt,
        words=words,
        cost=cost,
        value=value,
    )
