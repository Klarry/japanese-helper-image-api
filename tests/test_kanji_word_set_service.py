import asyncio

import pytest
from fastapi import HTTPException

from app.schemas.kanji_word_set import ExperimentType
from app.services import kanji_word_set_service as service


@pytest.mark.parametrize("experiment_type", list(ExperimentType))
def test_build_prompt_substitutes_kanji(experiment_type):
    prompt = service._build_prompt(experiment_type, "学")

    assert "{KANJI}" not in prompt
    assert "学" in prompt
    # Budget and formulas are fixed inside every template, never parameterized.
    assert "10" in prompt
    assert "cost" in prompt
    assert "value" in prompt


def test_build_prompt_does_not_leak_other_kanji():
    prompt = service._build_prompt(ExperimentType.DIRECT, "水")

    assert "水" in prompt
    assert "学" not in prompt


def test_build_prompt_experts_mentions_all_three_roles():
    prompt = service._build_prompt(ExperimentType.EXPERTS, "火")

    assert "Аналитик" in prompt
    assert "Инженер" in prompt
    assert "Критик" in prompt


@pytest.mark.parametrize(
    "text",
    [
        "words: 学生, 学校 cost: 4 value: 16",
        "words:学生,学校\ncost:4\nvalue:16",
        "Some reasoning...\nwords: 学生, 学校\ncost: 4\nvalue: 16",
        "WORDS: 学生, 学校 COST: 4 VALUE: 16",
    ],
)
def test_parse_result_success(text):
    words, cost, value = service._parse_result(text)

    assert words == ["学生", "学校"]
    assert cost == 4
    assert value == 16


def test_parse_result_missing_fields_raises_http_exception():
    with pytest.raises(HTTPException) as exc_info:
        service._parse_result("I cannot help with that.")

    assert exc_info.value.status_code == 502


def test_parse_result_no_words_raises_http_exception():
    with pytest.raises(HTTPException) as exc_info:
        service._parse_result("words:  ,  cost: 4 value: 16")

    assert exc_info.value.status_code == 502


def test_build_kanji_word_set_returns_structured_response(monkeypatch):
    captured_prompt = {}

    async def fake_generate_text(prompt: str) -> str:
        captured_prompt["value"] = prompt
        return "words: 学生, 学校 cost: 4 value: 16"

    monkeypatch.setattr(service, "generate_text", fake_generate_text)

    response = asyncio.run(service.build_kanji_word_set(ExperimentType.DIRECT, "学"))

    assert response.words == ["学生", "学校"]
    assert response.cost == 4
    assert response.value == 16
    assert response.prompt == captured_prompt["value"]
    assert "学" in response.prompt


def test_build_kanji_word_set_propagates_parse_error(monkeypatch):
    async def fake_generate_text(prompt: str) -> str:
        return "not in the expected format"

    monkeypatch.setattr(service, "generate_text", fake_generate_text)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(service.build_kanji_word_set(ExperimentType.STEP_BY_STEP, "学"))

    assert exc_info.value.status_code == 502
