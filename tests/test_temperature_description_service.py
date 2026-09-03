import asyncio

import pytest
from fastapi import HTTPException

from app.services import temperature_description_service as service


def test_build_prompt_substitutes_kanji():
    prompt = service._build_prompt("学")

    assert "{KANJI}" not in prompt
    assert "学" in prompt
    assert "N4-N3" in prompt


def test_build_prompt_does_not_leak_other_kanji():
    prompt = service._build_prompt("水")

    assert "水" in prompt
    assert "学" not in prompt


@pytest.mark.parametrize(
    "text",
    [
        "sentence: 学校に行きます。 translation: Я иду в школу.",
        "sentence:学校に行きます。\ntranslation:Я иду в школу.",
        "Some reasoning...\nsentence: 学校に行きます。\ntranslation: Я иду в школу.",
        "SENTENCE: 学校に行きます。 TRANSLATION: Я иду в школу.",
    ],
)
def test_parse_result_success(text):
    sentence, translation = service._parse_result(text)

    assert sentence == "学校に行きます。"
    assert translation == "Я иду в школу."


def test_parse_result_missing_fields_raises_http_exception():
    with pytest.raises(HTTPException) as exc_info:
        service._parse_result("I cannot help with that.")

    assert exc_info.value.status_code == 502


def test_parse_result_empty_translation_raises_http_exception():
    with pytest.raises(HTTPException) as exc_info:
        service._parse_result("sentence: 学校に行きます。 translation:   ")

    assert exc_info.value.status_code == 502


def test_build_temperature_description_passes_temperature_through(monkeypatch):
    captured = {}

    async def fake_generate_text(prompt: str, temperature: float | None = None) -> str:
        captured["prompt"] = prompt
        captured["temperature"] = temperature
        return "sentence: 学校に行きます。 translation: Я иду в школу."

    monkeypatch.setattr(service, "generate_text", fake_generate_text)

    response = asyncio.run(service.build_temperature_description("学", 0.7))

    assert response.sentence == "学校に行きます。"
    assert response.translation == "Я иду в школу."
    assert response.temperature == 0.7
    assert captured["temperature"] == 0.7
    assert "学" in captured["prompt"]


def test_build_temperature_description_propagates_parse_error(monkeypatch):
    async def fake_generate_text(prompt: str, temperature: float | None = None) -> str:
        return "not in the expected format"

    monkeypatch.setattr(service, "generate_text", fake_generate_text)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(service.build_temperature_description("学", 1.2))

    assert exc_info.value.status_code == 502
