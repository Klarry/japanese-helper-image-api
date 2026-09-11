import asyncio

import pytest
from fastapi import HTTPException

from app.services import agent_facts_extractor as facts_module
from app.services.agent_facts_extractor import FactsExtractor
from app.services.gemini_service import GeneratedText


def _stub(monkeypatch, handler):
    """handler(prompt) -> GeneratedText, in place of the shared gemini call."""
    prompts = []

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        prompts.append(prompt)
        return handler(prompt)

    monkeypatch.setattr(facts_module, "generate_text_with_usage", fake_generate_text_with_usage)
    return prompts


def _generated(text='{"goal": "сдать N3"}', input_tokens=120, output_tokens=30):
    return GeneratedText(text=text, input_tokens=input_tokens, output_tokens=output_tokens)


def _update(extractor, facts=None, message="Хочу сдать N3 к декабрю."):
    return asyncio.run(extractor.update(facts or {}, message))


def test_a_json_object_becomes_the_new_facts(monkeypatch):
    _stub(monkeypatch, lambda prompt: _generated('{"goal": "сдать N3", "level": "N4"}'))

    result = _update(FactsExtractor())

    assert result.facts == {"goal": "сдать N3", "level": "N4"}


def test_the_prompt_carries_the_facts_so_far_and_the_newest_message(monkeypatch):
    prompts = _stub(monkeypatch, lambda prompt: _generated())

    _update(FactsExtractor(), facts={"goal": "сдать N3"}, message="Ещё хочу разговорную практику.")

    assert "сдать N3" in prompts[0]
    assert "Ещё хочу разговорную практику." in prompts[0]


def test_the_prompt_asks_for_what_stays_useful_later(monkeypatch):
    prompts = _stub(monkeypatch, lambda prompt: _generated())

    _update(FactsExtractor())

    for asked_for in ("goals", "constraints", "preferences", "decisions"):
        assert asked_for in prompts[0]


def test_the_extractor_uses_the_shared_gemini_service_once(monkeypatch):
    prompts = _stub(monkeypatch, lambda prompt: _generated())

    _update(FactsExtractor())

    assert len(prompts) == 1


def test_a_json_object_wrapped_in_a_code_fence_is_still_read(monkeypatch):
    _stub(monkeypatch, lambda prompt: _generated('```json\n{"goal": "сдать N3"}\n```'))

    assert _update(FactsExtractor()).facts == {"goal": "сдать N3"}


def test_facts_are_capped_at_the_limit(monkeypatch):
    many = ", ".join(f'"key{index}": "value"' for index in range(10))
    _stub(monkeypatch, lambda prompt: _generated("{" + many + "}"))

    assert len(_update(FactsExtractor(facts_limit=3)).facts) == 3


def test_nested_values_are_dropped_rather_than_stored_as_junk(monkeypatch):
    _stub(monkeypatch, lambda prompt: _generated('{"goal": "сдать N3", "plan": {"step": 1}}'))

    assert _update(FactsExtractor()).facts == {"goal": "сдать N3"}


def test_non_text_values_are_kept_as_text(monkeypatch):
    _stub(monkeypatch, lambda prompt: _generated('{"weekly_hours": 6}'))

    assert _update(FactsExtractor()).facts == {"weekly_hours": "6"}


def test_an_answer_that_is_not_json_is_rejected(monkeypatch):
    _stub(monkeypatch, lambda prompt: _generated("Конечно! Вот факты:"))

    with pytest.raises(ValueError):
        _update(FactsExtractor())


def test_an_answer_that_is_json_but_not_an_object_is_rejected(monkeypatch):
    _stub(monkeypatch, lambda prompt: _generated('["сдать N3"]'))

    with pytest.raises(ValueError):
        _update(FactsExtractor())


def test_a_gemini_failure_propagates_to_the_caller(monkeypatch):
    async def failing(prompt, model=None, temperature=None):
        raise HTTPException(status_code=502, detail="No text found in Gemini response")

    monkeypatch.setattr(facts_module, "generate_text_with_usage", failing)

    with pytest.raises(HTTPException):
        _update(FactsExtractor())


def test_the_update_reports_what_it_cost(monkeypatch):
    _stub(monkeypatch, lambda prompt: _generated(input_tokens=300, output_tokens=40))

    assert _update(FactsExtractor()).tokens_used == 340
