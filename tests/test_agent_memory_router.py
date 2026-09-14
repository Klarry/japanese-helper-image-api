import asyncio

import pytest

from app.services import agent_memory_router as router_module
from app.services.agent_memory import LongTermMemory, WorkingMemory
from app.services.agent_memory_router import MemoryRouter
from app.services.gemini_service import GeneratedText


def _mock_answer(monkeypatch, text: str, input_tokens=120, output_tokens=20):
    captured = {}

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        captured["prompt"] = prompt
        return GeneratedText(text=text, input_tokens=input_tokens, output_tokens=output_tokens)

    monkeypatch.setattr(router_module, "generate_text_with_usage", fake_generate_text_with_usage)
    return captured


def _route(message="сообщение", working=None, long_term=None):
    return asyncio.run(
        MemoryRouter().route(message, working or WorkingMemory(), long_term or LongTermMemory())
    )


def test_a_long_term_message_updates_only_long_term_memory(monkeypatch):
    _mock_answer(monkeypatch, '{"long_term": {"profile": {"favorite_word": "学習"}}}')

    routing = _route("Запомни надолго: моё любимое японское слово — 学習.")

    assert routing.long_term.profile == {"favorite_word": "学習"}
    assert routing.working is None
    assert routing.layers_written == ["long_term"]


def test_a_task_message_updates_only_working_memory(monkeypatch):
    _mock_answer(
        monkeypatch,
        '{"working": {"constraints": ["уровень N4"], "requirements": ["показывать перевод"]}}',
    )

    routing = _route("Для текущей задачи запомни: уровень N4.")

    assert routing.working.constraints == ["уровень N4"]
    assert routing.working.requirements == ["показывать перевод"]
    assert routing.long_term is None


def test_an_ordinary_request_writes_to_no_layer_at_all(monkeypatch):
    """A layer the router does not name is left as it was - which is not the
    same as being emptied."""
    _mock_answer(monkeypatch, "{}")

    routing = _route("Теперь объясни грамматику этого предложения.")

    assert routing.working is None
    assert routing.long_term is None
    assert routing.layers_written == []


def test_both_layers_can_change_at_once(monkeypatch):
    _mock_answer(
        monkeypatch,
        '{"working": {"goals": ["пример со словом"]}, "long_term": {"knowledge": ["любит 学習"]}}',
    )

    routing = _route()

    assert routing.layers_written == ["working", "long_term"]


def test_the_current_memory_is_given_to_the_model_so_it_merges_instead_of_replacing(monkeypatch):
    captured = _mock_answer(monkeypatch, "{}")

    _route(
        "ещё одно требование",
        working=WorkingMemory(goals=["уже поставленная цель"]),
        long_term=LongTermMemory(profile={"favorite_word": "学習"}),
    )

    assert "уже поставленная цель" in captured["prompt"]
    assert "学習" in captured["prompt"]
    assert "ещё одно требование" in captured["prompt"]


def test_the_router_is_told_never_to_write_to_short_term_memory(monkeypatch):
    captured = _mock_answer(monkeypatch, "{}")

    _route()

    assert "never write to it" in captured["prompt"]


def test_a_fenced_json_answer_is_still_understood(monkeypatch):
    _mock_answer(monkeypatch, '```json\n{"working": {"goals": ["цель"]}}\n```')

    assert (_route()).working.goals == ["цель"]


def test_an_answer_that_is_not_json_is_rejected(monkeypatch):
    _mock_answer(monkeypatch, "конечно, я запомню это!")

    with pytest.raises(ValueError):
        _route()


def test_an_answer_that_is_not_an_object_is_rejected(monkeypatch):
    _mock_answer(monkeypatch, '["учить", "слово"]')

    with pytest.raises(ValueError):
        _route()


def test_a_layer_that_is_not_an_object_is_ignored_rather_than_stored(monkeypatch):
    _mock_answer(monkeypatch, '{"working": "уровень N4"}')

    assert (_route()).working is None


def test_routing_reports_what_it_cost(monkeypatch):
    """The extra Gemini call this layer costs is visible in the usage log
    rather than hidden inside the answer's own numbers."""
    _mock_answer(monkeypatch, "{}", input_tokens=200, output_tokens=15)

    assert (_route()).tokens_used == 215
