"""Day 17: how the agent decides whether a message needs a tool."""

import asyncio

import pytest

from app.services import agent_tool_planner as planner_module
from app.services.agent_tool_planner import MAX_CALLS, PlannedCall, ToolPlanner
from app.services.gemini_service import GeneratedText
from app.services.mcp_client import McpToolInfo

TOOL = McpToolInfo(
    name="get_japanese_word_info",
    title="Get Japanese word info",
    description="Look up a Japanese word or kanji in the JLPT vocabulary list.",
    parameters=("word",),
    input_schema={
        "type": "object",
        "properties": {"word": {"type": "string", "description": "A Japanese word or a single kanji"}},
        "required": ["word"],
    },
)


def _stub(monkeypatch, text, input_tokens=150, output_tokens=20):
    captured = {}

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        captured["prompt"] = prompt
        return GeneratedText(text=text, input_tokens=input_tokens, output_tokens=output_tokens)

    monkeypatch.setattr(planner_module, "generate_text_with_usage", fake_generate_text_with_usage)
    return captured


def _plan(message="Что означает 学習?", tools=(TOOL,), recent=()):
    return asyncio.run(ToolPlanner().plan(message, tools, recent))


def test_a_question_about_a_word_becomes_a_call_with_that_word(monkeypatch):
    _stub(monkeypatch, '{"calls": [{"tool": "get_japanese_word_info", "arguments": {"word": "学習"}}]}')

    plan = _plan()

    assert plan.calls == (PlannedCall("get_japanese_word_info", {"word": "学習"}),)
    assert plan.tokens_used == 170


def test_a_message_that_needs_no_lookup_plans_nothing(monkeypatch):
    _stub(monkeypatch, '{"calls": []}')

    assert _plan("Объясни грамматику 〜ながら").calls == ()


def test_the_model_is_shown_the_tools_exactly_as_the_server_listed_them(monkeypatch):
    """The catalogue in the prompt is list_tools() output, not a list written
    here: a tool the server adds is one the agent can use."""
    captured = _stub(monkeypatch, '{"calls": []}')

    _plan()

    assert "get_japanese_word_info: Look up a Japanese word or kanji" in captured["prompt"]
    assert '"description": "A Japanese word or a single kanji"' in captured["prompt"]
    assert "Learner's message:\nЧто означает 学習?" in captured["prompt"]


def test_the_recent_turn_is_given_so_a_follow_up_can_name_its_word(monkeypatch):
    captured = _stub(monkeypatch, '{"calls": []}')

    _plan("А какой у него уровень JLPT?", recent=[{"role": "user", "content": "Что значит 学習?"}])

    assert "user: Что значит 学習?" in captured["prompt"]


def test_no_tools_means_no_call_to_gemini(monkeypatch):
    async def must_not_run(prompt, model=None, temperature=None):
        raise AssertionError("the planner asked Gemini with nothing to offer")

    monkeypatch.setattr(planner_module, "generate_text_with_usage", must_not_run)

    assert _plan(tools=()).calls == ()


def test_a_call_to_a_tool_the_server_does_not_have_is_dropped(monkeypatch):
    _stub(
        monkeypatch,
        '{"calls": [{"tool": "delete_everything", "arguments": {}},'
        ' {"tool": "get_japanese_word_info", "arguments": {"word": "学"}}]}',
    )

    assert _plan().calls == (PlannedCall("get_japanese_word_info", {"word": "学"}),)


def test_a_call_without_an_arguments_object_is_dropped(monkeypatch):
    _stub(monkeypatch, '{"calls": [{"tool": "get_japanese_word_info", "arguments": "学習"}]}')

    assert _plan().calls == ()


def test_the_number_of_calls_is_capped(monkeypatch):
    call = '{"tool": "get_japanese_word_info", "arguments": {"word": "学"}}'
    _stub(monkeypatch, '{"calls": [' + ", ".join([call] * 6) + "]}")

    assert len(_plan().calls) == MAX_CALLS


def test_a_plan_in_a_code_fence_still_parses(monkeypatch):
    _stub(monkeypatch, '```json\n{"calls": [{"tool": "get_japanese_word_info", "arguments": {"word": "学習"}}]}\n```')

    assert len(_plan().calls) == 1


@pytest.mark.parametrize("answer", ["not json", "[]", '{"tool": "get_japanese_word_info"}'])
def test_an_answer_that_is_not_a_plan_raises(monkeypatch, answer):
    _stub(monkeypatch, answer)

    with pytest.raises(ValueError):
        _plan()
