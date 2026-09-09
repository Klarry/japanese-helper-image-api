import asyncio

import pytest
from fastapi import HTTPException

from app.services import japanese_learning_agent as agent_module
from app.services.agent_history_storage import AgentHistoryStorage
from app.services.gemini_service import GeneratedText
from app.services.japanese_learning_agent import JapaneseLearningAgent


def _stub_generate(monkeypatch, handler):
    """handler(prompt) -> GeneratedText. Replaces generate_text_with_usage."""
    calls = []

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        calls.append(prompt)
        return handler(prompt)

    monkeypatch.setattr(agent_module, "generate_text_with_usage", fake_generate_text_with_usage)
    return calls


def _stub_count_tokens(monkeypatch, handler):
    """handler(text) -> int. Replaces count_tokens."""
    calls = []

    async def fake_count_tokens(text, model=None):
        calls.append(text)
        return handler(text)

    monkeypatch.setattr(agent_module, "count_tokens", fake_count_tokens)
    return calls


def _generated(text="answer", input_tokens=10, output_tokens=5):
    return GeneratedText(text=text, input_tokens=input_tokens, output_tokens=output_tokens)


def _agent(tmp_path, name="history.json"):
    return JapaneseLearningAgent(
        history_storage=AgentHistoryStorage(file_path=str(tmp_path / name))
    )


# --- prompt building / persistence (unchanged behaviour) -------------------


def test_run_builds_a_prompt_that_includes_the_learners_message(monkeypatch, tmp_path):
    calls = _stub_generate(monkeypatch, lambda prompt: _generated())

    asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert len(calls) == 1
    assert "Explain the kanji 学." in calls[0]


def test_run_instructs_the_model_to_answer_in_russian_by_default(monkeypatch, tmp_path):
    calls = _stub_generate(monkeypatch, lambda prompt: _generated())

    asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert "Russian" in calls[0]


def test_run_calls_the_existing_gemini_service(monkeypatch, tmp_path):
    calls = _stub_generate(monkeypatch, lambda prompt: _generated())

    asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert len(calls) == 1


def test_run_returns_the_gemini_answer_unchanged(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, lambda prompt: _generated(text="Кандзи 学 значит «учиться»."))

    result = asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert result.response == "Кандзи 学 значит «учиться»."


def test_module_exposes_a_ready_to_use_agent_instance():
    assert isinstance(agent_module.agent, JapaneseLearningAgent)


def test_run_with_no_prior_history_does_not_mention_a_conversation(monkeypatch, tmp_path):
    calls = _stub_generate(monkeypatch, lambda prompt: _generated())

    asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert "Conversation so far" not in calls[0]


def test_run_saves_both_the_user_message_and_the_assistant_response(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, lambda prompt: _generated(text="学 means 'to study'."))
    agent = _agent(tmp_path)

    asyncio.run(agent.run("Explain the kanji 学."))

    assert agent.get_history() == [
        {"role": "user", "content": "Explain the kanji 学."},
        {"role": "assistant", "content": "学 means 'to study'."},
    ]


def test_run_includes_previous_turns_in_the_next_gemini_request(monkeypatch, tmp_path):
    responses = iter([_generated(text="学 means 'to study'."), _generated(text="学校で使う言葉です。")])
    calls = _stub_generate(monkeypatch, lambda prompt: next(responses))
    _stub_count_tokens(monkeypatch, lambda text: 50)
    agent = _agent(tmp_path)

    asyncio.run(agent.run("Explain the kanji 学."))
    asyncio.run(agent.run("Give me another example sentence for it."))

    second_prompt = calls[1]
    assert "Explain the kanji 学." in second_prompt
    assert "学 means 'to study'." in second_prompt
    assert "Give me another example sentence for it." in second_prompt


def test_history_survives_agent_recreation(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, lambda prompt: _generated())
    file_path = str(tmp_path / "history.json")
    first_agent = JapaneseLearningAgent(history_storage=AgentHistoryStorage(file_path=file_path))

    asyncio.run(first_agent.run("Explain the kanji 学."))

    restarted_agent = JapaneseLearningAgent(history_storage=AgentHistoryStorage(file_path=file_path))

    assert restarted_agent.get_history() == [
        {"role": "user", "content": "Explain the kanji 学."},
        {"role": "assistant", "content": "answer"},
    ]


def test_run_does_not_persist_anything_when_gemini_fails(monkeypatch, tmp_path):
    async def failing_generate(prompt, model=None, temperature=None):
        raise HTTPException(status_code=502, detail="No text found in Gemini response")

    monkeypatch.setattr(agent_module, "generate_text_with_usage", failing_generate)
    agent = _agent(tmp_path)

    with pytest.raises(HTTPException):
        asyncio.run(agent.run("Explain the kanji 学."))

    assert agent.get_history() == []


def test_clear_history_removes_all_messages(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, lambda prompt: _generated())
    agent = _agent(tmp_path)
    asyncio.run(agent.run("Explain the kanji 学."))

    agent.clear_history()

    assert agent.get_history() == []


# --- token usage -------------------------------------------------------


def test_the_first_message_reports_zero_history_tokens_without_calling_count_tokens(monkeypatch, tmp_path):
    """Scenario 1 (short dialogue): nothing has been said yet, so history is
    genuinely zero - no need to ask Gemini to count tokens in an empty string.
    """
    _stub_generate(monkeypatch, lambda prompt: _generated(input_tokens=30, output_tokens=12))
    count_calls = _stub_count_tokens(monkeypatch, lambda text: 999)

    result = asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert count_calls == []
    assert result.usage.history_tokens == 0
    assert result.usage.current_request_tokens == 30
    assert result.usage.response_tokens == 12
    assert result.usage.total_tokens == 42


def test_history_tokens_come_from_a_real_count_tokens_call_on_the_history_prefix(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, lambda prompt: _generated(input_tokens=100, output_tokens=20))
    count_calls = _stub_count_tokens(monkeypatch, lambda text: 64)
    agent = _agent(tmp_path)

    asyncio.run(agent.run("first message"))
    result = asyncio.run(agent.run("Give me another example for it."))

    assert len(count_calls) == 1
    assert "first message" in count_calls[0]
    assert "answer" in count_calls[0]
    assert "Conversation so far" in count_calls[0]
    # The new message itself is not part of the history being measured.
    assert "Give me another example for it." not in count_calls[0]
    assert result.usage.history_tokens == 64
    assert result.usage.current_request_tokens == 100 - 64
    assert result.usage.total_tokens == 120


def test_history_tokens_grow_as_the_conversation_grows(monkeypatch, tmp_path):
    """Scenario 2 (long dialogue): each additional turn makes the history
    prefix longer, so its real token count must keep increasing.
    """
    _stub_generate(monkeypatch, lambda prompt: _generated(input_tokens=200, output_tokens=20))
    # A real countTokens call on a longer prefix returns a bigger number -
    # simulate that by keying off the prefix's own length.
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    agent = _agent(tmp_path)

    history_token_counts = []
    for message in ["turn one", "turn two", "turn three", "turn four"]:
        result = asyncio.run(agent.run(message))
        history_token_counts.append(result.usage.history_tokens)

    assert history_token_counts == sorted(history_token_counts)
    assert history_token_counts[0] == 0
    assert history_token_counts[-1] > history_token_counts[0]
    assert len(set(history_token_counts)) > 1


def test_usage_is_null_where_gemini_does_not_report_prompt_tokens(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, lambda prompt: _generated(input_tokens=None, output_tokens=20))

    result = asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert result.usage.current_request_tokens is None
    assert result.usage.total_tokens is None
    assert result.usage.response_tokens == 20
    assert result.usage.history_tokens == 0


def test_a_history_token_count_failure_does_not_fail_the_chat_response(monkeypatch, tmp_path):
    """count_tokens is a best-effort measurement for the usage experiment -
    it must never take down a chat turn that otherwise succeeded.
    """
    _stub_generate(monkeypatch, lambda prompt: _generated(input_tokens=50, output_tokens=10))

    async def failing_count_tokens(text, model=None):
        raise HTTPException(status_code=502, detail="token count unavailable")

    monkeypatch.setattr(agent_module, "count_tokens", failing_count_tokens)
    agent = _agent(tmp_path)
    asyncio.run(agent.run("first message"))

    result = asyncio.run(agent.run("Give me another example for it."))

    assert result.response == "answer"
    assert result.usage.history_tokens is None
    assert result.usage.current_request_tokens is None
    # response tokens and history persistence are unaffected by the failure.
    assert result.usage.response_tokens == 10
    assert len(agent.get_history()) == 4


def test_a_context_limit_error_from_gemini_propagates_as_a_proper_error(monkeypatch, tmp_path):
    """Scenario 3: a dialogue too long for the model's context window must
    surface as a clean HTTP error, not a crash, and must not be persisted.
    """
    async def failing_generate(prompt, model=None, temperature=None):
        raise HTTPException(
            status_code=400,
            detail="The input token count exceeds the maximum number of tokens allowed",
        )

    monkeypatch.setattr(agent_module, "generate_text_with_usage", failing_generate)
    agent = _agent(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(agent.run("a very long message"))

    assert exc_info.value.status_code == 400
    assert agent.get_history() == []
