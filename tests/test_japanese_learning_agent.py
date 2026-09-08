import asyncio

import pytest
from fastapi import HTTPException

from app.services import japanese_learning_agent as agent_module
from app.services.agent_history_storage import AgentHistoryStorage
from app.services.japanese_learning_agent import JapaneseLearningAgent


def _stub_generate_text(monkeypatch, handler):
    calls = []

    async def fake_generate_text(prompt, temperature=None, model=None):
        calls.append(prompt)
        return handler(prompt)

    monkeypatch.setattr(agent_module, "generate_text", fake_generate_text)
    return calls


def _agent(tmp_path, name="history.json"):
    return JapaneseLearningAgent(
        history_storage=AgentHistoryStorage(file_path=str(tmp_path / name))
    )


def test_run_builds_a_prompt_that_includes_the_learners_message(monkeypatch, tmp_path):
    calls = _stub_generate_text(monkeypatch, lambda prompt: "answer")

    asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert len(calls) == 1
    assert "Explain the kanji 学." in calls[0]


def test_run_instructs_the_model_to_answer_in_russian_by_default(monkeypatch, tmp_path):
    calls = _stub_generate_text(monkeypatch, lambda prompt: "answer")

    asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert "Russian" in calls[0]


def test_run_calls_the_existing_gemini_service(monkeypatch, tmp_path):
    calls = _stub_generate_text(monkeypatch, lambda prompt: "answer")

    asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert len(calls) == 1


def test_run_returns_the_gemini_answer_unchanged(monkeypatch, tmp_path):
    _stub_generate_text(monkeypatch, lambda prompt: "Кандзи 学 значит «учиться».")

    result = asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert result == "Кандзи 学 значит «учиться»."


def test_module_exposes_a_ready_to_use_agent_instance():
    assert isinstance(agent_module.agent, JapaneseLearningAgent)


# --- persistent conversation context -------------------------------------


def test_run_with_no_prior_history_does_not_mention_a_conversation(monkeypatch, tmp_path):
    calls = _stub_generate_text(monkeypatch, lambda prompt: "answer")

    asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert "Conversation so far" not in calls[0]


def test_run_saves_both_the_user_message_and_the_assistant_response(monkeypatch, tmp_path):
    _stub_generate_text(monkeypatch, lambda prompt: "学 means 'to study'.")
    agent = _agent(tmp_path)

    asyncio.run(agent.run("Explain the kanji 学."))

    assert agent.get_history() == [
        {"role": "user", "content": "Explain the kanji 学."},
        {"role": "assistant", "content": "学 means 'to study'."},
    ]


def test_run_includes_previous_turns_in_the_next_gemini_request(monkeypatch, tmp_path):
    responses = iter(["学 means 'to study'.", "学校で使う言葉です。"])
    calls = _stub_generate_text(monkeypatch, lambda prompt: next(responses))
    agent = _agent(tmp_path)

    asyncio.run(agent.run("Explain the kanji 学."))
    asyncio.run(agent.run("Give me another example sentence for it."))

    second_prompt = calls[1]
    assert "Explain the kanji 学." in second_prompt
    assert "学 means 'to study'." in second_prompt
    assert "Give me another example sentence for it." in second_prompt


def test_history_survives_agent_recreation(monkeypatch, tmp_path):
    """Simulates a backend restart: a brand-new JapaneseLearningAgent
    pointed at the same history file must see the earlier conversation."""
    _stub_generate_text(monkeypatch, lambda prompt: "answer")
    file_path = str(tmp_path / "history.json")
    first_agent = JapaneseLearningAgent(history_storage=AgentHistoryStorage(file_path=file_path))

    asyncio.run(first_agent.run("Explain the kanji 学."))

    restarted_agent = JapaneseLearningAgent(history_storage=AgentHistoryStorage(file_path=file_path))

    assert restarted_agent.get_history() == [
        {"role": "user", "content": "Explain the kanji 学."},
        {"role": "assistant", "content": "answer"},
    ]


def test_run_does_not_persist_anything_when_gemini_fails(monkeypatch, tmp_path):
    async def failing_generate_text(prompt, temperature=None, model=None):
        raise HTTPException(status_code=502, detail="No text found in Gemini response")

    monkeypatch.setattr(agent_module, "generate_text", failing_generate_text)
    agent = _agent(tmp_path)

    with pytest.raises(HTTPException):
        asyncio.run(agent.run("Explain the kanji 学."))

    assert agent.get_history() == []


def test_clear_history_removes_all_messages(monkeypatch, tmp_path):
    _stub_generate_text(monkeypatch, lambda prompt: "answer")
    agent = _agent(tmp_path)
    asyncio.run(agent.run("Explain the kanji 学."))

    agent.clear_history()

    assert agent.get_history() == []
