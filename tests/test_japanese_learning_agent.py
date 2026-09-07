import asyncio

from app.services import japanese_learning_agent as agent_module
from app.services.japanese_learning_agent import JapaneseLearningAgent


def _stub_generate_text(monkeypatch, handler):
    calls = []

    async def fake_generate_text(prompt, temperature=None, model=None):
        calls.append(prompt)
        return handler(prompt)

    monkeypatch.setattr(agent_module, "generate_text", fake_generate_text)
    return calls


def test_run_builds_a_prompt_that_includes_the_learners_message(monkeypatch):
    calls = _stub_generate_text(monkeypatch, lambda prompt: "answer")

    asyncio.run(JapaneseLearningAgent().run("Explain the kanji 学."))

    assert len(calls) == 1
    assert "Explain the kanji 学." in calls[0]


def test_run_instructs_the_model_to_answer_in_russian_by_default(monkeypatch):
    calls = _stub_generate_text(monkeypatch, lambda prompt: "answer")

    asyncio.run(JapaneseLearningAgent().run("Explain the kanji 学."))

    assert "Russian" in calls[0]


def test_run_calls_the_existing_gemini_service(monkeypatch):
    calls = _stub_generate_text(monkeypatch, lambda prompt: "answer")

    asyncio.run(JapaneseLearningAgent().run("Explain the kanji 学."))

    assert len(calls) == 1


def test_run_returns_the_gemini_answer_unchanged(monkeypatch):
    _stub_generate_text(monkeypatch, lambda prompt: "Кандзи 学 значит «учиться».")

    result = asyncio.run(JapaneseLearningAgent().run("Explain the kanji 学."))

    assert result == "Кандзи 学 значит «учиться»."


def test_module_exposes_a_ready_to_use_agent_instance():
    assert isinstance(agent_module.agent, JapaneseLearningAgent)
