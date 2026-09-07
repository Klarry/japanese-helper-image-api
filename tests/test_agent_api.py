from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.services import japanese_learning_agent as agent_module
from app.services.japanese_learning_agent import JapaneseLearningAgent

client = TestClient(app)


def _mock_generate_text(monkeypatch, response_text: str):
    async def fake_generate_text(prompt, temperature=None, model=None):
        return response_text

    monkeypatch.setattr(agent_module, "generate_text", fake_generate_text)


def test_agent_chat_accepts_a_message_and_returns_the_generated_response(monkeypatch):
    _mock_generate_text(monkeypatch, "Кандзи 学 значит «учиться».")

    response = client.post(
        "/agent/chat",
        json={"message": "Explain the kanji 学 in Russian and give an example."},
    )

    assert response.status_code == 200
    assert response.json() == {"response": "Кандзи 学 значит «учиться»."}


def test_agent_chat_delegates_to_the_agent(monkeypatch):
    calls = []

    async def fake_run(self, message):
        calls.append(message)
        return "answer from the agent"

    monkeypatch.setattr(JapaneseLearningAgent, "run", fake_run)

    response = client.post("/agent/chat", json={"message": "Explain 学."})

    assert response.status_code == 200
    assert response.json() == {"response": "answer from the agent"}
    assert calls == ["Explain 学."]


def test_agent_chat_rejects_an_empty_message():
    response = client.post("/agent/chat", json={"message": ""})

    assert response.status_code == 422


def test_agent_chat_rejects_a_blank_message():
    response = client.post("/agent/chat", json={"message": "   "})

    assert response.status_code == 422


def test_agent_chat_rejects_a_missing_message():
    response = client.post("/agent/chat", json={})

    assert response.status_code == 422


def test_agent_chat_propagates_gemini_errors(monkeypatch):
    async def fake_generate_text(prompt, temperature=None, model=None):
        raise HTTPException(status_code=502, detail="No text found in Gemini response")

    monkeypatch.setattr(agent_module, "generate_text", fake_generate_text)

    response = client.post("/agent/chat", json={"message": "Explain 学."})

    assert response.status_code == 502
