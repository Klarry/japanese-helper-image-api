from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.services import japanese_learning_agent as agent_module
from app.services.agent_history_storage import AgentHistoryStorage
from app.services.japanese_learning_agent import JapaneseLearningAgent

client = TestClient(app)


def _mock_generate_text(monkeypatch, response_text: str):
    async def fake_generate_text(prompt, temperature=None, model=None):
        return response_text

    monkeypatch.setattr(agent_module, "generate_text", fake_generate_text)


def _use_temp_history(monkeypatch, tmp_path):
    """Point the shared `agent` singleton at a throwaway history file so
    these API tests never touch the real data/agent_history.json."""
    temp_storage = AgentHistoryStorage(file_path=str(tmp_path / "history.json"))
    monkeypatch.setattr(agent_module.agent, "_history", temp_storage)
    return temp_storage


def test_agent_chat_accepts_a_message_and_returns_the_generated_response(monkeypatch, tmp_path):
    _use_temp_history(monkeypatch, tmp_path)
    _mock_generate_text(monkeypatch, "Кандзи 学 значит «учиться».")

    response = client.post(
        "/agent/chat",
        json={"message": "Explain the kanji 学 in Russian and give an example."},
    )

    assert response.status_code == 200
    assert response.json() == {"response": "Кандзи 学 значит «учиться»."}


def test_agent_chat_delegates_to_the_agent(monkeypatch, tmp_path):
    _use_temp_history(monkeypatch, tmp_path)
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


def test_agent_chat_propagates_gemini_errors(monkeypatch, tmp_path):
    _use_temp_history(monkeypatch, tmp_path)

    async def fake_generate_text(prompt, temperature=None, model=None):
        raise HTTPException(status_code=502, detail="No text found in Gemini response")

    monkeypatch.setattr(agent_module, "generate_text", fake_generate_text)

    response = client.post("/agent/chat", json={"message": "Explain 学."})

    assert response.status_code == 502


# --- GET/DELETE /agent/history ---------------------------------------------


def test_agent_history_is_empty_before_any_chat(monkeypatch, tmp_path):
    _use_temp_history(monkeypatch, tmp_path)

    response = client.get("/agent/history")

    assert response.status_code == 200
    assert response.json() == {"messages": []}


def test_agent_history_returns_messages_saved_by_chat(monkeypatch, tmp_path):
    _use_temp_history(monkeypatch, tmp_path)
    _mock_generate_text(monkeypatch, "学 means 'to study'.")

    client.post("/agent/chat", json={"message": "Explain the kanji 学."})
    response = client.get("/agent/history")

    assert response.status_code == 200
    assert response.json() == {
        "messages": [
            {"role": "user", "content": "Explain the kanji 学."},
            {"role": "assistant", "content": "学 means 'to study'."},
        ]
    }


def test_deleting_agent_history_clears_it(monkeypatch, tmp_path):
    _use_temp_history(monkeypatch, tmp_path)
    _mock_generate_text(monkeypatch, "answer")
    client.post("/agent/chat", json={"message": "Explain 学."})

    delete_response = client.delete("/agent/history")
    get_response = client.get("/agent/history")

    assert delete_response.status_code == 200
    assert delete_response.json() == {"messages": []}
    assert get_response.json() == {"messages": []}


def test_agent_history_survives_a_fresh_storage_instance(monkeypatch, tmp_path):
    """Same idea as the backend-restart test at the service layer, but
    through the API: history persisted via one storage instance must be
    readable through a newly constructed one pointed at the same file."""
    temp_storage = _use_temp_history(monkeypatch, tmp_path)
    _mock_generate_text(monkeypatch, "answer")
    client.post("/agent/chat", json={"message": "Explain 学."})

    monkeypatch.setattr(
        agent_module.agent,
        "_history",
        AgentHistoryStorage(file_path=str(tmp_path / "history.json")),
    )

    response = client.get("/agent/history")

    assert response.json() == {
        "messages": [
            {"role": "user", "content": "Explain 学."},
            {"role": "assistant", "content": "answer"},
        ]
    }
