from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.agent import AgentChatResponse, AgentCompressionStatus, AgentTokenUsage
from app.services import agent_history_compressor as compressor_module
from app.services import japanese_learning_agent as agent_module
from app.services.agent_history_storage import AgentHistoryStorage
from app.services.agent_usage_log import AgentUsageLog
from app.services.gemini_service import GeneratedText
from app.services.japanese_learning_agent import JapaneseLearningAgent

client = TestClient(app)


def _mock_generate(monkeypatch, response_text: str, input_tokens=10, output_tokens=5):
    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        return GeneratedText(text=response_text, input_tokens=input_tokens, output_tokens=output_tokens)

    monkeypatch.setattr(agent_module, "generate_text_with_usage", fake_generate_text_with_usage)


def _mock_count_tokens(monkeypatch, total_tokens: int):
    async def fake_count_tokens(text, model=None):
        return total_tokens

    monkeypatch.setattr(agent_module, "count_tokens", fake_count_tokens)


def _mock_summary(monkeypatch, summary_text: str):
    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        return GeneratedText(text=summary_text, input_tokens=100, output_tokens=20)

    monkeypatch.setattr(compressor_module, "generate_text_with_usage", fake_generate_text_with_usage)


def _isolate_agent(monkeypatch, tmp_path, compression_enabled=False):
    """Point the shared `agent` singleton at throwaway files, and state the
    compression mode explicitly, so these API tests never touch the real
    data/ files and never depend on the environment they run in."""
    temp_storage = AgentHistoryStorage(file_path=str(tmp_path / "history.json"))
    monkeypatch.setattr(agent_module.agent, "_history", temp_storage)
    monkeypatch.setattr(agent_module.agent, "_usage_log", AgentUsageLog(file_path=str(tmp_path / "usage.json")))
    monkeypatch.setattr(agent_module.agent, "_compression_enabled", compression_enabled)
    return temp_storage


def test_agent_chat_accepts_a_message_and_returns_the_generated_response(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "Кандзи 学 значит «учиться».", input_tokens=40, output_tokens=8)

    response = client.post(
        "/agent/chat",
        json={"message": "Explain the kanji 学 in Russian and give an example."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Кандзи 学 значит «учиться»."


def test_agent_chat_response_includes_real_token_usage(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer", input_tokens=40, output_tokens=8)

    response = client.post("/agent/chat", json={"message": "Explain 学."})

    assert response.json()["usage"] == {
        "current_request_tokens": 40,
        "history_tokens": 0,
        "response_tokens": 8,
        "total_tokens": 48,
    }


def test_agent_chat_delegates_to_the_agent(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    calls = []

    async def fake_run(self, message, compression_enabled=None):
        calls.append((message, compression_enabled))
        return AgentChatResponse(
            response="answer from the agent",
            usage=AgentTokenUsage(),
            compression=AgentCompressionStatus(),
        )

    monkeypatch.setattr(JapaneseLearningAgent, "run", fake_run)

    response = client.post("/agent/chat", json={"message": "Explain 学."})

    assert response.status_code == 200
    assert response.json()["response"] == "answer from the agent"
    # No mode in the request means "use however the server is configured".
    assert calls == [("Explain 学.", None)]


def test_agent_chat_passes_the_requested_mode_to_the_agent(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    calls = []

    async def fake_run(self, message, compression_enabled=None):
        calls.append((message, compression_enabled))
        return AgentChatResponse(
            response="answer",
            usage=AgentTokenUsage(),
            compression=AgentCompressionStatus(),
        )

    monkeypatch.setattr(JapaneseLearningAgent, "run", fake_run)

    client.post("/agent/chat", json={"message": "Explain 学.", "compression_enabled": True})

    assert calls == [("Explain 学.", True)]


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
    _isolate_agent(monkeypatch, tmp_path)

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        raise HTTPException(status_code=502, detail="No text found in Gemini response")

    monkeypatch.setattr(agent_module, "generate_text_with_usage", fake_generate_text_with_usage)

    response = client.post("/agent/chat", json={"message": "Explain 学."})

    assert response.status_code == 502


def test_agent_chat_reports_a_context_limit_error_as_a_client_error_not_a_crash(monkeypatch, tmp_path):
    """Scenario 3 at the API layer: a too-long conversation must come back
    as a proper HTTP error response, never an unhandled 500."""
    _isolate_agent(monkeypatch, tmp_path)

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        raise HTTPException(
            status_code=400,
            detail="The input token count exceeds the maximum number of tokens allowed",
        )

    monkeypatch.setattr(agent_module, "generate_text_with_usage", fake_generate_text_with_usage)

    response = client.post("/agent/chat", json={"message": "a very long message"})

    assert response.status_code == 400
    assert "exceeds the maximum" in response.json()["detail"]
    assert client.get("/agent/history").json() == {"messages": [], "summary": ""}


# --- GET/DELETE /agent/history ---------------------------------------------


def test_agent_history_is_empty_before_any_chat(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    response = client.get("/agent/history")

    assert response.status_code == 200
    assert response.json() == {"messages": [], "summary": ""}


def test_agent_history_returns_messages_saved_by_chat(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "学 means 'to study'.")

    client.post("/agent/chat", json={"message": "Explain the kanji 学."})
    response = client.get("/agent/history")

    assert response.status_code == 200
    assert response.json() == {
        "messages": [
            {"role": "user", "content": "Explain the kanji 学."},
            {"role": "assistant", "content": "学 means 'to study'."},
        ],
        "summary": "",
    }


def test_deleting_agent_history_clears_it(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer")
    client.post("/agent/chat", json={"message": "Explain 学."})

    delete_response = client.delete("/agent/history")
    get_response = client.get("/agent/history")

    assert delete_response.status_code == 200
    assert delete_response.json() == {"messages": [], "summary": ""}
    assert get_response.json() == {"messages": [], "summary": ""}


def test_agent_history_survives_a_fresh_storage_instance(monkeypatch, tmp_path):
    """Same idea as the backend-restart test at the service layer, but
    through the API: history persisted via one storage instance must be
    readable through a newly constructed one pointed at the same file."""
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer")
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
        ],
        "summary": "",
    }


# --- compression through the API -------------------------------------------


def _hold_a_conversation(turns):
    for index in range(turns):
        assert client.post("/agent/chat", json={"message": f"question {index}"}).status_code == 200


def test_history_reports_the_summary_and_the_recent_messages(monkeypatch, tmp_path):
    """Scenarios 2 and 4 through the API: after enough turns the stored
    conversation is a summary plus the six newest messages."""
    _isolate_agent(monkeypatch, tmp_path, compression_enabled=True)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    _mock_summary(monkeypatch, "Разбирали 学習 и 勉強.")

    _hold_a_conversation(turns=8)
    body = client.get("/agent/history").json()

    assert body["summary"] == "Разбирали 学習 и 勉強."
    assert len(body["messages"]) == 6


def test_history_after_a_restart_still_has_the_summary(monkeypatch, tmp_path):
    """Scenario 5 through the API."""
    _isolate_agent(monkeypatch, tmp_path, compression_enabled=True)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    _mock_summary(monkeypatch, "Разбирали 学習 и 勉強.")
    _hold_a_conversation(turns=8)

    monkeypatch.setattr(
        agent_module.agent,
        "_history",
        AgentHistoryStorage(file_path=str(tmp_path / "history.json")),
    )
    body = client.get("/agent/history").json()

    assert body["summary"] == "Разбирали 学習 и 勉強."
    assert len(body["messages"]) == 6


def test_deleting_the_history_clears_the_summary_too(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path, compression_enabled=True)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    _mock_summary(monkeypatch, "Разбирали 学習 и 勉強.")
    _hold_a_conversation(turns=8)

    client.delete("/agent/history")

    assert client.get("/agent/history").json() == {"messages": [], "summary": ""}


def test_compression_is_off_unless_it_is_switched_on(monkeypatch, tmp_path):
    """The default run keeps every message and never asks for a summary."""
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)

    _hold_a_conversation(turns=8)
    body = client.get("/agent/history").json()

    assert body["summary"] == ""
    assert len(body["messages"]) == 16


# --- GET /agent/usage ------------------------------------------------------


def test_usage_is_empty_before_any_chat(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    response = client.get("/agent/usage")

    assert response.status_code == 200
    assert response.json() == {"entries": []}


def test_usage_records_one_entry_per_request(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer", input_tokens=40, output_tokens=8)
    _mock_count_tokens(monkeypatch, 25)

    _hold_a_conversation(turns=3)
    entries = client.get("/agent/usage").json()["entries"]

    assert len(entries) == 3
    assert entries[0]["total_tokens"] == 48
    assert entries[0]["compression_enabled"] is False


def test_usage_entries_record_the_mode_they_were_produced_in(monkeypatch, tmp_path):
    """What makes the before/after comparison possible at all."""
    _isolate_agent(monkeypatch, tmp_path, compression_enabled=True)
    _mock_generate(monkeypatch, "answer", input_tokens=40, output_tokens=8)
    _mock_count_tokens(monkeypatch, 25)

    _hold_a_conversation(turns=1)
    entries = client.get("/agent/usage").json()["entries"]

    assert entries[0]["compression_enabled"] is True
    assert entries[0]["history_tokens"] == 0
    assert entries[0]["timestamp"]


# --- choosing the mode per request -----------------------------------------


def test_a_request_can_turn_compression_on_for_a_server_that_defaults_to_off(monkeypatch, tmp_path):
    """How the Android toggle works: the client states the mode, the server
    default is only the fallback."""
    _isolate_agent(monkeypatch, tmp_path, compression_enabled=False)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    _mock_summary(monkeypatch, "Разбирали 学習 и 勉強.")

    for index in range(8):
        client.post("/agent/chat", json={"message": f"question {index}", "compression_enabled": True})

    body = client.get("/agent/history").json()

    assert body["summary"] == "Разбирали 学習 и 勉強."
    assert len(body["messages"]) == 6


def test_a_request_can_turn_compression_off_for_a_server_that_defaults_to_on(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path, compression_enabled=True)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    _mock_summary(monkeypatch, "сводка")

    for index in range(8):
        client.post("/agent/chat", json={"message": f"question {index}", "compression_enabled": False})

    body = client.get("/agent/history").json()

    assert body["summary"] == ""
    assert len(body["messages"]) == 16


def test_the_response_reports_the_compression_status(monkeypatch, tmp_path):
    """What the screen shows under the token usage: the mode, the summary's
    real size, and how many messages are kept word for word."""
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    _mock_summary(monkeypatch, "Разбирали 学習 и 勉強.")

    for index in range(9):
        response = client.post(
            "/agent/chat",
            json={"message": f"question {index}", "compression_enabled": True},
        )

    assert response.json()["compression"] == {
        "enabled": True,
        "summary_tokens": 20,
        "messages_sent": 6,
    }


def test_the_status_reports_no_summary_while_compression_is_off(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer")

    response = client.post("/agent/chat", json={"message": "Explain 学.", "compression_enabled": False})

    # Nothing had been said yet, so nothing but the message itself was sent.
    assert response.json()["compression"] == {
        "enabled": False,
        "summary_tokens": 0,
        "messages_sent": 0,
    }


def test_the_usage_log_records_the_mode_the_request_asked_for(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path, compression_enabled=False)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 25)

    client.post("/agent/chat", json={"message": "off please", "compression_enabled": False})
    client.post("/agent/chat", json={"message": "on please", "compression_enabled": True})

    entries = client.get("/agent/usage").json()["entries"]

    assert [entry["compression_enabled"] for entry in entries] == [False, True]
