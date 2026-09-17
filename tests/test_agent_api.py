from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.agent import (
    AgentChatResponse,
    AgentCompressionStatus,
    AgentTokenUsage,
    ContextStrategy,
)
from app.services import agent_facts_extractor as facts_module
from app.services import agent_history_compressor as compressor_module
from app.services import agent_memory_router as memory_router_module
from app.services import agent_task_tracker as task_tracker_module
from app.services import japanese_learning_agent as agent_module
from app.services.agent_history_storage import AgentHistoryStorage
from app.services.agent_invariants import AgentInvariantsStorage
from app.services.agent_memory import AgentLongTermMemoryStorage
from app.services.agent_user_profile import AgentUserProfileStorage
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
    monkeypatch.setattr(
        agent_module.agent,
        "_long_term",
        AgentLongTermMemoryStorage(file_path=str(tmp_path / "long_term.json")),
    )
    monkeypatch.setattr(
        agent_module.agent,
        "_profile",
        AgentUserProfileStorage(file_path=str(tmp_path / "profile.json")),
    )
    monkeypatch.setattr(
        agent_module.agent,
        "_invariants",
        AgentInvariantsStorage(file_path=str(tmp_path / "invariants.json")),
    )
    monkeypatch.setattr(agent_module.agent, "_compression_enabled", compression_enabled)
    # Off unless a test is about it, so no other test pays for the extra
    # Gemini call - the same rule the compression flag above follows.
    monkeypatch.setattr(agent_module.agent, "_task_tracking_enabled", False)
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

    async def fake_run(self, message, compression_enabled=None, strategy=None):
        calls.append((message, compression_enabled, strategy))
        return AgentChatResponse(
            response="answer from the agent",
            usage=AgentTokenUsage(),
            compression=AgentCompressionStatus(),
        )

    monkeypatch.setattr(JapaneseLearningAgent, "run", fake_run)

    response = client.post("/agent/chat", json={"message": "Explain 学."})

    assert response.status_code == 200
    assert response.json()["response"] == "answer from the agent"
    # Nothing in the request means "use however the server is configured".
    assert calls == [("Explain 学.", None, None)]


def test_agent_chat_passes_the_requested_mode_to_the_agent(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    calls = []

    async def fake_run(self, message, compression_enabled=None, strategy=None):
        calls.append((message, compression_enabled, strategy))
        return AgentChatResponse(
            response="answer",
            usage=AgentTokenUsage(),
            compression=AgentCompressionStatus(),
        )

    monkeypatch.setattr(JapaneseLearningAgent, "run", fake_run)

    client.post("/agent/chat", json={"message": "Explain 学.", "compression_enabled": True})
    client.post("/agent/chat", json={"message": "Explain 学.", "strategy": "sliding_window"})

    assert calls == [
        ("Explain 学.", True, None),
        ("Explain 学.", None, ContextStrategy.SLIDING_WINDOW),
    ]


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


# --- choosing a strategy ---------------------------------------------------


def _mock_facts(monkeypatch, facts_json: str):
    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        return GeneratedText(text=facts_json, input_tokens=200, output_tokens=30)

    monkeypatch.setattr(facts_module, "generate_text_with_usage", fake_generate_text_with_usage)


def test_the_strategy_can_be_chosen_and_read_back(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    response = client.put("/agent/strategy", json={"strategy": "sliding_window"})

    assert response.status_code == 200
    assert response.json() == {"strategy": "sliding_window"}
    assert client.get("/agent/context").json()["strategy"] == "sliding_window"


def test_an_unknown_strategy_is_rejected(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    assert client.put("/agent/strategy", json={"strategy": "telepathy"}).status_code == 422


def test_the_chosen_strategy_answers_later_chats(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    client.put("/agent/strategy", json={"strategy": "sliding_window"})

    response = client.post("/agent/chat", json={"message": "вопрос"})

    assert response.json()["strategy"] == "sliding_window"


def test_a_chat_can_name_the_strategy_for_that_request_only(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    client.put("/agent/strategy", json={"strategy": "sliding_window"})

    response = client.post("/agent/chat", json={"message": "вопрос", "strategy": "full"})

    assert response.json()["strategy"] == "full"
    assert client.get("/agent/context").json()["strategy"] == "sliding_window"


# --- GET /agent/context ----------------------------------------------------


def test_context_is_empty_before_anything_is_said(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    body = client.get("/agent/context").json()

    assert body["branch"] == "main"
    assert body["branches"] == ["main"]
    assert body["checkpoints"] == []
    assert body["messages"] == []
    assert body["context"] == ""


def test_context_reports_what_the_next_request_would_send(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    client.put("/agent/strategy", json={"strategy": "sliding_window"})

    for index in range(10):
        client.post("/agent/chat", json={"message": f"question {index}"})

    body = client.get("/agent/context").json()

    assert len(body["messages"]) == 6
    assert "question 9" in body["context"]
    assert "question 0" not in body["context"]


def test_context_reports_the_sticky_facts(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    _mock_facts(monkeypatch, '{"goal": "сдать N3"}')
    client.put("/agent/strategy", json={"strategy": "sticky_facts"})

    client.post("/agent/chat", json={"message": "Хочу сдать N3."})
    body = client.get("/agent/context").json()

    assert body["facts"] == {"goal": "сдать N3"}
    assert "goal: сдать N3" in body["context"]


# --- checkpoints and branches ----------------------------------------------


def _say(message: str):
    assert client.post("/agent/chat", json={"message": message}).status_code == 200


def test_a_checkpoint_can_be_taken_without_naming_it(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    _say("общий вопрос")

    response = client.post("/agent/checkpoint")

    assert response.status_code == 200
    assert response.json() == {"name": "cp-1", "branch": "main", "messages": 2}


def test_two_branches_forked_from_one_checkpoint_stay_apart(monkeypatch, tmp_path):
    """The whole branching scenario through the API."""
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    _say("общий вопрос")
    client.post("/agent/checkpoint", json={"name": "fork"})

    assert client.post("/agent/branch", json={"name": "formal", "checkpoint": "fork"}).status_code == 200
    assert client.post("/agent/branch", json={"name": "casual", "checkpoint": "fork"}).status_code == 200

    client.put("/agent/branch", json={"name": "formal"})
    _say("формальный вопрос")
    client.put("/agent/branch", json={"name": "casual"})
    _say("разговорный вопрос")

    client.put("/agent/branch", json={"name": "formal"})
    formal = client.get("/agent/context").json()
    client.put("/agent/branch", json={"name": "casual"})
    casual = client.get("/agent/context").json()

    assert formal["branch"] == "formal"
    assert "формальный вопрос" in formal["context"]
    assert "разговорный вопрос" not in formal["context"]
    assert casual["branch"] == "casual"
    assert "разговорный вопрос" in casual["context"]
    assert "формальный вопрос" not in casual["context"]
    assert casual["branches"] == ["casual", "formal", "main"]


def test_creating_a_branch_leaves_the_current_one_alone(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    _say("общий вопрос")
    client.post("/agent/checkpoint", json={"name": "fork"})

    body = client.post("/agent/branch", json={"name": "formal", "checkpoint": "fork"}).json()

    assert body == {"branch": "main", "branches": ["formal", "main"]}


def test_forking_from_a_checkpoint_that_does_not_exist_is_reported(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    response = client.post("/agent/branch", json={"name": "formal", "checkpoint": "nope"})

    assert response.status_code == 404


def test_reusing_a_branch_name_is_reported(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    client.post("/agent/checkpoint", json={"name": "fork"})
    client.post("/agent/branch", json={"name": "formal", "checkpoint": "fork"})

    response = client.post("/agent/branch", json={"name": "formal", "checkpoint": "fork"})

    assert response.status_code == 409


def test_switching_to_a_branch_that_does_not_exist_is_reported(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    assert client.put("/agent/branch", json={"name": "nope"}).status_code == 404


def test_history_follows_the_branch_that_was_switched_to(monkeypatch, tmp_path):
    """GET /agent/history keeps meaning the current conversation - which is
    now the current branch's."""
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    _say("общий вопрос")
    client.post("/agent/checkpoint", json={"name": "fork"})
    client.post("/agent/branch", json={"name": "side", "checkpoint": "fork"})
    client.put("/agent/branch", json={"name": "side"})
    _say("вопрос в ветке")

    side = [entry["content"] for entry in client.get("/agent/history").json()["messages"]]
    client.put("/agent/branch", json={"name": "main"})
    main = [entry["content"] for entry in client.get("/agent/history").json()["messages"]]

    assert "вопрос в ветке" in side
    assert "вопрос в ветке" not in main


def test_branches_survive_a_fresh_storage_instance(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer")
    _mock_count_tokens(monkeypatch, 120)
    _say("общий вопрос")
    client.post("/agent/checkpoint", json={"name": "fork"})
    client.post("/agent/branch", json={"name": "side", "checkpoint": "fork"})
    client.put("/agent/branch", json={"name": "side"})

    monkeypatch.setattr(
        agent_module.agent,
        "_history",
        AgentHistoryStorage(file_path=str(tmp_path / "history.json")),
    )
    body = client.get("/agent/context").json()

    assert body["branch"] == "side"
    assert body["branches"] == ["main", "side"]
    assert body["checkpoints"] == ["fork"]


# --- comparing the strategies ----------------------------------------------


def test_usage_entries_say_which_strategy_produced_them(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "answer", input_tokens=40, output_tokens=8)
    _mock_count_tokens(monkeypatch, 25)

    client.post("/agent/chat", json={"message": "вопрос", "strategy": "full"})
    client.post("/agent/chat", json={"message": "вопрос", "strategy": "sliding_window"})

    entries = client.get("/agent/usage").json()["entries"]

    assert [entry["strategy"] for entry in entries] == ["full", "sliding_window"]


# --- memory layers (Day 11) ------------------------------------------------


def _mock_memory_router(monkeypatch, *answers: str):
    """Answer the router's Gemini call with each JSON in turn, repeating the
    last one - so a scenario can say what each message routes to."""
    prompts = []
    queue = list(answers) or ["{}"]

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        prompts.append(prompt)
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        return GeneratedText(text=answer, input_tokens=150, output_tokens=25)

    monkeypatch.setattr(memory_router_module, "generate_text_with_usage", fake_generate_text_with_usage)
    return prompts


def _say_with_memory(message: str):
    response = client.post("/agent/chat", json={"message": message, "strategy": "layered_memory"})
    assert response.status_code == 200
    return response


def _memory():
    return client.get("/agent/memory").json()


def _scenario(monkeypatch, tmp_path):
    """The three-layer scenario: one message aimed at long-term memory, one
    at the current task, then three ordinary turns that only continue the
    conversation."""
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ агента")
    _mock_count_tokens(monkeypatch, 90)
    client.put("/agent/strategy", json={"strategy": "layered_memory"})
    _mock_memory_router(
        monkeypatch,
        '{"long_term": {"profile": {"favorite_word": "学習"}}}',
        '{"working": {"constraints": ["уровень N4", "минималистичный интерфейс"],'
        ' "requirements": ["обязательно показывать перевод"]}}',
        "{}",
    )
    _say_with_memory("Запомни надолго: моё любимое японское слово — 学習.")
    _say_with_memory(
        "Для текущей задачи запомни: уровень N4, минималистичный интерфейс "
        "и обязательно показывать перевод."
    )
    _say_with_memory("Давай создадим пример предложения со словом 学習.")
    _say_with_memory("Сделай его уровня N4.")
    _say_with_memory("Теперь объясни грамматику этого предложения.")


def test_each_message_lands_in_the_layer_it_was_aimed_at(monkeypatch, tmp_path):
    _scenario(monkeypatch, tmp_path)

    memory = _memory()

    assert memory["long_term"]["profile"] == {"favorite_word": "学習"}
    assert memory["working"]["constraints"] == ["уровень N4", "минималистичный интерфейс"]
    assert memory["working"]["requirements"] == ["обязательно показывать перевод"]
    assert len(memory["short_term"]["messages"]) == 10


def test_the_layers_do_not_leak_into_each_other(monkeypatch, tmp_path):
    _scenario(monkeypatch, tmp_path)

    memory = _memory()

    assert memory["long_term"]["preferences"] == []
    assert memory["long_term"]["knowledge"] == []
    assert "学習" not in str(memory["working"])
    assert "N4" not in str(memory["long_term"])
    # The conversation is not copied into the other two layers - they hold
    # what was distilled from it, not a second transcript.
    assert "объясни грамматику" not in str(memory["working"])
    assert "объясни грамматику" not in str(memory["long_term"])


def test_an_ordinary_request_leaves_the_other_layers_untouched(monkeypatch, tmp_path):
    _scenario(monkeypatch, tmp_path)
    before = _memory()

    _say_with_memory("И ещё один пример, пожалуйста.")
    after = _memory()

    assert after["working"] == before["working"]
    assert after["long_term"] == before["long_term"]
    assert len(after["short_term"]["messages"]) == len(before["short_term"]["messages"]) + 2


def test_every_layer_is_named_in_the_prompt_that_is_sent(monkeypatch, tmp_path):
    """The layers are handed to the model as labelled sections rather than
    merged into one history."""
    _isolate_agent(monkeypatch, tmp_path)
    prompts = []

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        prompts.append(prompt)
        return GeneratedText(text="ответ", input_tokens=40, output_tokens=8)

    monkeypatch.setattr(agent_module, "generate_text_with_usage", fake_generate_text_with_usage)
    _mock_count_tokens(monkeypatch, 90)
    _mock_memory_router(monkeypatch, '{"long_term": {"profile": {"favorite_word": "学習"}}}', "{}")
    _say_with_memory("Запомни надолго: моё любимое японское слово — 学習.")
    _say_with_memory("Давай создадим пример предложения.")

    assert "LONG-TERM MEMORY" in prompts[-1]
    assert "SHORT-TERM MEMORY" in prompts[-1]


def test_clearing_short_term_memory_keeps_the_task_and_the_learner(monkeypatch, tmp_path):
    _scenario(monkeypatch, tmp_path)

    memory = client.delete("/agent/memory/short_term").json()

    assert memory["short_term"]["messages"] == []
    assert memory["working"]["constraints"] == ["уровень N4", "минималистичный интерфейс"]
    assert memory["long_term"]["profile"] == {"favorite_word": "学習"}


def test_clearing_working_memory_keeps_the_conversation_and_the_learner(monkeypatch, tmp_path):
    _scenario(monkeypatch, tmp_path)

    memory = client.delete("/agent/memory/working").json()

    assert memory["working"] == {"goals": [], "requirements": [], "constraints": [], "decisions": []}
    assert len(memory["short_term"]["messages"]) == 10
    assert memory["long_term"]["profile"] == {"favorite_word": "学習"}


def test_clearing_long_term_memory_keeps_the_conversation_and_the_task(monkeypatch, tmp_path):
    _scenario(monkeypatch, tmp_path)

    memory = client.delete("/agent/memory/long_term").json()

    assert memory["long_term"]["profile"] == {}
    assert len(memory["short_term"]["messages"]) == 10
    assert memory["working"]["constraints"] == ["уровень N4", "минималистичный интерфейс"]


def test_a_cleared_layer_stops_being_sent_to_the_model(monkeypatch, tmp_path):
    _scenario(monkeypatch, tmp_path)
    client.delete("/agent/memory/long_term")

    context = client.get("/agent/context").json()["context"]

    assert "LONG-TERM MEMORY" not in context
    assert "WORKING MEMORY" in context


def test_clearing_an_unknown_layer_is_reported(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    assert client.delete("/agent/memory/everything").status_code == 422


def test_long_term_memory_survives_a_restart(monkeypatch, tmp_path):
    """Both files are re-opened from scratch, the way they would be after
    the process restarts."""
    _scenario(monkeypatch, tmp_path)

    monkeypatch.setattr(
        agent_module.agent, "_history", AgentHistoryStorage(file_path=str(tmp_path / "history.json"))
    )
    monkeypatch.setattr(
        agent_module.agent,
        "_long_term",
        AgentLongTermMemoryStorage(file_path=str(tmp_path / "long_term.json")),
    )
    memory = _memory()

    assert memory["long_term"]["profile"] == {"favorite_word": "学習"}
    assert memory["working"]["constraints"] == ["уровень N4", "минималистичный интерфейс"]
    assert len(memory["short_term"]["messages"]) == 10


def test_clearing_the_dialogue_ends_the_task_but_not_the_learner(monkeypatch, tmp_path):
    """DELETE /agent/history ends the conversation, so it takes the two
    layers scoped to one with it - and deliberately not long-term memory."""
    _scenario(monkeypatch, tmp_path)

    client.delete("/agent/history")
    memory = _memory()

    assert memory["short_term"]["messages"] == []
    assert memory["working"]["constraints"] == []
    assert memory["long_term"]["profile"] == {"favorite_word": "学習"}


def test_a_layer_can_be_written_directly(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    client.put("/agent/memory/long_term", json={"preferences": ["краткие ответы"]})
    memory = client.put("/agent/memory/working", json={"goals": ["составить пример"]}).json()

    assert memory["long_term"]["preferences"] == ["краткие ответы"]
    assert memory["working"]["goals"] == ["составить пример"]


def test_updating_one_field_of_a_layer_keeps_the_rest(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    client.put("/agent/memory/working", json={"goals": ["составить пример"], "constraints": ["N4"]})

    memory = client.put("/agent/memory/working", json={"decisions": ["взяли 学習"]}).json()

    assert memory["working"]["goals"] == ["составить пример"]
    assert memory["working"]["constraints"] == ["N4"]
    assert memory["working"]["decisions"] == ["взяли 学習"]


def test_the_conversation_can_be_written_directly_as_short_term_memory(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    memory = client.put(
        "/agent/memory/short_term",
        json={"messages": [{"role": "user", "content": "начнём заново"}]},
    ).json()

    assert memory["short_term"]["messages"] == [{"role": "user", "content": "начнём заново"}]
    assert client.get("/agent/history").json()["messages"] == [
        {"role": "user", "content": "начнём заново"}
    ]


def test_the_other_strategies_never_write_to_the_memory_layers(monkeypatch, tmp_path):
    """Day 11 changes what the layered strategy does, and nothing else."""
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ")
    _mock_count_tokens(monkeypatch, 90)
    _mock_memory_router(monkeypatch, '{"long_term": {"profile": {"favorite_word": "学習"}}}')

    for strategy in ("full", "sliding_window", "branching"):
        client.post(
            "/agent/chat",
            json={"message": "Запомни надолго: моё любимое слово — 学習.", "strategy": strategy},
        )

    memory = _memory()

    assert memory["long_term"]["profile"] == {}
    assert memory["working"]["goals"] == []


def test_memory_routing_is_recorded_as_its_own_token_cost(monkeypatch, tmp_path):
    """The extra Gemini call the layer routing costs is logged separately,
    not folded into the answer's own numbers."""
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ", input_tokens=40, output_tokens=8)
    _mock_count_tokens(monkeypatch, 25)
    _mock_memory_router(monkeypatch, '{"working": {"goals": ["пример"]}}')

    _say_with_memory("Для текущей задачи запомни: нужен пример.")
    entry = client.get("/agent/usage").json()["entries"][-1]

    assert entry["strategy"] == "layered_memory"
    assert entry["memory_tokens"] == 175
    assert entry["total_tokens"] == 48


def test_a_failed_routing_keeps_the_memory_and_still_answers(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ агента")
    _mock_count_tokens(monkeypatch, 90)
    client.put("/agent/memory/working", json={"goals": ["составить пример"]})
    _mock_memory_router(monkeypatch, "конечно, запомню!")

    response = _say_with_memory("Для текущей задачи запомни: уровень N4.")

    assert response.json()["response"] == "ответ агента"
    assert _memory()["working"]["goals"] == ["составить пример"]


# --- user profile (Day 12) -------------------------------------------------

PROFILE_A = {
    "japanese_level": "N4",
    "explanation_style": "simple",
    "answer_format": "short",
    "translation_language": "Russian",
    "preferred_language": "Russian",
}
PROFILE_B = {
    "japanese_level": "N2",
    "explanation_style": "detailed",
    "answer_format": "detailed",
    "translation_language": "English",
    "preferred_language": "English",
}


def _capture_prompts(monkeypatch, response_text="ответ агента"):
    prompts = []

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        prompts.append(prompt)
        return GeneratedText(text=response_text, input_tokens=40, output_tokens=8)

    monkeypatch.setattr(agent_module, "generate_text_with_usage", fake_generate_text_with_usage)
    return prompts


def _set_profile(profile: dict):
    response = client.put("/agent/profile", json=profile)
    assert response.status_code == 200
    return response.json()


def test_the_profile_is_applied_without_being_asked_for(monkeypatch, tmp_path):
    """The learner says nothing about level or format - the request still
    carries both."""
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    _set_profile(PROFILE_A)

    client.post("/agent/chat", json={"message": "Объясни слово 学習."})

    assert "N4" not in "Объясни слово 学習."
    assert "USER PROFILE" in prompts[0]
    assert "N4" in prompts[0]
    assert "Answer format: short" in prompts[0]


def test_the_same_message_reaches_gemini_differently_for_two_profiles(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)

    _set_profile(PROFILE_A)
    client.post("/agent/chat", json={"message": "Объясни слово 学習.", "strategy": "full"})
    client.delete("/agent/history")
    _set_profile(dict(PROFILE_B, preferences=[]))
    client.post("/agent/chat", json={"message": "Объясни слово 学習.", "strategy": "full"})

    first, second = prompts
    assert first != second
    assert "N4" in first and "N2" not in first
    assert "N2" in second and "N4" not in second
    assert "Translate Japanese into: Russian" in first
    assert "Translate Japanese into: English" in second
    # The message itself is identical - the whole difference is the profile.
    assert first.replace(_profile_block(first), "") == second.replace(_profile_block(second), "")


def _profile_block(prompt: str) -> str:
    """The USER PROFILE section of a prompt, up to the blank line after it."""
    start = prompt.index("USER PROFILE")
    return prompt[start : prompt.index("\n\n", start) + 2]


def test_the_profile_applies_under_every_strategy(monkeypatch, tmp_path):
    """It is not a strategy and does not belong to one - it is how the
    learner wants to be answered, whatever the context strategy is."""
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    _mock_memory_router(monkeypatch, "{}")
    _set_profile(PROFILE_A)

    for strategy in ("full", "sliding_window", "sticky_facts", "branching", "layered_memory"):
        client.post("/agent/chat", json={"message": "Объясни слово 学習.", "strategy": strategy})

    assert len(prompts) == 5
    assert all("USER PROFILE" in prompt and "N4" in prompt for prompt in prompts)


def test_without_a_profile_nothing_is_added_to_the_prompt(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)

    client.post("/agent/chat", json={"message": "Объясни слово 学習."})

    assert "USER PROFILE" not in prompts[0]


def test_the_profile_comes_after_the_context_and_before_the_message(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    _set_profile(PROFILE_A)
    client.post("/agent/chat", json={"message": "Первый вопрос.", "strategy": "full"})

    client.post("/agent/chat", json={"message": "Второй вопрос.", "strategy": "full"})

    prompt = prompts[-1]
    assert prompt.index("Conversation so far:") < prompt.index("USER PROFILE")
    assert prompt.index("USER PROFILE") < prompt.index("Learner's new request: Второй вопрос.")


def test_the_profile_is_not_counted_as_conversation_history(monkeypatch, tmp_path):
    """History is what the conversation produced; the profile is a setting,
    so turning one on must not inflate the history token count."""
    _isolate_agent(monkeypatch, tmp_path)
    _capture_prompts(monkeypatch)

    async def fake_count_tokens(text, model=None):
        return len(text)

    monkeypatch.setattr(agent_module, "count_tokens", fake_count_tokens)
    client.post("/agent/chat", json={"message": "Первый вопрос.", "strategy": "full"})
    without_profile = client.post(
        "/agent/chat", json={"message": "Второй вопрос.", "strategy": "full"}
    ).json()["usage"]["history_tokens"]

    _set_profile(PROFILE_A)
    with_profile = client.post(
        "/agent/chat", json={"message": "Второй вопрос.", "strategy": "full"}
    ).json()["usage"]["history_tokens"]

    # The conversation grew by one exchange between the two measurements, so
    # history grew - but by the exchange, not by the profile.
    assert with_profile > without_profile
    assert "N4" not in client.get("/agent/context").json()["context"]


def test_the_profile_never_becomes_part_of_the_conversation(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ агента")
    _mock_count_tokens(monkeypatch, 90)
    _mock_memory_router(monkeypatch, "{}")
    _set_profile(PROFILE_A)

    client.post("/agent/chat", json={"message": "Объясни слово 学習.", "strategy": "layered_memory"})
    history = client.get("/agent/history").json()["messages"]
    memory = client.get("/agent/memory").json()

    assert [entry["content"] for entry in history] == ["Объясни слово 学習.", "ответ агента"]
    assert "N4" not in str(memory["working"])
    assert "N4" not in str(memory["long_term"])


def test_clearing_memory_and_history_leaves_the_profile_alone(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _set_profile(PROFILE_A)

    client.delete("/agent/history")
    for layer in ("short_term", "working", "long_term"):
        client.delete(f"/agent/memory/{layer}")

    assert client.get("/agent/profile").json()["japanese_level"] == "N4"


def test_clearing_the_profile_leaves_the_memory_layers_alone(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _set_profile(PROFILE_A)
    client.put("/agent/memory/working", json={"constraints": ["уровень N4"]})
    client.put("/agent/memory/long_term", json={"profile": {"favorite_word": "学習"}})

    cleared = client.delete("/agent/profile").json()
    memory = client.get("/agent/memory").json()

    assert cleared == {
        "preferred_language": "",
        "japanese_level": "",
        "explanation_style": "",
        "answer_format": "",
        "translation_language": "",
        "preferences": [],
    }
    assert memory["working"]["constraints"] == ["уровень N4"]
    assert memory["long_term"]["profile"] == {"favorite_word": "学習"}


def test_updating_one_setting_keeps_the_rest(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _set_profile(PROFILE_A)

    updated = client.put("/agent/profile", json={"japanese_level": "N2"}).json()

    assert updated["japanese_level"] == "N2"
    assert updated["answer_format"] == "short"
    assert updated["translation_language"] == "Russian"


def test_extra_preferences_are_kept_and_applied(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    _set_profile(dict(PROFILE_A, preferences=["без ромадзи", "примеры из жизни"]))

    client.post("/agent/chat", json={"message": "Объясни слово 学習."})

    assert "- Also: без ромадзи" in prompts[0]
    assert "- Also: примеры из жизни" in prompts[0]


def test_the_profile_survives_a_restart(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    _set_profile(PROFILE_A)

    monkeypatch.setattr(
        agent_module.agent,
        "_profile",
        AgentUserProfileStorage(file_path=str(tmp_path / "profile.json")),
    )
    body = client.get("/agent/profile").json()
    client.post("/agent/chat", json={"message": "Объясни слово 学習."})

    assert body["japanese_level"] == "N4"
    assert body["translation_language"] == "Russian"
    assert "N4" in prompts[0]


# --- task state machine (Day 13) -------------------------------------------


def _mock_task_tracker(monkeypatch, *answers: str):
    """Turn tracking on for this test and answer the tracker's Gemini call
    with each JSON in turn, repeating the last one."""
    monkeypatch.setattr(agent_module.agent, "_task_tracking_enabled", True)
    queue = list(answers) or ["{}"]

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        return GeneratedText(text=answer, input_tokens=160, output_tokens=20)

    monkeypatch.setattr(task_tracker_module, "generate_text_with_usage", fake_generate_text_with_usage)


def _task() -> dict:
    return client.get("/agent/task").json()


def test_a_task_starts_in_planning_and_nowhere_else(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ агента")
    _mock_count_tokens(monkeypatch, 90)
    _mock_task_tracker(
        monkeypatch,
        '{"task_stage": "planning", "current_step": "составляем план",'
        ' "expected_action": "согласовать план"}',
    )

    client.post("/agent/chat", json={"message": "Давай составим план по грамматике N4."})

    assert _task() == {
        "task_stage": "planning",
        "current_step": "составляем план",
        "expected_action": "согласовать план",
        "allowed_next": ["execution"],
    }


def test_an_ordinary_question_starts_no_task(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ агента")
    _mock_count_tokens(monkeypatch, 90)
    _mock_task_tracker(monkeypatch, '{"task_stage": "idle"}')

    client.post("/agent/chat", json={"message": "Что значит 学習?"})

    assert _task()["task_stage"] == "idle"
    assert _task()["allowed_next"] == ["planning"]


def test_a_task_cannot_skip_a_stage_however_the_model_answers(monkeypatch, tmp_path):
    """The tracker proposes; the state machine decides. A jump from planning
    to done leaves the task exactly where it was."""
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ агента")
    _mock_count_tokens(monkeypatch, 90)
    _mock_task_tracker(
        monkeypatch,
        '{"task_stage": "planning", "current_step": "составляем план", "expected_action": "начать"}',
        '{"task_stage": "done", "current_step": "всё готово", "expected_action": "поздравить"}',
    )
    client.post("/agent/chat", json={"message": "Давай составим план."})

    client.post("/agent/chat", json={"message": "Всё, заканчиваем."})

    assert _task()["task_stage"] == "planning"
    assert _task()["current_step"] == "составляем план"


def test_the_task_state_is_sent_with_every_request(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    _mock_task_tracker(
        monkeypatch,
        '{"task_stage": "planning", "current_step": "составляем план", "expected_action": "начать"}',
    )
    client.post("/agent/chat", json={"message": "Давай составим план."})

    client.post("/agent/chat", json={"message": "Продолжим."})

    prompt = prompts[-1]
    assert "TASK STATE" in prompt
    assert "- Stage: planning" in prompt
    assert "- Current step: составляем план" in prompt


def test_the_task_state_comes_last_before_the_message(monkeypatch, tmp_path):
    """"Carry on from this step" is the last thing the model reads before
    the question."""
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    _set_profile(PROFILE_A)
    _mock_task_tracker(
        monkeypatch, '{"task_stage": "planning", "current_step": "составляем план"}'
    )
    client.post("/agent/chat", json={"message": "Давай составим план.", "strategy": "full"})

    client.post("/agent/chat", json={"message": "Продолжим.", "strategy": "full"})

    prompt = prompts[-1]
    assert prompt.index("Conversation so far:") < prompt.index("USER PROFILE")
    assert prompt.index("USER PROFILE") < prompt.index("TASK STATE")
    assert prompt.index("TASK STATE") < prompt.index("Learner's new request: Продолжим.")


def test_an_idle_task_adds_nothing_to_the_prompt(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    _mock_task_tracker(monkeypatch, '{"task_stage": "idle"}')

    client.post("/agent/chat", json={"message": "Что значит 学習?"})

    assert "TASK STATE" not in prompts[0]


def test_tracking_is_recorded_as_its_own_token_cost(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ", input_tokens=40, output_tokens=8)
    _mock_count_tokens(monkeypatch, 25)
    _mock_task_tracker(monkeypatch, '{"task_stage": "planning", "current_step": "план"}')

    client.post("/agent/chat", json={"message": "Давай составим план."})
    entry = client.get("/agent/usage").json()["entries"][-1]

    assert entry["task_tokens"] == 180
    assert entry["total_tokens"] == 48


def test_a_failed_tracking_keeps_the_task_where_it_was_and_still_answers(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ агента")
    _mock_count_tokens(monkeypatch, 90)
    _mock_task_tracker(monkeypatch, '{"task_stage": "planning", "current_step": "составляем план"}')
    client.post("/agent/chat", json={"message": "Давай составим план."})
    _mock_task_tracker(monkeypatch, "конечно, продолжаем!")

    response = client.post("/agent/chat", json={"message": "Продолжим."})

    assert response.json()["response"] == "ответ агента"
    assert _task()["current_step"] == "составляем план"


def test_clearing_the_task_leaves_the_conversation_and_everything_else_alone(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ агента")
    _mock_count_tokens(monkeypatch, 90)
    _set_profile(PROFILE_A)
    client.put("/agent/memory/working", json={"constraints": ["уровень N4"]})
    _mock_task_tracker(monkeypatch, '{"task_stage": "planning", "current_step": "составляем план"}')
    client.post("/agent/chat", json={"message": "Давай составим план."})

    cleared = client.delete("/agent/task").json()

    assert cleared == {
        "task_stage": "idle",
        "current_step": "",
        "expected_action": "",
        "allowed_next": ["planning"],
    }
    assert len(client.get("/agent/history").json()["messages"]) == 2
    assert client.get("/agent/memory").json()["working"]["constraints"] == ["уровень N4"]
    assert client.get("/agent/profile").json()["japanese_level"] == "N4"


def test_clearing_the_conversation_ends_the_task_with_it(monkeypatch, tmp_path):
    """The task belongs to the conversation, like the working memory."""
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ агента")
    _mock_count_tokens(monkeypatch, 90)
    _mock_task_tracker(monkeypatch, '{"task_stage": "planning", "current_step": "составляем план"}')
    client.post("/agent/chat", json={"message": "Давай составим план."})

    client.delete("/agent/history")

    assert _task()["task_stage"] == "idle"


def test_the_task_never_becomes_part_of_the_conversation_or_the_memory(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ агента")
    _mock_count_tokens(monkeypatch, 90)
    _mock_memory_router(monkeypatch, "{}")
    _mock_task_tracker(
        monkeypatch, '{"task_stage": "planning", "current_step": "составляем план N4"}'
    )

    client.post("/agent/chat", json={"message": "Давай составим план.", "strategy": "layered_memory"})
    history = client.get("/agent/history").json()["messages"]
    memory = client.get("/agent/memory").json()

    assert [entry["content"] for entry in history] == ["Давай составим план.", "ответ агента"]
    assert "составляем план N4" not in str(memory)


def test_the_eight_step_scenario(monkeypatch, tmp_path):
    """Start a task, move it to execution, pause, restart the app, carry on
    from the step it was on, validate, finish."""
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)

    # 1-2. start the task, then move planning -> execution
    _mock_task_tracker(
        monkeypatch,
        '{"task_stage": "planning", "current_step": "план из 4 шагов",'
        ' "expected_action": "согласовать план"}',
        '{"task_stage": "execution", "current_step": "шаг 2 из 4 — предложения с 〜ながら",'
        ' "expected_action": "показать пять предложений"}',
    )
    client.post("/agent/chat", json={"message": "Давай составим план по грамматике N4."})
    assert _task()["task_stage"] == "planning"
    client.post("/agent/chat", json={"message": "План подходит, приступаем."})
    assert _task()["task_stage"] == "execution"

    # 3-4. pause and restart: a brand new storage instance over the same file
    paused = _task()
    monkeypatch.setattr(
        agent_module.agent, "_history", AgentHistoryStorage(file_path=str(tmp_path / "history.json"))
    )

    # 5. execution came back, with the step it was on
    assert _task() == paused
    assert _task()["task_stage"] == "execution"
    assert _task()["current_step"] == "шаг 2 из 4 — предложения с 〜ながら"

    # 6. carry on: the request says where the task is, not what it was about
    _mock_task_tracker(
        monkeypatch,
        '{"task_stage": "execution", "current_step": "шаг 3 из 4 — разбор ошибок",'
        ' "expected_action": "исправить предложения"}',
    )
    client.post("/agent/chat", json={"message": "Продолжаем."})
    resumed = prompts[-1]
    assert "- Stage: execution" in resumed
    assert "шаг 2 из 4 — предложения с 〜ながら" in resumed
    assert "do not repeat explanations already given" in resumed

    # 7. validation
    _mock_task_tracker(
        monkeypatch,
        '{"task_stage": "validation", "current_step": "проверяем предложения",'
        ' "expected_action": "подтвердить, что всё верно"}',
    )
    client.post("/agent/chat", json={"message": "Проверь, что получилось."})
    assert _task()["task_stage"] == "validation"
    assert _task()["allowed_next"] == ["done"]

    # 8. done
    _mock_task_tracker(
        monkeypatch, '{"task_stage": "done", "current_step": "задача завершена", "expected_action": ""}'
    )
    client.post("/agent/chat", json={"message": "Отлично, всё верно. Закончили."})
    assert _task()["task_stage"] == "done"
    assert _task()["allowed_next"] == []


# --- invariants (Day 14) ---------------------------------------------------


def _invariants() -> list[dict]:
    return client.get("/agent/invariants").json()["invariants"]


def _ids() -> list[str]:
    return [item["id"] for item in _invariants()]


def test_the_project_rules_are_there_from_the_start(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    rules = [item["rule"] for item in _invariants()]

    assert "Backend: FastAPI" in rules
    assert "LLM: Gemini" in rules
    assert "Storage: JSON files" in rules
    assert "Android: Kotlin" in rules
    assert "Android layering: ViewModel -> Repository -> API" in rules
    assert "SQLite is not used" in rules
    assert "The LLM API is called only from the backend, never from Android" in rules
    assert any("No new LLM integration" in rule for rule in rules)


def test_every_rule_says_which_of_the_four_categories_it_is(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    categories = {item["category"] for item in _invariants()}

    assert categories <= {"architecture", "technology_stack", "technical_decisions", "business_rules"}
    assert {"architecture", "technology_stack", "technical_decisions"} <= categories


def test_the_rules_are_sent_with_every_request_under_every_strategy(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    _mock_memory_router(monkeypatch, "{}")

    for strategy in ("full", "sliding_window", "sticky_facts", "branching", "layered_memory"):
        client.post("/agent/chat", json={"message": "Предложи улучшение архитектуры.", "strategy": strategy})

    assert len(prompts) == 5
    assert all("INVARIANTS" in prompt and "SQLite is not used" in prompt for prompt in prompts)


def test_the_rules_come_before_the_profile_and_the_task_state(monkeypatch, tmp_path):
    """Hardest first: the rules bind whatever is asked, the profile is a
    default the message can override, the task state says where to carry on."""
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    _set_profile(PROFILE_A)
    _mock_task_tracker(monkeypatch, '{"task_stage": "planning", "current_step": "план"}')
    client.post("/agent/chat", json={"message": "Начнём.", "strategy": "full"})

    client.post("/agent/chat", json={"message": "Продолжим.", "strategy": "full"})

    prompt = prompts[-1]
    assert prompt.index("Conversation so far:") < prompt.index("INVARIANTS")
    assert prompt.index("INVARIANTS") < prompt.index("USER PROFILE")
    assert prompt.index("USER PROFILE") < prompt.index("TASK STATE")
    assert prompt.index("TASK STATE") < prompt.index("Learner's new request: Продолжим.")


def test_the_request_carries_what_to_do_about_a_conflict(monkeypatch, tmp_path):
    """The rules alone would only produce a refusal - the protocol is what
    turns it into a named rule, a reason and an alternative."""
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)

    client.post(
        "/agent/chat",
        json={"message": "Давай перенесём вызов Gemini API прямо в Android и возьмём SQLite вместо JSON."},
    )

    prompt = prompts[0]
    assert "The LLM API is called only from the backend, never from Android" in prompt
    assert "SQLite is not used" in prompt
    assert "name the rule it conflicts with" in prompt
    assert "offer an alternative that respects the rules" in prompt


def test_a_rule_can_be_added_changed_and_removed(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    added = client.post(
        "/agent/invariants",
        json={"category": "business_rules", "rule": "Материалы только для уровней JLPT"},
    ).json()["invariants"]
    new_id = added[-1]["id"]
    changed = client.put(
        f"/agent/invariants/{new_id}",
        json={"category": "business_rules", "rule": "Материалы только для уровней N5-N1"},
    ).json()["invariants"]
    removed = client.delete(f"/agent/invariants/{new_id}").json()["invariants"]

    assert new_id == "business_rules-1"
    assert changed[-1]["rule"] == "Материалы только для уровней N5-N1"
    assert new_id not in [item["id"] for item in removed]
    assert len(removed) == len(added) - 1


def test_changing_a_rule_keeps_its_place_in_the_list(monkeypatch, tmp_path):
    """The order the rules are read in should not shuffle because one of
    them was reworded."""
    _isolate_agent(monkeypatch, tmp_path)
    before = _ids()

    client.put(
        "/agent/invariants/stack-storage",
        json={"category": "technology_stack", "rule": "Storage: JSON files only"},
    )

    assert _ids() == before
    assert [item["rule"] for item in _invariants() if item["id"] == "stack-storage"] == [
        "Storage: JSON files only"
    ]


def test_a_rule_added_applies_to_the_very_next_message(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    client.post(
        "/agent/invariants",
        json={"category": "business_rules", "rule": "Никаких платных сервисов"},
    )

    client.post("/agent/chat", json={"message": "Что посоветуешь?"})

    assert "Business rules:" in prompts[0]
    assert "Никаких платных сервисов" in prompts[0]


def test_a_rule_removed_stops_being_sent(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    client.delete("/agent/invariants/decision-no-sqlite")

    client.post("/agent/chat", json={"message": "Что посоветуешь?"})

    assert "SQLite is not used" not in prompts[0]
    assert "Backend: FastAPI" in prompts[0]


def test_removing_a_rule_that_is_not_there_is_reported(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    assert client.delete("/agent/invariants/nope").status_code == 404


def test_a_rule_with_an_unknown_category_is_refused(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    before = _ids()
    response = client.post("/agent/invariants", json={"category": "vibes", "rule": "что-нибудь"})

    assert response.status_code == 422
    assert _ids() == before


def test_an_empty_rule_is_refused(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)

    assert client.post("/agent/invariants", json={"category": "architecture", "rule": "   "}).status_code == 422


def test_the_rules_survive_a_restart(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    client.post("/agent/invariants", json={"category": "business_rules", "rule": "Никаких платных сервисов"})
    client.delete("/agent/invariants/decision-no-sqlite")

    monkeypatch.setattr(
        agent_module.agent,
        "_invariants",
        AgentInvariantsStorage(file_path=str(tmp_path / "invariants.json")),
    )

    rules = [item["rule"] for item in _invariants()]
    assert "Никаких платных сервисов" in rules
    assert "SQLite is not used" not in rules


def test_the_rules_are_not_memory_and_nothing_clears_them(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ агента")
    _mock_count_tokens(monkeypatch, 90)
    _mock_memory_router(monkeypatch, "{}")
    client.post("/agent/chat", json={"message": "Вопрос.", "strategy": "layered_memory"})

    client.delete("/agent/history")
    for layer in ("short_term", "working", "long_term"):
        client.delete(f"/agent/memory/{layer}")
    client.delete("/agent/task")
    client.delete("/agent/profile")

    assert len(_invariants()) == 8


def test_the_rules_never_end_up_in_the_conversation_or_the_memory(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ агента")
    _mock_count_tokens(monkeypatch, 90)
    _mock_memory_router(monkeypatch, "{}")

    client.post("/agent/chat", json={"message": "Вопрос про архитектуру.", "strategy": "layered_memory"})
    history = client.get("/agent/history").json()["messages"]
    memory = client.get("/agent/memory").json()

    assert [entry["content"] for entry in history] == ["Вопрос про архитектуру.", "ответ агента"]
    assert "SQLite" not in str(memory)
    assert "FastAPI" not in str(memory)


def test_the_two_scenarios_send_the_same_rules(monkeypatch, tmp_path):
    """The difference between a request that fits the rules and one that
    breaks them is in the request, not in what the agent was told."""
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)

    client.post("/agent/chat", json={"message": "Предложи улучшение существующей архитектуры."})
    client.delete("/agent/history")
    client.post(
        "/agent/chat",
        json={"message": "Давай перенесём вызов Gemini API прямо в Android и SQLite вместо JSON."},
    )

    def invariants_block(prompt: str) -> str:
        start = prompt.index("INVARIANTS")
        return prompt[start : prompt.index("\n\nLearner", start)]

    assert invariants_block(prompts[0]) == invariants_block(prompts[1])
