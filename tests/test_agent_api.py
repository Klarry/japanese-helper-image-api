import asyncio
import sys

import pytest

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
from app.services import agent_tool_planner as tool_planner_module
from app.services import japanese_learning_agent as agent_module
from app.services.agent_history_storage import AgentHistoryStorage
from app.services.agent_invariants import AgentInvariantsStorage
from app.services.agent_memory import AgentLongTermMemoryStorage
from app.services.agent_tools import McpToolbox
from app.services.digest import DigestStore, DigestTaskStorage, collect_once
from app.services.agent_user_profile import AgentUserProfileStorage
from app.services.agent_usage_log import AgentUsageLog
from app.services.gemini_service import GeneratedText
from app.services.japanese_learning_agent import JapaneseLearningAgent
from app.services.mcp_client import jlpt_vocab_server_parameters
from mcp import StdioServerParameters
from tests.jlpt_api_standin import jlpt_api

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
    # Likewise the MCP tools (Day 17): a test that is about them turns them on.
    monkeypatch.setattr(agent_module.agent, "_tools_enabled", False)
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
        "plan": "",
        "validation_passed": False,
        "validation_note": "",
        "next_requirement": "the plan has not been approved yet",
        "blocked": None,
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
        "plan": "",
        "validation_passed": False,
        "validation_note": "",
        "next_requirement": "",
        "blocked": None,
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
        ' "expected_action": "показать пять предложений", "plan": "план из 4 шагов"}',
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
        monkeypatch,
        '{"task_stage": "done", "current_step": "задача завершена", "expected_action": "",'
        ' "validation_passed": true}',
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


# --- controlled transitions (Day 15) ---------------------------------------


def _transition(stage: str, current_step: str = "", expected_action: str = ""):
    return client.post(
        "/agent/task/transition",
        json={"task_stage": stage, "current_step": current_step, "expected_action": expected_action},
    )


def _start_planning(plan: str | None = None) -> dict:
    """Bring a task to planning through the API, optionally with its plan
    approved - the starting point most of these tests need."""
    _transition("planning", "составляем план", "согласовать план")

    if plan is not None:
        client.post("/agent/task/plan", json={"plan": plan})

    return _task()


def test_a_task_is_driven_stage_by_stage_through_the_api(monkeypatch, tmp_path):
    """1. The normal path: planning -> execution -> validation -> done, with
    each stage's own work done before the one after it is entered."""
    _isolate_agent(monkeypatch, tmp_path)

    assert _transition("planning", "составляем план").json()["task_stage"] == "planning"
    client.post("/agent/task/plan", json={"plan": "план из 4 шагов"})
    assert _transition("execution", "шаг 1 из 4").json()["task_stage"] == "execution"
    assert _transition("validation", "проверяем предложения").json()["task_stage"] == "validation"
    client.post("/agent/task/validation", json={"passed": True, "notes": "всё верно"})
    finished = _transition("done", "задача завершена").json()

    assert finished["task_stage"] == "done"
    assert finished["allowed_next"] == []
    assert finished["blocked"] is None


def test_planning_cannot_jump_to_done(monkeypatch, tmp_path):
    """2. Refused, with the four things a refusal owes the caller - and the
    task exactly where it was."""
    _isolate_agent(monkeypatch, tmp_path)
    _start_planning(plan="план из 4 шагов")

    response = _transition("done", "всё готово")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["current_stage"] == "planning"
    assert detail["requested_stage"] == "done"
    assert detail["required_next"] == ["execution"]
    assert detail["unmet_condition"] == "the task has not been through execution and validation yet"
    assert _task()["task_stage"] == "planning"
    assert _task()["current_step"] == "составляем план"


def test_planning_cannot_jump_to_validation(monkeypatch, tmp_path):
    """3."""
    _isolate_agent(monkeypatch, tmp_path)
    _start_planning(plan="план из 4 шагов")

    response = _transition("validation", "проверяем")

    assert response.status_code == 409
    assert response.json()["detail"]["unmet_condition"] == "the task has not been through execution yet"
    assert _task()["task_stage"] == "planning"


def test_execution_cannot_jump_to_done(monkeypatch, tmp_path):
    """4."""
    _isolate_agent(monkeypatch, tmp_path)
    _start_planning(plan="план из 4 шагов")
    _transition("execution", "шаг 1 из 4")

    response = _transition("done", "всё готово")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["required_next"] == ["validation"]
    assert detail["unmet_condition"] == "the task has not been through validation yet"
    assert _task()["task_stage"] == "execution"


def test_execution_cannot_start_without_an_approved_plan(monkeypatch, tmp_path):
    """The edge is legal and the move is still refused: planning has not
    produced the thing execution needs."""
    _isolate_agent(monkeypatch, tmp_path)
    _start_planning()

    response = _transition("execution", "шаг 1 из 4")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["current_stage"] == "planning"
    assert detail["required_next"] == ["execution"]
    assert detail["unmet_condition"] == "the plan has not been approved yet"
    assert _task()["task_stage"] == "planning"
    assert _task()["next_requirement"] == "the plan has not been approved yet"


def test_done_cannot_be_reached_without_a_validation_that_passed(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _start_planning(plan="план из 4 шагов")
    _transition("execution", "шаг 1 из 4")
    _transition("validation", "проверяем")
    client.post("/agent/task/validation", json={"passed": False, "notes": "две ошибки в 〜ながら"})

    response = _transition("done", "готово")

    assert response.status_code == 409
    assert response.json()["detail"]["unmet_condition"] == "validation has not passed yet"
    assert _task()["task_stage"] == "validation"
    assert _task()["validation_note"] == "две ошибки в 〜ながら"


def test_the_refusal_stays_on_the_task_after_the_error_response(monkeypatch, tmp_path):
    """The 409 is gone as soon as it is read; the reason the task did not
    move is still worth showing on the screen that asked."""
    _isolate_agent(monkeypatch, tmp_path)
    _start_planning(plan="план из 4 шагов")

    _transition("done", "всё готово")

    blocked = _task()["blocked"]
    assert blocked["current_stage"] == "planning"
    assert blocked["requested_stage"] == "done"
    assert blocked["required_next"] == ["execution"]
    assert "cannot move to 'done'" in blocked["message"]


def test_a_refused_move_is_explained_by_the_agent_on_the_next_message(monkeypatch, tmp_path):
    """Nothing is silently ignored: the next request carries the refusal, so
    the answer can say which stage the task is in and what is missing."""
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    _start_planning(plan="план из 4 шагов")
    _transition("done", "всё готово")

    client.post("/agent/chat", json={"message": "Ну что, закончили?"})

    assert "- Stage: planning" in prompts[-1]
    assert "Refused move:" in prompts[-1]
    assert "the task has not been through execution and validation yet" in prompts[-1]


def test_the_agents_own_illegal_proposal_is_refused_the_same_way(monkeypatch, tmp_path):
    """The tracker goes through the same check as a client: a model that
    decides the task is finished cannot finish it."""
    _isolate_agent(monkeypatch, tmp_path)
    _mock_generate(monkeypatch, "ответ агента")
    _mock_count_tokens(monkeypatch, 90)
    _start_planning(plan="план из 4 шагов")
    _mock_task_tracker(
        monkeypatch,
        '{"task_stage": "done", "current_step": "задача завершена", "expected_action": ""}',
    )

    client.post("/agent/chat", json={"message": "Всё, мы закончили."})

    assert _task()["task_stage"] == "planning"
    assert _task()["blocked"]["requested_stage"] == "done"


def test_a_plan_can_only_be_approved_while_the_task_is_planning(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _start_planning(plan="план из 4 шагов")
    _transition("execution", "шаг 1 из 4")

    response = client.post("/agent/task/plan", json={"plan": "другой план"})

    assert response.status_code == 409
    assert response.json()["detail"]["current_stage"] == "execution"
    assert _task()["plan"] == "план из 4 шагов"


def test_validation_can_only_be_recorded_while_the_task_is_validating(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _start_planning(plan="план из 4 шагов")
    _transition("execution", "шаг 1 из 4")

    response = client.post("/agent/task/validation", json={"passed": True})

    assert response.status_code == 409
    assert response.json()["detail"]["unmet_condition"] == "the task is not in 'validation'"
    assert _task()["validation_passed"] is False


def test_a_stage_that_does_not_exist_never_reaches_the_machine(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _start_planning(plan="план из 4 шагов")

    assert _transition("halfway").status_code == 422
    assert client.post("/agent/task/plan", json={"plan": "   "}).status_code == 422
    assert _task()["task_stage"] == "planning"


def test_clearing_the_task_forgets_the_plan_the_validation_and_the_refusal(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _start_planning(plan="план из 4 шагов")
    _transition("done", "всё готово")

    cleared = client.delete("/agent/task").json()

    assert cleared["task_stage"] == "idle"
    assert cleared["plan"] == ""
    assert cleared["blocked"] is None


def test_the_eight_step_scenario_of_day_fifteen(monkeypatch, tmp_path):
    """The whole assignment in one run: the normal path, the three refused
    jumps, a pause in execution, a restart, carrying on from execution and
    finishing only after validation passes."""
    _isolate_agent(monkeypatch, tmp_path)
    prompts = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)

    # 1. planning
    _transition("planning", "план из 4 шагов", "согласовать план")
    assert _task()["allowed_next"] == ["execution"]
    assert _task()["next_requirement"] == "the plan has not been approved yet"

    # 2. planning -> done: refused
    assert _transition("done").status_code == 409
    # 3. planning -> validation: refused
    assert _transition("validation").status_code == 409
    assert _task()["task_stage"] == "planning"

    # 1 (continued). the plan is approved, and only then execution starts
    client.post("/agent/task/plan", json={"plan": "4 шага: разбор, примеры, проверка, вывод"})
    assert _transition("execution", "шаг 2 из 4 — предложения с 〜ながら").status_code == 200

    # 4. execution -> done: refused
    refused = _transition("done", "всё готово")
    assert refused.status_code == 409
    assert refused.json()["detail"]["required_next"] == ["validation"]

    # 5. pause in execution, 6. restart: a new storage instance over the same file
    paused = _task()
    monkeypatch.setattr(
        agent_module.agent, "_history", AgentHistoryStorage(file_path=str(tmp_path / "history.json"))
    )

    # 7. carry on from execution, with the step and the plan it was on
    assert _task() == paused
    assert _task()["task_stage"] == "execution"
    assert _task()["current_step"] == "шаг 2 из 4 — предложения с 〜ながら"
    assert _task()["plan"] == "4 шага: разбор, примеры, проверка, вывод"

    client.post("/agent/chat", json={"message": "Продолжаем."})
    resumed = prompts[-1]
    assert "- Stage: execution" in resumed
    assert "шаг 2 из 4 — предложения с 〜ながら" in resumed
    assert "- Approved plan: 4 шага: разбор, примеры, проверка, вывод" in resumed

    # 8. validation, and done only once it passed
    _transition("validation", "проверяем предложения")
    assert _transition("done").status_code == 409
    client.post("/agent/task/validation", json={"passed": True, "notes": "все пять верны"})
    finished = _transition("done", "задача завершена").json()

    assert finished["task_stage"] == "done"
    assert finished["allowed_next"] == []
    assert finished["validation_passed"] is True
    assert _transition("planning").status_code == 409


# --- MCP tools (Day 17) ----------------------------------------------------
#
# The whole path, for real: /agent/chat -> the agent -> the planner -> the MCP
# client -> `python -m mcp_servers.jlpt_vocab` in a subprocess -> the
# backend's HTTP client for the JLPT API -> back again. Only the two Gemini
# calls (the plan and the answer) are stubbed, and the far end of the HTTP
# request is the local stand-in for the API.

LOOKUP_学習 = '{"calls": [{"tool": "get_japanese_word_info", "arguments": {"word": "学習"}}]}'
QUESTION = "Что означает 学習? Дай чтение и перевод."


def _enable_tools(monkeypatch, api_url: str, server=jlpt_vocab_server_parameters):
    monkeypatch.setenv("JLPT_VOCAB_API_URL", api_url)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setattr(agent_module.agent, "_tools_enabled", True)
    monkeypatch.setattr(agent_module.agent, "_toolbox", McpToolbox(server))


def _mock_tool_planner(monkeypatch, answer: str) -> list[str]:
    prompts = []

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        prompts.append(prompt)
        return GeneratedText(text=answer, input_tokens=240, output_tokens=25)

    monkeypatch.setattr(tool_planner_module, "generate_text_with_usage", fake_generate_text_with_usage)
    return prompts


def test_the_agent_looks_the_word_up_through_mcp_and_answers_with_it(monkeypatch, tmp_path):
    """3-7. The agent decides to call the tool, the word reaches the API, the
    result comes back through MCP and is in front of the model when it
    writes the answer - right above the question."""
    _isolate_agent(monkeypatch, tmp_path)
    answers = _capture_prompts(monkeypatch, response_text="学習 (がくしゅう) — изучение, N3.")
    _mock_count_tokens(monkeypatch, 90)

    with jlpt_api() as (url, received):
        _enable_tools(monkeypatch, url)
        plans = _mock_tool_planner(monkeypatch, LOOKUP_学習)
        response = client.post("/agent/chat", json={"message": QUESTION})

    assert response.status_code == 200
    body = response.json()

    # the planner was shown the tool list_tools() returned
    assert "get_japanese_word_info" in plans[0]
    # the tool got the word and asked the API with it
    assert received == [{"path": "/api/words", "query": {"word": "学習"}}]
    # the result came back through MCP and is reported with the answer
    assert body["tool_calls"] == [
        {
            "tool": "get_japanese_word_info",
            "arguments": {"word": "学習"},
            "ok": True,
            "result": {
                "query": "学習",
                "found": True,
                "matches": [
                    {
                        "word": "学習",
                        "reading": "がくしゅう",
                        "romaji": "gakushū",
                        "meaning": "study, learning",
                        "jlpt_level": "N3",
                    }
                ],
                "source": "jlpt-vocab-api.vercel.app",
            },
            "error": "",
        }
    ]
    # and the model answered with it in front of it
    prompt = answers[-1]
    assert "TOOL RESULTS" in prompt
    assert '"reading": "がくしゅう"' in prompt
    assert '"meaning": "study, learning"' in prompt
    assert prompt.index("TOOL RESULTS") < prompt.index(f"Learner's request: {QUESTION}")
    assert body["response"] == "学習 (がくしゅう) — изучение, N3."


def test_a_message_that_needs_no_lookup_starts_no_tool(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    answers = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)

    with jlpt_api() as (url, received):
        _enable_tools(monkeypatch, url)
        _mock_tool_planner(monkeypatch, '{"calls": []}')
        body = client.post("/agent/chat", json={"message": "Объясни грамматику 〜ながら"}).json()

    assert body["tool_calls"] == []
    assert received == []
    assert "TOOL RESULTS" not in answers[-1]


def test_an_api_failure_still_ends_in_an_answer_told_that_the_lookup_failed(monkeypatch, tmp_path):
    """8. The API answers 503: the call is reported as failed, with the
    reason, and the model is told to say so rather than guess."""
    _isolate_agent(monkeypatch, tmp_path)
    answers = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)

    with jlpt_api(status=503) as (url, _):
        _enable_tools(monkeypatch, url)
        _mock_tool_planner(monkeypatch, LOOKUP_学習)
        response = client.post("/agent/chat", json={"message": QUESTION})

    assert response.status_code == 200
    call = response.json()["tool_calls"][0]
    assert call["ok"] is False
    assert "status 503" in call["error"]
    assert "FAILED" in answers[-1]
    assert "say so plainly instead of presenting a guess" in answers[-1]


def test_an_mcp_server_that_cannot_start_still_ends_in_an_answer(monkeypatch, tmp_path):
    """8. No server at all: the chat still answers, and the model is told the
    dictionary could not be checked."""
    _isolate_agent(monkeypatch, tmp_path)
    answers = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    broken = lambda: StdioServerParameters(command=sys.executable, args=["-m", "mcp_servers.does_not_exist"])  # noqa: E731
    _enable_tools(monkeypatch, "http://127.0.0.1:9/", server=broken)
    plans = _mock_tool_planner(monkeypatch, LOOKUP_学習)

    response = client.post("/agent/chat", json={"message": QUESTION})

    assert response.status_code == 200
    assert response.json()["tool_calls"] == []
    assert plans == []
    assert "could not be checked in the dictionary" in answers[-1]


def test_a_plan_that_cannot_be_read_means_an_answer_without_tools(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    answers = _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)

    with jlpt_api() as (url, received):
        _enable_tools(monkeypatch, url)
        _mock_tool_planner(monkeypatch, "I think you should look it up")
        response = client.post("/agent/chat", json={"message": QUESTION})

    assert response.status_code == 200
    assert response.json()["tool_calls"] == []
    assert received == []
    assert "TOOL RESULTS" not in answers[-1]


def test_deciding_is_recorded_as_tool_tokens(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)

    with jlpt_api() as (url, _):
        _enable_tools(monkeypatch, url)
        _mock_tool_planner(monkeypatch, LOOKUP_学習)
        client.post("/agent/chat", json={"message": QUESTION})

    assert client.get("/agent/usage").json()["entries"][-1]["tool_tokens"] == 265


def test_with_tools_switched_off_nothing_is_planned_or_called(monkeypatch, tmp_path):
    _isolate_agent(monkeypatch, tmp_path)
    _capture_prompts(monkeypatch)
    _mock_count_tokens(monkeypatch, 90)
    plans = _mock_tool_planner(monkeypatch, LOOKUP_学習)

    body = client.post("/agent/chat", json={"message": QUESTION}).json()

    assert plans == []
    assert body["tool_calls"] == []


# --- the periodic digest through the agent (Day 18) ------------------------

START_DIGEST = (
    '{"calls": [{"tool": "create_periodic_digest",'
    ' "arguments": {"interval_seconds": 15, "query": "N5 words"}}]}'
)
READ_DIGEST = '{"calls": [{"tool": "get_latest_digest", "arguments": {}}]}'


@pytest.fixture
def digest_files(tmp_path, monkeypatch):
    monkeypatch.setenv("DIGEST_TASKS_FILE_PATH", str(tmp_path / "digest_tasks.json"))
    monkeypatch.setenv("DIGEST_STORE_FILE_PATH", str(tmp_path / "digest_store.json"))
    return tmp_path


def test_the_agent_can_start_a_periodic_task_from_a_plain_message(monkeypatch, tmp_path, digest_files):
    _isolate_agent(monkeypatch, tmp_path)
    answers = _capture_prompts(monkeypatch, response_text="Хорошо, буду собирать слова каждые 15 секунд.")
    _mock_count_tokens(monkeypatch, 90)

    with jlpt_api() as (url, _):
        _enable_tools(monkeypatch, url)
        _mock_tool_planner(monkeypatch, START_DIGEST)
        body = client.post(
            "/agent/chat", json={"message": "Собирай слова уровня N5 каждые 15 секунд"}
        ).json()

    call = body["tool_calls"][0]
    assert call["tool"] == "create_periodic_digest"
    assert call["ok"] and call["result"]["task_id"] == "digest-1"
    # the task is on disk, where the scheduler will find it
    assert DigestTaskStorage().latest().interval_seconds == 15
    assert "TOOL RESULTS" in answers[-1]


def test_the_agent_reads_back_what_the_scheduled_runs_collected(monkeypatch, tmp_path, digest_files):
    """The runs happen outside the request, exactly as the scheduler does
    them; the next message reads the digest through MCP and answers with it."""
    _isolate_agent(monkeypatch, tmp_path)
    answers = _capture_prompts(monkeypatch, response_text="Собрано 6 слов за 2 запуска.")
    _mock_count_tokens(monkeypatch, 90)
    task = DigestTaskStorage().create("N5 words", 15)

    with jlpt_api() as (url, _):
        _enable_tools(monkeypatch, url)
        asyncio.run(collect_once(task, DigestStore()))
        asyncio.run(collect_once(task, DigestStore()))
        _mock_tool_planner(monkeypatch, READ_DIGEST)
        body = client.post("/agent/chat", json={"message": "Покажи сводку собранных слов"}).json()

    call = body["tool_calls"][0]
    assert call["tool"] == "get_latest_digest"
    assert call["result"]["runs"] == 2
    assert call["result"]["items_collected"] == 6
    # and the model wrote its answer with that in front of it
    assert '"runs": 2' in answers[-1]
    assert '"items_collected": 6' in answers[-1]
    assert body["response"] == "Собрано 6 слов за 2 запуска."
