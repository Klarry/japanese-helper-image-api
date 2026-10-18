"""Day 33: the support assistant against the real local model.

Opt-in - it needs Ollama running with the model, and the help-centre index
built (python -m app.index_documents --corpus support --embeddings local
--strategy structural --index-dir data/index/support):

    RUN_DAY33_INTEGRATION=1 OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M \\
        .venv-mac/bin/python -m pytest tests/test_day33_integration.py -v

Everything is real: the support-crm MCP server, the JSON files, the index,
Ollama. Every Gemini function is a trap. What is checked is the contract,
not the wording: an answer from Ollama, the ticket's user resolved through
MCP, every citation pointing at a chunk the help centre holds.
"""

import asyncio
import os

import pytest

from app.evaluate_day33 import trap_all_gemini
from app.services.llm_provider import OLLAMA
from app.services.support_assistant import ANSWERED, INSUFFICIENT, SupportAssistant

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DAY33_INTEGRATION") != "1", reason="set RUN_DAY33_INTEGRATION=1 to use the real Ollama, MCP server and index"
)


def test_a_real_local_support_answer():
    trap_all_gemini()
    answer = asyncio.run(SupportAssistant(OLLAMA).ask("Почему не работает авторизация?", "TICKET-1042")).as_dict()

    assert answer["provider"] == "ollama" and answer["model"]
    assert [tool["name"] for tool in answer["tools_used"]] == ["get_ticket", "get_user"]
    assert all(tool["ok"] for tool in answer["tools_used"])
    assert answer["ticket"]["ticket_id"] == "TICKET-1042" and answer["answer"]
    assert answer["status"] in (ANSWERED, INSUFFICIENT)
    assert {citation["chunk_id"] for citation in answer["citations"]} <= {source["chunk_id"] for source in answer["sources"]}
