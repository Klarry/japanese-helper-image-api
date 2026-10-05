"""Day 26: the local LLM, its endpoint and its CLI.

Nothing here calls a real model. Every test drives the service through
``httpx.MockTransport``, so the request this module actually builds is the
one under test - the payload, the URL, the timeout handling - while no
socket is ever opened. The one test that does want a real model is at the
bottom and skips itself unless Ollama answers.
"""

import asyncio
import json
import os
import pathlib

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import ollama_service
from app.services.ollama_service import OllamaUnavailable

client = TestClient(app)


def serve(handler):
    """Point the service at a mock transport instead of the network."""

    def factory():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return factory


def answering(body: dict, status: int = 200):
    """A transport that returns one canned reply to everything."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body)

    handler.seen = seen

    return handler


def raising(error: Exception):
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    return handler


def tags(*names: str) -> dict:
    return {"models": [{"name": name} for name in names]}


# --- configuration ---------------------------------------------------------


def test_the_url_and_the_model_come_from_the_environment():
    """Neither is written into the business logic: both are module-level
    settings read from the environment, with the defaults config declares."""
    source = (ollama_service.__file__).replace(".pyc", ".py")
    text = open(source, encoding="utf-8").read()

    assert 'os.getenv("OLLAMA_BASE_URL"' in text
    assert 'os.getenv("OLLAMA_MODEL"' in text
    assert ollama_service.BASE_URL == "http://localhost:11434"
    assert ollama_service.MODEL == "qwen3:4b"


def test_the_local_model_needs_no_cloud_key():
    """The whole point of running a model locally.

    app.core.config reads GEMINI_API_KEY at import time and raises without
    it, so this proves the claim the way it matters: a fresh interpreter,
    the variable deliberately unset, importing the service and reading its
    settings. Grepping the source for the import would pass on a comment.
    """
    import subprocess
    import sys

    environment = {key: value for key, value in os.environ.items() if key != "GEMINI_API_KEY"}
    result = subprocess.run(
        [sys.executable, "-c", "from app.services import ollama_service as o; print(o.MODEL)"],
        capture_output=True,
        text=True,
        env=environment,
        cwd=pathlib.Path(ollama_service.__file__).parents[2],
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "qwen3:4b"


def test_nothing_existing_was_rewired_through_the_local_model():
    """Day 26 adds a provider; it does not replace one. If the agent, the
    RAG pipeline, the mini chat or the MCP tools ever import this service,
    that is a different day's decision and this test should be the one
    that notices."""
    service_dir = pathlib.Path(ollama_service.__file__).parent
    importers = [
        path.name
        for path in service_dir.glob("*.py")
        if "ollama_service" in path.read_text(encoding="utf-8") and path.name != "ollama_service.py"
    ]

    assert importers == []


# --- health check ----------------------------------------------------------


def test_health_reports_the_model_and_what_is_installed(monkeypatch):
    handler = answering(tags("qwen3:4b", "llama3:8b"))
    monkeypatch.setattr(ollama_service, "_client", serve(handler))

    health = asyncio.run(ollama_service.check_ollama_health())

    assert health["available"] is True
    assert health["model"] == "qwen3:4b"
    assert health["model_installed"] is True
    assert health["models"] == ["llama3:8b", "qwen3:4b"]
    assert str(handler.seen[0].url).endswith("/api/tags")


def test_health_says_so_when_the_model_was_never_pulled(monkeypatch):
    """Ollama being up and the model being there are different facts."""
    monkeypatch.setattr(ollama_service, "_client", serve(answering(tags("llama3:8b"))))

    health = asyncio.run(ollama_service.check_ollama_health())

    assert health["available"] is True
    assert health["model_installed"] is False


def test_health_does_not_hide_a_server_that_is_not_running(monkeypatch):
    monkeypatch.setattr(
        ollama_service, "_client", serve(raising(httpx.ConnectError("refused")))
    )

    health = asyncio.run(ollama_service.check_ollama_health())

    assert health["available"] is False
    assert "ollama serve" in health["error"]
    assert health["model"] == "qwen3:4b"


def test_health_does_not_raise_when_ollama_is_down(monkeypatch):
    """A health check that throws is no use to whoever is asking whether it
    is safe to call."""
    monkeypatch.setattr(ollama_service, "_client", serve(raising(httpx.ConnectError("refused"))))

    assert asyncio.run(ollama_service.check_ollama_health())["available"] is False


def test_health_reports_an_http_error_from_the_server(monkeypatch):
    monkeypatch.setattr(ollama_service, "_client", serve(answering({}, status=500)))

    health = asyncio.run(ollama_service.check_ollama_health())

    assert health["available"] is False
    assert "500" in health["error"]


# --- generation ------------------------------------------------------------


def test_a_successful_generation_returns_the_text_and_the_model(monkeypatch):
    handler = answering(
        {
            "model": "qwen3:4b",
            "response": "MCP is a protocol.",
            "prompt_eval_count": 12,
            "eval_count": 34,
        }
    )
    monkeypatch.setattr(ollama_service, "_client", serve(handler))

    answer = asyncio.run(ollama_service.generate("What is MCP?"))

    assert answer.response == "MCP is a protocol."
    assert answer.model == "qwen3:4b"
    assert answer.prompt_tokens == 12
    assert answer.response_tokens == 34
    assert answer.as_dict() == {"response": "MCP is a protocol.", "model": "qwen3:4b"}


def test_the_request_is_the_one_the_api_documents(monkeypatch):
    handler = answering({"response": "ok"})
    monkeypatch.setattr(ollama_service, "_client", serve(handler))

    asyncio.run(ollama_service.generate("What is MCP?"))

    request = handler.seen[0]

    assert str(request.url) == "http://localhost:11434/api/generate"
    assert json.loads(request.content) == {
        "model": "qwen3:4b",
        "prompt": "What is MCP?",
        "stream": False,
    }


def test_streaming_is_off_because_there_is_nobody_watching(monkeypatch):
    """A caller that got half an answer could not tell it from a whole one."""
    handler = answering({"response": "ok"})
    monkeypatch.setattr(ollama_service, "_client", serve(handler))

    asyncio.run(ollama_service.generate("hi"))

    assert json.loads(handler.seen[0].content)["stream"] is False


def test_a_model_can_be_chosen_per_call(monkeypatch):
    handler = answering({"response": "ok"})
    monkeypatch.setattr(ollama_service, "_client", serve(handler))

    asyncio.run(ollama_service.generate("hi", model="llama3:8b"))

    assert json.loads(handler.seen[0].content)["model"] == "llama3:8b"


def test_a_thinking_model_does_not_narrate_into_the_answer(monkeypatch):
    """qwen3 thinks out loud. The narration is separated from the answer -
    kept, because it is real output, but not printed as the answer."""
    monkeypatch.setattr(
        ollama_service,
        "_client",
        serve(answering({"response": "<think>weighing it up</think>MCP is a protocol."})),
    )

    answer = asyncio.run(ollama_service.generate("What is MCP?"))

    assert answer.response == "MCP is a protocol."
    assert answer.thinking == "weighing it up"


def test_ollamas_own_thinking_field_is_used_when_it_sends_one(monkeypatch):
    """Newer builds return it separately instead of inline."""
    monkeypatch.setattr(
        ollama_service,
        "_client",
        serve(answering({"response": "An answer.", "thinking": "weighing it up"})),
    )

    answer = asyncio.run(ollama_service.generate("q"))

    assert answer.response == "An answer."
    assert answer.thinking == "weighing it up"


# --- the five ways this goes wrong -----------------------------------------


def test_a_server_that_is_not_running_says_how_to_start_it(monkeypatch):
    monkeypatch.setattr(ollama_service, "_client", serve(raising(httpx.ConnectError("refused"))))

    with pytest.raises(OllamaUnavailable) as caught:
        asyncio.run(ollama_service.generate("hi"))

    assert caught.value.reason == "not_running"
    assert "ollama serve" in caught.value.message
    assert "http://localhost:11434" in caught.value.message


def test_a_timeout_says_it_timed_out_and_how_long_it_waited(monkeypatch):
    monkeypatch.setattr(ollama_service, "_client", serve(raising(httpx.ReadTimeout("slow"))))

    with pytest.raises(OllamaUnavailable) as caught:
        asyncio.run(ollama_service.generate("hi"))

    assert caught.value.reason == "timeout"
    assert "300" in caught.value.message
    assert "OLLAMA_TIMEOUT" in caught.value.message


def test_a_model_that_was_never_pulled_says_how_to_pull_it(monkeypatch):
    """Ollama answers a request for a model it does not have with a 404."""
    monkeypatch.setattr(
        ollama_service,
        "_client",
        serve(answering({"error": 'model "qwen3:4b" not found'}, status=404)),
    )

    with pytest.raises(OllamaUnavailable) as caught:
        asyncio.run(ollama_service.generate("hi"))

    assert caught.value.reason == "model_not_installed"
    assert "ollama pull qwen3:4b" in caught.value.message


def test_an_http_error_carries_the_status_and_the_body(monkeypatch):
    """Not swallowed: the body is the only thing that explains a 500."""
    monkeypatch.setattr(
        ollama_service, "_client", serve(answering({"error": "out of memory"}, status=500))
    )

    with pytest.raises(OllamaUnavailable) as caught:
        asyncio.run(ollama_service.generate("hi"))

    assert caught.value.reason == "http_error"
    assert "500" in caught.value.message
    assert "out of memory" in caught.value.message


def test_an_empty_response_is_an_error_not_an_answer(monkeypatch):
    monkeypatch.setattr(ollama_service, "_client", serve(answering({"response": "   "})))

    with pytest.raises(OllamaUnavailable) as caught:
        asyncio.run(ollama_service.generate("hi"))

    assert caught.value.reason == "empty_response"
    assert "qwen3:4b" in caught.value.message


def test_a_model_that_only_thought_still_answered_nothing(monkeypatch):
    monkeypatch.setattr(
        ollama_service, "_client", serve(answering({"response": "<think>hmm</think>"}))
    )

    with pytest.raises(OllamaUnavailable) as caught:
        asyncio.run(ollama_service.generate("hi"))

    assert caught.value.reason == "empty_response"


# --- the endpoint ----------------------------------------------------------


def test_the_endpoint_returns_the_model_and_the_response(monkeypatch):
    monkeypatch.setattr(
        ollama_service,
        "_client",
        serve(answering({"model": "qwen3:4b", "response": "MCP is a protocol."})),
    )

    response = client.post("/local-llm", json={"prompt": "What is MCP?"})

    assert response.status_code == 200
    assert response.json() == {"model": "qwen3:4b", "response": "MCP is a protocol."}


def test_the_endpoint_refuses_an_empty_prompt():
    assert client.post("/local-llm", json={"prompt": "   "}).status_code == 422


def test_the_endpoint_passes_the_real_reason_through(monkeypatch):
    """503, and the message says how to start the server - the person
    reading the response is the person who can fix it."""
    monkeypatch.setattr(ollama_service, "_client", serve(raising(httpx.ConnectError("refused"))))

    response = client.post("/local-llm", json={"prompt": "hi"})

    assert response.status_code == 503
    assert "ollama serve" in response.json()["detail"]


def test_a_bad_answer_from_a_running_server_is_a_bad_gateway(monkeypatch):
    monkeypatch.setattr(ollama_service, "_client", serve(answering({"response": ""})))

    assert client.post("/local-llm", json={"prompt": "hi"}).status_code == 502


def test_the_health_endpoint_is_200_even_when_ollama_is_down(monkeypatch):
    """A health check that returns an error status for an unhealthy
    dependency cannot be told apart from one that is broken itself."""
    monkeypatch.setattr(ollama_service, "_client", serve(raising(httpx.ConnectError("refused"))))

    response = client.get("/local-llm/health")

    assert response.status_code == 200
    assert response.json()["available"] is False
    assert "ollama serve" in response.json()["error"]


def test_the_health_endpoint_reports_a_working_model(monkeypatch):
    monkeypatch.setattr(ollama_service, "_client", serve(answering(tags("qwen3:4b"))))

    body = client.get("/local-llm/health").json()

    assert body == {
        "available": True,
        "model": "qwen3:4b",
        "model_installed": True,
        "models": ["qwen3:4b"],
        "error": None,
    }


# --- the CLI ---------------------------------------------------------------


def test_the_cli_asks_the_three_prompts_the_assignment_names():
    from app.local_llm import DEMO_PROMPTS

    difficulties = [name for name, _ in DEMO_PROMPTS]
    prompts = [prompt for _, prompt in DEMO_PROMPTS]

    assert difficulties == ["simple", "medium", "complex"]
    assert prompts[0] == "What is MCP?"
    assert "MCP client and MCP server" in prompts[1]
    assert "multiple MCP servers" in prompts[2]
    assert len(set(prompts)) == 3


def test_the_cli_checks_the_server_before_it_sends_anything(monkeypatch, capsys):
    """"Connection refused" after a thirty-second wait is a worse answer
    than "it is not running, here is how to start it"."""
    from app import local_llm

    async def down(model=""):
        return {"available": False, "model": "qwen3:4b", "error": "Ollama is not running.\nollama serve"}

    async def must_not_be_called(*args, **kwargs):
        raise AssertionError("the CLI asked a server it had been told was down")

    monkeypatch.setattr(local_llm.ollama_service, "check_ollama_health", down)
    monkeypatch.setattr(local_llm.ollama_service, "generate", must_not_be_called)

    assert asyncio.run(local_llm.main(["What is MCP?"])) == 1
    assert "ollama serve" in capsys.readouterr().out


def test_the_cli_prints_the_model_and_the_response(monkeypatch, capsys):
    from app import local_llm

    async def up(model=""):
        return {"available": True, "model": "qwen3:4b", "model_installed": True, "models": ["qwen3:4b"]}

    async def answer(prompt, model=""):
        return ollama_service.LocalAnswer(response="MCP is a protocol.", model="qwen3:4b", seconds=1.5)

    monkeypatch.setattr(local_llm.ollama_service, "check_ollama_health", up)
    monkeypatch.setattr(local_llm.ollama_service, "generate", answer)

    assert asyncio.run(local_llm.main(["What is MCP?"])) == 0

    printed = capsys.readouterr().out

    assert "Ollama: available" in printed
    assert "Model: qwen3:4b" in printed
    assert "Prompt:\nWhat is MCP?" in printed
    assert "Response:\nMCP is a protocol." in printed


def test_the_cli_says_when_the_model_is_missing_rather_than_asking_for_it(monkeypatch, capsys):
    from app import local_llm

    async def missing(model=""):
        return {
            "available": True,
            "model": "qwen3:4b",
            "model_installed": False,
            "models": ["llama3:8b"],
        }

    monkeypatch.setattr(local_llm.ollama_service, "check_ollama_health", missing)

    assert asyncio.run(local_llm.main(["hi"])) == 1
    assert "ollama pull qwen3:4b" in capsys.readouterr().out


def test_the_cli_needs_something_to_ask():
    from app import local_llm

    assert asyncio.run(local_llm.main([])) == 2


# --- the one test that wants a real model ----------------------------------


def _ollama_is_running() -> bool:
    try:
        return httpx.get(f"{ollama_service.BASE_URL}{ollama_service.TAGS_PATH}", timeout=2).status_code == 200
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _ollama_is_running(), reason="Ollama is not running on this machine")
def test_the_real_local_model_answers():
    """The only test here that loads weights. It skips itself unless Ollama
    is actually up, so the suite stays runnable on a machine without it -
    and runs for real on one where it is."""
    health = asyncio.run(ollama_service.check_ollama_health())

    assert health["available"] is True

    if not health["model_installed"]:
        pytest.skip(f"{health['model']} is not pulled on this machine")

    answer = asyncio.run(ollama_service.generate("What is MCP? Answer in one sentence."))

    assert answer.response.strip()
    assert answer.model.startswith(health["model"].split(":")[0])
    assert answer.response_tokens is None or answer.response_tokens > 0
