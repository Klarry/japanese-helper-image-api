"""Day 27: the app's chat, answered by the model on this machine.

The claim this day makes is not "there is an endpoint" - Day 26 already had
one. It is that in this mode **no cloud model is involved**. So the tests
that matter here are the ones that would catch a fallback: Gemini is
sabotaged so that any call to it raises, and the local flow is driven end to
end through a mock Ollama. If a fallback ever appears, these fail.
"""

import asyncio
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import gemini_service, ollama_service

client = TestClient(app)


def serve(handler):
    def factory():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return factory


def answering(body: dict, status: int = 200):
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


def no_cloud(monkeypatch):
    """Make every way out to Gemini explode.

    Not a mock that records the call - one that fails the test the moment it
    happens. A fallback that silently worked would otherwise look exactly
    like a local answer.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError("the local flow called Gemini")

    for name in (
        "_post",
        "_post_to_gemini",
        "generate_text",
        "generate_text_with_usage",
        "count_tokens",
        "embed_texts",
        "search_image",
    ):
        monkeypatch.setattr(gemini_service, name, forbidden)


# --- the endpoint ----------------------------------------------------------


def test_the_app_gets_the_answer_the_model_and_who_produced_it(monkeypatch):
    no_cloud(monkeypatch)
    monkeypatch.setattr(
        ollama_service,
        "_client",
        serve(answering({"model": "qwen3:4b", "response": "MCP is a protocol."})),
    )

    response = client.post("/local-chat", json={"message": "What is MCP?"})
    body = response.json()

    assert response.status_code == 200
    assert body["response"] == "MCP is a protocol."
    assert body["model"] == "qwen3:4b"
    assert body["provider"] == "ollama"


def test_the_message_reaches_ollama_unchanged(monkeypatch):
    """The device sends a message; the backend sends that message. Nothing
    on the way rewrites it, and nothing adds a system prompt it did not ask
    for."""
    import json

    handler = answering({"response": "ok"})
    monkeypatch.setattr(ollama_service, "_client", serve(handler))

    client.post("/local-chat", json={"message": "Что такое MCP?"})
    sent = json.loads(handler.seen[0].content)

    assert str(handler.seen[0].url) == "http://localhost:11434/api/generate"
    assert sent == {"model": "qwen3:4b", "prompt": "Что такое MCP?", "stream": False}


def test_the_device_sends_a_message_and_nothing_else():
    """It cannot choose the model or the provider: those are the mode, and a
    client that could set them could also contradict them."""
    from app.schemas.local_llm import LocalChatRequest

    assert set(LocalChatRequest.model_fields) == {"message"}


def test_an_empty_message_is_refused_before_the_model_is_loaded():
    assert client.post("/local-chat", json={"message": "   "}).status_code == 422


# --- no cloud, and no fallback to it ---------------------------------------


def test_the_local_flow_never_calls_gemini(monkeypatch):
    """The whole point of Day 27. Every Gemini entry point raises; a 200
    here means the answer came from the local model and could not have come
    from anywhere else."""
    no_cloud(monkeypatch)
    monkeypatch.setattr(ollama_service, "_client", serve(answering({"response": "local"})))

    assert client.post("/local-chat", json={"message": "hi"}).status_code == 200


def test_a_local_model_that_is_down_does_not_become_a_cloud_model(monkeypatch):
    """The failure that would be easiest to hide. If Ollama is unreachable
    the request fails - it does not quietly succeed from a datacentre."""
    no_cloud(monkeypatch)
    monkeypatch.setattr(ollama_service, "_client", serve(raising(httpx.ConnectError("refused"))))

    response = client.post("/local-chat", json={"message": "hi"})

    assert response.status_code == 503
    assert response.json()["error"] == "Local LLM is unavailable"


def test_the_route_module_has_no_way_to_reach_the_cloud():
    """Structural, not behavioural: the module that answers this mode does
    not import the cloud service at all, so no later edit can reach it by
    accident.

    Read from the parsed imports and the module's own namespace rather than
    by searching the text - a comment explaining that Gemini is not called
    would fail a text search, and a cloud call hidden behind an alias would
    pass one.
    """
    import ast

    from app.api.routes import local_llm

    tree = ast.parse(open(local_llm.__file__, encoding="utf-8").read())
    imported = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported += [f"{node.module}.{alias.name}" for alias in node.names]

    assert imported, "the module imports nothing at all - the parse went wrong"
    assert not [name for name in imported if "gemini" in name.lower()]
    assert not [
        name
        for name, value in vars(local_llm).items()
        if getattr(value, "__name__", "").lower().find("gemini") >= 0
    ]


# --- what the app is told when it cannot be answered -----------------------


def test_an_unreachable_model_says_so_in_one_line_and_keeps_the_reason(monkeypatch):
    """One fixed line for the screen, and the real cause beside it - the
    Day 26 rule kept: nothing is swallowed."""
    monkeypatch.setattr(ollama_service, "_client", serve(raising(httpx.ConnectError("refused"))))

    body = client.post("/local-chat", json={"message": "hi"}).json()

    assert body["error"] == "Local LLM is unavailable"
    assert body["reason"] == "not_running"
    assert "ollama serve" in body["detail"]


def test_an_empty_answer_is_a_failure_not_an_empty_bubble(monkeypatch):
    monkeypatch.setattr(ollama_service, "_client", serve(answering({"response": "   "})))

    response = client.post("/local-chat", json={"message": "hi"})

    assert response.status_code == 502
    assert response.json()["reason"] == "empty_response"


def test_an_http_error_from_ollama_reaches_the_app_with_its_body(monkeypatch):
    monkeypatch.setattr(
        ollama_service, "_client", serve(answering({"error": "out of memory"}, status=500))
    )

    body = client.post("/local-chat", json={"message": "hi"}).json()

    assert body["error"] == "Local LLM is unavailable"
    assert body["reason"] == "http_error"
    assert "out of memory" in body["detail"]


def test_a_model_that_was_never_pulled_tells_the_app_to_pull_it(monkeypatch):
    monkeypatch.setattr(
        ollama_service, "_client", serve(answering({"error": "not found"}, status=404))
    )

    body = client.post("/local-chat", json={"message": "hi"}).json()

    assert body["reason"] == "model_not_installed"
    assert "ollama pull qwen3:4b" in body["detail"]


# --- what the app reads to draw its status line ----------------------------


def test_the_health_endpoint_tells_the_app_the_model_is_there(monkeypatch):
    monkeypatch.setattr(
        ollama_service,
        "_client",
        serve(answering({"models": [{"name": "qwen3:4b"}]})),
    )

    body = client.get("/local-llm/health").json()

    assert body["available"] is True
    assert body["model"] == "qwen3:4b"


def test_the_health_endpoint_is_still_200_when_the_model_is_not(monkeypatch):
    """So "unavailable" and "this check is broken" stay distinguishable on
    the screen."""
    monkeypatch.setattr(ollama_service, "_client", serve(raising(httpx.ConnectError("refused"))))

    response = client.get("/local-llm/health")

    assert response.status_code == 200
    assert response.json()["available"] is False


# --- the logging the assignment asks for -----------------------------------


def test_the_local_path_announces_itself_in_the_log(monkeypatch, caplog):
    """Provider, model, request, response - labelled [LocalLLM], and emitted
    from the service rather than a route, so every caller gets them."""
    monkeypatch.setattr(ollama_service, "_client", serve(answering({"response": "ok"})))

    with caplog.at_level(logging.INFO, logger="app.services.ollama_service"):
        client.post("/local-chat", json={"message": "hi"})

    printed = "\n".join(record.getMessage() for record in caplog.records)

    assert "[LocalLLM] Provider: ollama" in printed
    assert "[LocalLLM] Model: qwen3:4b" in printed
    assert "[LocalLLM] Request sent" in printed
    assert "[LocalLLM] Response received" in printed


# --- the one test that wants a real model ----------------------------------


def _ollama_is_running() -> bool:
    try:
        return httpx.get(
            f"{ollama_service.BASE_URL}{ollama_service.TAGS_PATH}", timeout=2
        ).status_code == 200
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _ollama_is_running(), reason="Ollama is not running on this machine")
def test_the_real_local_model_answers_the_app():
    """The same request the Android app makes, against the model actually
    installed on this machine."""
    health = asyncio.run(ollama_service.check_ollama_health())

    if not health["model_installed"]:
        pytest.skip(f"{health['model']} is not pulled on this machine")

    response = client.post("/local-chat", json={"message": "What is MCP? Answer in one sentence."})
    body = response.json()

    assert response.status_code == 200
    assert body["response"].strip()
    assert body["provider"] == "ollama"
    assert body["seconds"] > 0
