"""Day 30: the local model as a service other machines can use.

Every test here runs against a stand-in for Ollama - httpx's MockTransport,
or an async fake of ``ollama_service.generate`` where a test needs to hold
a generation open. None of them proves that a real model answered over a
real network; ``test_day30_network.py`` and ``python -m app.evaluate_day30``
do that against a running service, and say so.
"""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.routes import local_llm
from app.main import app
from app.services import gemini_service, ollama_service
from app.services.local_chat_guard import (
    ConcurrencyGate,
    LocalChatLimits,
    RateLimiter,
    estimate_tokens,
    plan_prompt,
)
from app.services.agent_history_storage import format_transcript
from app.services.ollama_service import LocalAnswer, OllamaUnavailable

client = TestClient(app)


def serve(handler):
    def factory():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return factory


class Ollama:
    """Answers /api/tags and /api/generate, and remembers every prompt."""

    def __init__(self, models=("qwen3:4b",), reply="Model Context Protocol.", down=False):
        self.models = list(models)
        self.reply = reply
        self.down = down
        self.prompts: list[str] = []
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("refused")

        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": name} for name in self.models]})

        body = json.loads(request.content)
        self.bodies.append(body)
        self.prompts.append(body["prompt"])

        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "response": self.reply,
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 21,
                "eval_count": 9,
                "eval_duration": 300_000_000,
            },
        )


@pytest.fixture
def limits(tmp_path):
    """Fresh guards with the given limits, conversations kept in tmp."""

    def apply(**overrides) -> LocalChatLimits:
        chosen = LocalChatLimits(history_dir=str(tmp_path / "chats"), **overrides)
        local_llm.reset_guard(chosen)
        return chosen

    apply()
    return apply


def ollama(monkeypatch, **kwargs) -> Ollama:
    fake = Ollama(**kwargs)
    monkeypatch.setattr(ollama_service, "_client", serve(fake))
    return fake


def no_cloud(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the local service called Gemini")

    for name in ("_post", "_post_to_gemini", "generate_text", "generate_text_with_usage", "count_tokens"):
        monkeypatch.setattr(gemini_service, name, forbidden)


def chat(message="What is MCP?", **extra):
    return client.post("/local-chat", json={"message": message, **extra})


# --- 1-2. health -----------------------------------------------------------------


def test_health_says_ready_only_when_the_model_can_answer(monkeypatch, limits):
    ollama(monkeypatch)
    body = client.get("/local-llm/health").json()

    assert body["available"] is True
    assert body["model_installed"] is True
    assert body["ready"] is True
    assert body["service"]["rate_limit"] == 10
    assert body["service"]["max_concurrent"] == 2
    assert body["service"]["options"] == {"options": {"num_ctx": 4096}}
    assert client.get("/local-llm/health?strict=true").status_code == 200


def test_health_with_ollama_down_is_not_healthy(monkeypatch, limits):
    ollama(monkeypatch, down=True)

    relaxed = client.get("/local-llm/health")
    strict = client.get("/local-llm/health?strict=true")

    # The app's contract (Day 27): 200, verdict in the body.
    assert relaxed.status_code == 200
    assert relaxed.json()["ready"] is False
    # For anything that reads only the status line: FastAPI being up is not
    # the model being up.
    assert strict.status_code == 503
    assert strict.json()["available"] is False
    assert "ollama serve" in strict.json()["error"]


def test_health_with_the_model_missing_is_not_healthy(monkeypatch, limits):
    ollama(monkeypatch, models=("llama3:8b",))
    strict = client.get("/local-llm/health?strict=true")

    assert strict.status_code == 503
    assert strict.json()["available"] is True
    assert strict.json()["model_installed"] is False


# --- 3. a successful chat ----------------------------------------------------------


def test_a_chat_answer_carries_its_id_model_and_measured_time(monkeypatch, limits):
    fake = ollama(monkeypatch)
    response = chat()
    body = response.json()

    assert response.status_code == 200
    # Day 27's four fields, unchanged.
    assert body["response"] == "Model Context Protocol."
    assert body["model"] == "qwen3:4b"
    assert body["provider"] == "ollama"
    assert isinstance(body["seconds"], float)
    # Day 30's additions.
    assert len(body["request_id"]) == 12
    assert response.headers["X-Request-ID"] == body["request_id"]
    assert set(body["timing"]) == {"generation_ms", "total_ms"}
    assert body["timing"]["total_ms"] >= body["timing"]["generation_ms"] >= 0
    assert body["usage"]["prompt_tokens"] == 21
    assert body["usage"]["response_tokens"] == 9
    # Without a conversation the prompt is the message itself, as on Day 27.
    assert fake.prompts == ["What is MCP?"]
    assert fake.bodies[0]["options"]["num_ctx"] == 4096


def test_two_requests_get_two_ids(monkeypatch, limits):
    ollama(monkeypatch)

    assert chat().json()["request_id"] != chat().json()["request_id"]


# --- 4. invalid and oversized input ---------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"message": "   "},
        {},
        {"message": 42},
        {"message": "hi", "conversation_id": "../../etc/passwd"},
        {"message": "hi", "conversation_id": "x" * 65},
    ],
)
def test_an_invalid_request_is_refused_before_the_model(monkeypatch, limits, body):
    fake = ollama(monkeypatch)

    assert client.post("/local-chat", json=body).status_code == 422
    assert fake.prompts == []


def test_a_message_over_the_character_limit_is_413(monkeypatch, limits):
    fake = ollama(monkeypatch)
    response = chat("a" * 4001)

    assert response.status_code == 413
    assert response.json()["reason"] == "input_too_large"
    assert response.json()["request_id"]
    assert fake.prompts == []
    assert chat("a" * 4000).status_code == 200


def test_characters_are_not_tokens(monkeypatch, limits):
    """3 500 characters of Japanese pass the character limit and fail the
    context budget: a kana is about a token, not a third of one."""
    fake = ollama(monkeypatch)
    response = chat("あ" * 3500)

    assert response.status_code == 413
    assert response.json()["reason"] == "context_too_large"
    assert "refused rather than cut" in response.json()["detail"]
    assert fake.prompts == []


def test_token_estimates_lean_high_and_know_about_kana():
    assert estimate_tokens("あいうえお") == 5
    assert estimate_tokens("abcdef") == 2
    assert estimate_tokens("学習 is study") >= 2 + 3


# --- 5. rate limit ----------------------------------------------------------------------


def test_the_request_over_the_limit_gets_429_with_retry_after(monkeypatch, limits):
    limits(rate_limit=3, rate_window_seconds=60)
    fake = ollama(monkeypatch)

    statuses = [chat().status_code for _ in range(4)]
    refused = chat()

    assert statuses == [200, 200, 200, 429]
    assert refused.status_code == 429
    assert refused.json()["reason"] == "rate_limited"
    assert 1 <= int(refused.headers["Retry-After"]) <= 60
    assert refused.json()["retry_after"] == int(refused.headers["Retry-After"])
    assert len(fake.prompts) == 3


def test_the_window_slides():
    now = [0.0]
    limiter = RateLimiter(2, 60, clock=lambda: now[0])

    assert [limiter.check("a").allowed for _ in range(3)] == [True, True, False]
    assert limiter.check("b").allowed is True
    now[0] = 59.9
    assert limiter.check("a").allowed is False
    now[0] = 60.0
    assert limiter.check("a").allowed is True


def test_a_forwarded_for_header_is_not_believed_by_default(monkeypatch, limits):
    limits(rate_limit=1)
    ollama(monkeypatch)

    first = client.post("/local-chat", json={"message": "hi"}, headers={"X-Forwarded-For": "10.0.0.1"})
    second = client.post("/local-chat", json={"message": "hi"}, headers={"X-Forwarded-For": "10.0.0.2"})

    assert (first.status_code, second.status_code) == (200, 429)


def test_a_trusted_proxy_is_believed(monkeypatch, limits):
    limits(rate_limit=1, trusted_proxies=("testclient",))
    ollama(monkeypatch)

    first = client.post("/local-chat", json={"message": "hi"}, headers={"X-Forwarded-For": "10.0.0.1"})
    second = client.post("/local-chat", json={"message": "hi"}, headers={"X-Forwarded-For": "10.0.0.2"})

    assert (first.status_code, second.status_code) == (200, 200)


# --- 6-8. concurrency, timeout, simultaneous requests ----------------------------------------


class HeldModel:
    """A generate() that stays busy until released - and counts how many
    are running at once, which is what shows the loop is not blocked."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.running = 0
        self.peak = 0

    async def __call__(self, prompt, model="", options=None):
        self.running += 1
        self.peak = max(self.peak, self.running)

        try:
            await self.release.wait()
        finally:
            self.running -= 1

        return LocalAnswer(response=f"answer to {prompt}", model=model or "qwen3:4b", seconds=0.01)


async def _post_many(count: int, release_after: float, held: HeldModel):
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        tasks = [
            asyncio.create_task(http.post("/local-chat", json={"message": f"q{index}"}))
            for index in range(count)
        ]
        await asyncio.sleep(release_after)
        held.release.set()

        return await asyncio.gather(*tasks)


def test_more_than_the_cap_at_once_gets_busy_not_a_queue(monkeypatch, limits):
    limits(max_concurrent=2)
    held = HeldModel()
    monkeypatch.setattr(ollama_service, "generate", held)

    responses = asyncio.run(_post_many(3, 0.2, held))
    statuses = sorted(response.status_code for response in responses)
    busy = next(response for response in responses if response.status_code == 503)

    assert statuses == [200, 200, 503]
    assert busy.json()["reason"] == "busy"
    assert held.peak == 2
    assert local_llm.guard.gate.active == 0


def test_simultaneous_requests_run_side_by_side(monkeypatch, limits):
    """Five at once with five slots: all answered, and all five were inside
    the model together - the event loop was never blocked by one of them."""
    limits(max_concurrent=5)
    held = HeldModel()
    monkeypatch.setattr(ollama_service, "generate", held)

    responses = asyncio.run(_post_many(5, 0.2, held))

    assert [response.status_code for response in responses] == [200] * 5
    assert len({response.json()["request_id"] for response in responses}) == 5
    assert held.peak == 5
    assert local_llm.guard.gate.active == 0


def test_a_generation_that_runs_too_long_is_cancelled_and_frees_its_slot(monkeypatch, limits):
    limits(timeout_seconds=0.05, max_concurrent=1)

    async def slow(prompt, model="", options=None):
        await asyncio.sleep(5)

    monkeypatch.setattr(ollama_service, "generate", slow)
    response = chat()

    assert response.status_code == 503
    assert response.json()["reason"] == "timeout"
    assert "LOCAL_CHAT_TIMEOUT_SECONDS" in response.json()["detail"]
    assert local_llm.guard.gate.active == 0


def test_a_failed_generation_does_not_leak_a_slot(monkeypatch, limits):
    limits(max_concurrent=1)
    fake = ollama(monkeypatch, down=True)

    assert chat().status_code == 503
    assert local_llm.guard.gate.active == 0
    fake.down = False
    assert chat().status_code == 200


def test_the_gate_cannot_go_negative():
    gate = ConcurrencyGate(1)
    gate.leave()

    assert gate.active == 0
    assert gate.try_enter() is True
    assert gate.try_enter() is False


# --- 9. no cloud ---------------------------------------------------------------------------


def test_no_outcome_of_the_local_chat_reaches_gemini(monkeypatch, limits):
    limits(rate_limit=4, max_concurrent=1)
    no_cloud(monkeypatch)
    fake = ollama(monkeypatch)

    answered = chat()
    fake.down = True
    down = chat()
    too_long = chat("a" * 5000)
    chat()
    limited = chat()

    assert answered.json()["provider"] == "ollama"
    for refused in (down, too_long, limited):
        assert refused.json()["provider"] == "ollama"
        assert "response" not in refused.json()
    assert (down.status_code, too_long.status_code, limited.status_code) == (503, 413, 429)


# --- conversations -----------------------------------------------------------------------------


def test_a_conversation_carries_its_earlier_turns(monkeypatch, limits, tmp_path):
    fake = ollama(monkeypatch)

    first = chat("My name is Klara.", conversation_id="demo-1").json()
    second = chat("What is my name?", conversation_id="demo-1").json()

    assert first["context"]["history_messages_used"] == 0
    assert second["context"]["history_messages_used"] == 2
    assert "Learner: My name is Klara." in fake.prompts[1]
    assert fake.prompts[1].endswith("Learner: What is my name?\nAssistant:")
    assert (tmp_path / "chats" / "demo-1.json").exists()
    # A different conversation knows nothing of it.
    chat("What is my name?", conversation_id="demo-2")
    assert "Klara" not in fake.prompts[2]


def test_history_that_does_not_fit_is_dropped_oldest_first_and_counted(monkeypatch, limits):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "700")
    limits(output_reserve_tokens=100)
    fake = ollama(monkeypatch)

    for index in range(4):
        chat(f"turn {index} " + "x" * 600, conversation_id="long")

    last = chat("And now?", conversation_id="long").json()

    assert last["context"]["history_messages_dropped"] > 0
    assert last["context"]["estimated_prompt_tokens"] <= last["context"]["prompt_budget_tokens"]
    assert fake.prompts[-1].endswith("Learner: And now?\nAssistant:")
    assert "turn 0" not in fake.prompts[-1]


def test_the_current_message_is_never_the_part_that_is_cut():
    with pytest.raises(Exception) as caught:
        plan_prompt("x" * 3000, [], num_ctx=1000, reserve=200, recent=8, render=format_transcript)

    assert "budget" in str(caught.value)


def test_one_conversation_answers_one_message_at_a_time(monkeypatch, limits):
    limits(max_concurrent=2)
    held = HeldModel()
    monkeypatch.setattr(ollama_service, "generate", held)

    async def both():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            first = asyncio.create_task(http.post("/local-chat", json={"message": "a", "conversation_id": "c"}))
            await asyncio.sleep(0.1)
            second = await http.post("/local-chat", json={"message": "b", "conversation_id": "c"})
            held.release.set()
            return await first, second

    first, second = asyncio.run(both())

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["reason"] == "conversation_busy"


# --- 10. everything else is where it was ----------------------------------------------------------


def test_the_other_endpoints_are_still_mounted():
    paths = set(app.openapi()["paths"])

    assert {"/local-llm", "/local-llm/health", "/local-chat", "/rag/chat", "/agent/chat"} <= paths


def test_the_rag_endpoint_is_not_rate_limited_by_the_local_chat(monkeypatch, limits):
    """The guard belongs to /local-chat. /rag/chat (Day 28) and the Gemini
    paths keep their own behaviour - here, a refused provider name is still
    the 422 it was."""
    limits(rate_limit=1)
    ollama(monkeypatch)
    chat()

    assert chat().status_code == 429
    assert client.post("/rag/chat", json={"message": "hi", "provider": "openai"}).status_code == 422
