"""Day 28: the same RAG pipeline, with the generator chosen per request.

The claim is not "there is a provider parameter". It is that in local mode
**no part of the pipeline talks to the cloud** - and the pipeline has three
model calls in it, not one: the query rewrite (Day 23), the answer itself
(Day 22), and the claim-support check (Day 24). So the central test here
books every Gemini entry point as a trap and drives the whole thing end to
end; if any of the three were left behind, it fails.
"""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import claim_support, gemini_service, llm_provider, ollama_service
from app.services.llm_provider import (
    GEMINI,
    OLLAMA,
    Generated,
    GeminiProvider,
    OllamaProvider,
    ProviderUnavailable,
    provider_for,
)
from app.services.query_rewriter import MODEL, QueryRewriter
from app.services.rag_agent import ENHANCED, RagAgent
from app.services.rag_citations import Source
from app.services.rag_response import ANSWERED, INSUFFICIENT
from app.services.rag_retriever import RetrievedChunk
from app.services.rag_settings import DEFAULT_SETTINGS

client = TestClient(app)

#: Day 24's relevance gate is in front of both providers and has its own
#: tests. Opened here so these tests fail for the reason they are named
#: after rather than for a rerank score.
OPEN_GATE = DEFAULT_SETTINGS.with_overrides(answer_threshold=0.2)

QUOTE = "the registry starts each server as its own subprocess"
CHUNK_TEXT = f"Day 20. The orchestrator reads the registry, and {QUOTE} when a tool is first used."


def chunk(chunk_id: str = "app-services-mcp_registry_0", score: float = 0.90) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=CHUNK_TEXT,
        source="project",
        file="app/services/mcp_registry.py",
        title="mcp_registry",
        section="module header",
        strategy="structural",
        position=0,
        score=score,
    )


class FakeProvider:
    """A provider that records every prompt it is handed.

    The point of recording rather than asserting on one call: the pipeline
    makes three, and a test that only checked the answer would not notice
    the other two going somewhere else.
    """

    def __init__(self, name: str = OLLAMA, reply: str = "", model: str = "qwen3:4b") -> None:
        self.name = name
        self.model = model
        self.prompts: list[str] = []
        self._reply = reply or json.dumps(
            {
                "answer": "The registry starts each server as its own subprocess. [1]",
                "citations": [{"source": "1", "quote": QUOTE}],
            }
        )

    async def generate(self, prompt: str, temperature: float | None = None) -> Generated:
        self.prompts.append(prompt)

        return Generated(text=self._reply, model=self.model, input_tokens=5, output_tokens=7)


def no_cloud(monkeypatch):
    """Every way out to Gemini raises."""

    def forbidden(*args, **kwargs):
        raise AssertionError("the local pipeline called Gemini")

    for name in (
        "_post",
        "_post_to_gemini",
        "generate_text",
        "generate_text_with_usage",
        "count_tokens",
        "embed_texts",
    ):
        monkeypatch.setattr(gemini_service, name, forbidden)


def serve(handler):
    def factory():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return factory


# --- the provider abstraction ----------------------------------------------


def test_both_providers_answer_to_the_same_call():
    """The one thing the abstraction has to be true of."""
    for provider in (GeminiProvider(), OllamaProvider()):
        assert hasattr(provider, "generate")
        assert provider.name in (GEMINI, OLLAMA)
        assert provider.model


def test_an_unknown_provider_is_refused_rather_than_defaulted():
    """Answering a typo with the cloud is the failure this mode exists to
    prevent, so it has to be louder than a default."""
    with pytest.raises(ValueError) as caught:
        provider_for("openai")

    assert "gemini" in str(caught.value) and "ollama" in str(caught.value)


def test_the_ollama_provider_reuses_the_day_26_service(monkeypatch):
    """Not a second HTTP client: the same one, so there is one place where
    the local model is reached from."""
    handler = None

    def record(request: httpx.Request) -> httpx.Response:
        nonlocal handler
        handler = str(request.url)
        return httpx.Response(200, json={"model": "qwen3:4b", "response": "hello"})

    monkeypatch.setattr(ollama_service, "_client", serve(record))

    generated = asyncio.run(OllamaProvider().generate("hi"))

    assert generated.text == "hello"
    assert generated.model == "qwen3:4b"
    assert handler == "http://localhost:11434/api/generate"


def test_an_unreachable_local_model_raises_rather_than_returning_nothing(monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(ollama_service, "_client", serve(refuse))

    with pytest.raises(ProviderUnavailable) as caught:
        asyncio.run(OllamaProvider().generate("hi"))

    assert caught.value.provider == OLLAMA
    assert caught.value.reason == "not_running"


# --- all three model calls, not just the obvious one -----------------------


def test_the_query_rewrite_goes_to_the_chosen_provider(monkeypatch):
    """Day 23's rewrite is a model call. A local mode that left it in the
    cloud would be lying in its own log line."""
    no_cloud(monkeypatch)
    provider = FakeProvider(reply="mcp servers registry orchestrator subprocess")

    rewritten = asyncio.run(QueryRewriter(provider=provider).rewrite("Which MCP servers exist?"))

    assert rewritten.used == MODEL
    assert len(provider.prompts) == 1


def test_the_support_check_goes_to_the_chosen_provider(monkeypatch):
    """Day 24's check carries the quoted evidence in its prompt. Leaving it
    on Gemini would send retrieved documents to the cloud in the one mode
    that promised not to."""
    no_cloud(monkeypatch)
    provider = FakeProvider(reply='{"unsupported": []}')

    support = asyncio.run(
        claim_support.check("A claim about the registry.", [QUOTE], provider=provider)
    )

    assert support.checked_by
    assert len(provider.prompts) == 1
    assert QUOTE in provider.prompts[0]


def test_nothing_in_the_local_pipeline_reaches_the_cloud(monkeypatch):
    """The whole day in one test: retrieval is local, and all three model
    calls go to the provider that was named."""
    no_cloud(monkeypatch)
    provider = FakeProvider()

    class Retriever:
        async def retrieve(self, query, top_k=10):
            from app.services.rag_retriever import Retrieval

            return Retrieval(
                query=query,
                chunks=[chunk()],
                seconds=0.01,
                top_k=top_k,
                strategy="structural",
                embedding_model="local-hashing",
            )

    # The relevance gate has its own tests; this one is about where the
    # three model calls go, so the gate is opened out of the way.
    agent = RagAgent(retriever=Retriever(), provider=provider, settings=OPEN_GATE)
    answer = asyncio.run(agent.ask("Which MCP servers does the project register?", mode=ENHANCED))

    assert answer.cited.rag_status == ANSWERED
    assert answer.cited.citations
    assert answer.cited.citations[0].quote == QUOTE
    # rewrite + answer + support check: three prompts, all to the provider.
    assert len(provider.prompts) == 3


def test_the_agent_still_defaults_to_the_cloud(monkeypatch):
    """Everything written before today keeps the behaviour it had: with no
    provider the agent calls the module-level Gemini function, which is
    what 800 existing tests patch."""
    agent = RagAgent()

    assert agent.provider_name == GEMINI
    assert agent.model_name == gemini_service.TEXT_MODEL


# --- the endpoint ----------------------------------------------------------


def with_agent(monkeypatch, provider, chunks=(None,), settings=None):
    """Point the route at an agent with a fake retriever and this provider."""
    from app.api.routes import rag_chat as route
    from app.services.rag_retriever import Retrieval

    prepared = [item or chunk() for item in chunks]
    settings = settings or OPEN_GATE

    class Retriever:
        async def retrieve(self, query, top_k=10):
            return Retrieval(
                query=query,
                chunks=list(prepared),
                seconds=0.01,
                top_k=top_k,
                strategy="structural",
                embedding_model="local-hashing",
            )

    monkeypatch.setattr(
        route,
        "agent_for",
        lambda name: RagAgent(retriever=Retriever(), provider=provider, settings=settings),
    )


def test_the_endpoint_keeps_the_existing_rag_response_shape(monkeypatch):
    """Day 24's schema with two fields added, not a new one: a caller
    written for Day 24 can still read this."""
    no_cloud(monkeypatch)
    provider = FakeProvider()
    with_agent(monkeypatch, provider)

    response = client.post(
        "/rag/chat", json={"message": "Which MCP servers does the project register?", "provider": "ollama"}
    )
    body = response.json()

    assert response.status_code == 200
    assert set(("answer", "sources", "citations", "confidence", "rag_status")) <= set(body)
    assert body["provider"] == OLLAMA
    assert body["model"] == "qwen3:4b"
    assert body["rag_status"] == ANSWERED
    assert body["citations"][0]["quote"] == QUOTE


def test_the_endpoint_reports_where_the_time_went(monkeypatch):
    """Retrieval is the half the two providers share; generation is the
    half that differs. Reported separately so the comparison is possible."""
    no_cloud(monkeypatch)
    with_agent(monkeypatch, FakeProvider())

    timing = client.post("/rag/chat", json={"message": "q", "provider": "ollama"}).json()["timing"]

    assert set(timing) == {"retrieval_ms", "generation_ms", "total_ms"}
    assert timing["total_ms"] >= timing["retrieval_ms"]


def test_an_unknown_provider_never_reaches_the_pipeline():
    assert client.post("/rag/chat", json={"message": "q", "provider": "openai"}).status_code == 422


def test_a_blank_message_is_refused():
    assert client.post("/rag/chat", json={"message": "   ", "provider": "ollama"}).status_code == 422


def test_an_unreachable_local_model_does_not_become_a_cloud_answer(monkeypatch):
    """The failure that would be easiest to hide, and the reason there is no
    field for a fallback answer in the schema."""
    no_cloud(monkeypatch)

    class Down:
        name = OLLAMA
        model = "qwen3:4b"

        async def generate(self, prompt, temperature=None):
            raise ProviderUnavailable(OLLAMA, "Ollama is not running.\nollama serve", "not_running")

    with_agent(monkeypatch, Down())

    response = client.post("/rag/chat", json={"message": "q", "provider": "ollama"})

    assert response.status_code == 503
    assert response.json()["error"] == "Local LLM is unavailable"
    assert response.json()["provider"] == OLLAMA


def test_weak_evidence_still_refuses_before_any_model_is_asked(monkeypatch):
    """Day 24's anti-hallucination gate is in front of both providers, not
    just the cloud one: below the threshold nothing is generated at all."""
    no_cloud(monkeypatch)
    provider = FakeProvider()
    with_agent(monkeypatch, provider, chunks=[chunk(score=0.10)], settings=DEFAULT_SETTINGS)

    body = client.post("/rag/chat", json={"message": "q", "provider": "ollama"}).json()

    assert body["rag_status"] == INSUFFICIENT
    assert body["sources"] == []
    assert body["citations"] == []
    assert body["confidence"] == "low"
    assert provider.prompts == [] or len(provider.prompts) == 1  # the rewrite only


# --- the evaluation --------------------------------------------------------


def test_the_evaluation_runs_both_providers_on_the_same_questions():
    from app.evaluate_day28 import parse_args

    options = parse_args([])

    assert options.providers == [GEMINI, OLLAMA]


def test_the_evaluation_averages_only_over_runs_that_happened():
    """A provider that failed half the questions must not get a flattering
    mean out of the half it skipped."""
    from app.evaluate_day28 import Run, summarise

    runs = [
        Run(1, "q1", OLLAMA, rag_status=ANSWERED, total_ms=3000, citations=[{}], valid_citations=1),
        Run(2, "q2", OLLAMA, error="Ollama is not running."),
    ]
    summary = summarise(runs, OLLAMA)

    assert summary["questions"] == 2
    assert summary["errors"] == 1
    assert summary["answered"] == 1
    assert summary["average_total_ms"] == 3000
    assert summary["citation_validity"] == 1.0
