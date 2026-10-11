"""Day 29: tuning the local model without moving anything else.

What has to stay true while the knobs are added:

- the original configuration is the Day 26-28 request, byte for byte, and
  it is still what runs unless something says otherwise;
- a profile's options reach Ollama under Ollama's own names, and only the
  ones that are set;
- the Day 29 prompt changes how the answer is asked for, not how it is
  judged - citation validation and the relevance gate treat it the same;
- local mode still never falls back to Gemini, and Gemini mode is untouched;
- the evaluation compares profiles on the same context and writes down
  what it measured, including the failures.
"""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import evaluate_day29
from app.main import app
from app.services import gemini_service, ollama_profiles, ollama_service
from app.services.gemini_service import GeneratedText
from app.services.llm_provider import (
    OLLAMA,
    GeminiProvider,
    Generated,
    OllamaProvider,
    ProviderUnavailable,
    provider_for,
)
from app.services.ollama_profiles import (
    BASELINE,
    OPTIMIZED,
    OPTIMIZED_PARAMS,
    ORIGINAL,
    active_profile,
    profile_for,
)
from app.services.ollama_service import OllamaOptions, OllamaUnavailable, split_thinking
from app.services.rag_agent import ENHANCED, RagAgent
from app.services.rag_prompt import (
    DONT_KNOW,
    PROMPT_V1,
    PROMPT_V2,
    cited_prompt,
    cited_prompt_for,
    cited_prompt_v2,
)
from app.services.rag_response import ANSWERED, INSUFFICIENT
from app.services.rag_retriever import Retrieval, RetrievedChunk
from app.services.rag_settings import DEFAULT_SETTINGS

client = TestClient(app)
OPEN_GATE = DEFAULT_SETTINGS.with_overrides(answer_threshold=0.2, query_rewrite=False)

QUOTE = "A server that cannot be reached takes only its own tools with it."
CHUNK_TEXT = f"Discovery asks every registered server with list_tools(). {QUOTE} The rest keep working."


def chunk(score: float = 0.9, text: str = CHUNK_TEXT) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="README_28",
        text=text,
        source="project",
        file="README.md",
        title="README",
        section="The tool pipeline and its servers (Days 19-20)",
        strategy="structural",
        position=28,
        score=score,
    )


class Retriever:
    def __init__(self, score: float = 0.9) -> None:
        self._score = score

    async def retrieve(self, query, top_k=10):
        return Retrieval(
            query=query,
            chunks=(chunk(self._score),),
            seconds=0.01,
            top_k=top_k,
            strategy="structural",
            embedding_model="local-hashing",
        )


class Recorder:
    """A provider that keeps every prompt and replies with a fixed text."""

    def __init__(self, reply: str, prompt_version: str | None = None, name: str = OLLAMA) -> None:
        self.name = name
        self.model = "qwen3:4b"
        self.prompts: list[str] = []
        self._reply = reply

        if prompt_version is not None:
            self.prompt_version = prompt_version

    async def generate(self, prompt, temperature=None):
        self.prompts.append(prompt)

        return Generated(text=self._reply, model=self.model, input_tokens=3, output_tokens=4)


def reply(*quotes: str, answer: str = "No - it takes only its own tools with it.") -> str:
    return json.dumps(
        {"answer": answer, "citations": [{"source": 1, "quote": quote} for quote in quotes]}
    )


def serve(handler):
    def factory():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return factory


def ollama_reply(text: str = "hello", **extra) -> dict:
    return {"model": "qwen3:4b", "response": text, "done": True, **extra}


def capture(monkeypatch, body: dict | None = None):
    """Point the one Ollama client at a recorder; return what it saw."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": json.loads(request.content or b"{}")})
        return httpx.Response(200, json=body or ollama_reply())

    monkeypatch.setattr(ollama_service, "_client", serve(handler))

    return seen


def no_cloud(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the local path called Gemini")

    for name in ("_post", "_post_to_gemini", "generate_text", "generate_text_with_usage", "count_tokens"):
        monkeypatch.setattr(gemini_service, name, forbidden)


# --- configuration selection -------------------------------------------------


def test_the_four_profiles_carry_the_values_the_experiment_names():
    assert ORIGINAL.options.request_fields() == {}
    assert BASELINE.options.as_dict() == {
        "temperature": 0.7,
        "num_predict": 1024,
        "num_ctx": 4096,
        "think": None,
    }
    assert OPTIMIZED.options.as_dict() == {
        "temperature": 0.2,
        "num_predict": 512,
        "num_ctx": 4096,
        "think": False,
    }
    # The parameters-only profile differs from the optimised one in the
    # prompt and nothing else - that is what lets the two effects be told apart.
    assert OPTIMIZED_PARAMS.options == OPTIMIZED.options
    assert (OPTIMIZED_PARAMS.prompt_version, OPTIMIZED.prompt_version) == (PROMPT_V1, PROMPT_V2)
    assert ORIGINAL.prompt_version == BASELINE.prompt_version == PROMPT_V1


def test_the_service_keeps_running_the_original_profile_unless_told_otherwise():
    assert active_profile({}) is ORIGINAL
    assert active_profile({"OLLAMA_PROFILE": "optimized"}) is OPTIMIZED


def test_an_unknown_profile_is_refused_not_defaulted():
    with pytest.raises(ValueError, match="unknown Ollama profile"):
        profile_for("fastest")

    with pytest.raises(ValueError):
        active_profile({"OLLAMA_PROFILE": "fastest"})


def test_single_parameters_can_be_overridden_from_the_environment():
    profile = active_profile(
        {
            "OLLAMA_PROFILE": "optimized",
            "OLLAMA_TEMPERATURE": "0.1",
            "OLLAMA_NUM_PREDICT": "256",
            "OLLAMA_NUM_CTX": "8192",
            "OLLAMA_THINK": "true",
        }
    )

    assert profile.name == "optimized+env"
    assert profile.options == OllamaOptions(temperature=0.1, num_predict=256, num_ctx=8192, think=True)
    assert profile.prompt_version == PROMPT_V2


def test_a_malformed_override_is_an_error_rather_than_ignored():
    with pytest.raises(ValueError, match="OLLAMA_NUM_CTX"):
        active_profile({"OLLAMA_NUM_CTX": "large"})

    with pytest.raises(ValueError, match="OLLAMA_THINK"):
        active_profile({"OLLAMA_THINK": "maybe"})


@pytest.mark.parametrize(
    "options",
    [
        {"temperature": 2.5},
        {"num_predict": 0},
        {"num_ctx": 64},
    ],
)
def test_impossible_options_are_rejected(options):
    with pytest.raises(ValueError):
        OllamaOptions(**options)


def test_no_token_limit_is_a_legal_choice():
    assert OllamaOptions(num_predict=-1).request_fields() == {"options": {"num_predict": -1}}


# --- the request Ollama receives ---------------------------------------------


def test_without_options_the_request_is_the_day_26_one_byte_for_byte(monkeypatch):
    seen = capture(monkeypatch)
    asyncio.run(ollama_service.generate("What is MCP?"))

    assert seen[0]["body"] == {"model": "qwen3:4b", "prompt": "What is MCP?", "stream": False}


def test_options_go_out_under_ollamas_own_names(monkeypatch):
    seen = capture(monkeypatch)
    asyncio.run(ollama_service.generate("q", options=OPTIMIZED.options))
    body = seen[0]["body"]

    assert body["options"] == {"temperature": 0.2, "num_predict": 512, "num_ctx": 4096}
    # think is a top-level field of /api/generate, not a sampling option.
    assert body["think"] is False
    assert "think" not in body["options"]


def test_only_the_options_that_are_set_are_sent(monkeypatch):
    seen = capture(monkeypatch)
    asyncio.run(ollama_service.generate("q", options=OllamaOptions(num_ctx=8192)))

    assert seen[0]["body"]["options"] == {"num_ctx": 8192}
    assert "think" not in seen[0]["body"]


def test_the_provider_sends_its_profile_and_nothing_for_the_original(monkeypatch):
    seen = capture(monkeypatch)
    asyncio.run(OllamaProvider(profile=OPTIMIZED).generate("q"))
    asyncio.run(OllamaProvider(profile=ORIGINAL).generate("q"))

    assert seen[0]["url"] == seen[1]["url"] == "http://localhost:11434/api/generate"
    assert seen[0]["body"]["options"]["temperature"] == 0.2
    assert set(seen[1]["body"]) == {"model", "prompt", "stream"}


def test_the_provider_reads_the_profile_from_the_environment(monkeypatch):
    seen = capture(monkeypatch)
    monkeypatch.setenv("OLLAMA_PROFILE", "baseline")
    provider = provider_for("ollama")
    asyncio.run(provider.generate("q"))

    assert provider.profile is BASELINE
    assert seen[0]["body"]["options"] == {"temperature": 0.7, "num_predict": 1024, "num_ctx": 4096}


def test_a_per_call_temperature_does_not_override_the_profile(monkeypatch):
    """The support check asks for temperature 0. The profile measured is
    the profile that runs, so the profile wins on every call."""
    seen = capture(monkeypatch)
    asyncio.run(OllamaProvider(profile=OPTIMIZED).generate("q", temperature=0.0))

    assert seen[0]["body"]["options"]["temperature"] == 0.2


def test_what_ollama_measured_comes_back_with_the_answer(monkeypatch):
    capture(
        monkeypatch,
        ollama_reply(
            "fine",
            done_reason="stop",
            eval_count=120,
            eval_duration=2_000_000_000,
            prompt_eval_count=900,
            prompt_eval_duration=300_000_000,
            load_duration=1_500_000_000,
            total_duration=3_900_000_000,
        ),
    )
    answer = asyncio.run(ollama_service.generate("q", options=OPTIMIZED.options))

    assert (answer.load_ms, answer.prompt_eval_ms, answer.eval_ms, answer.ollama_total_ms) == (
        1500,
        300,
        2000,
        3900,
    )
    assert answer.tokens_per_second == 60.0
    assert answer.done_reason == "stop"
    assert answer.metrics()["options"]["num_predict"] == 512

    generated = asyncio.run(OllamaProvider(profile=OPTIMIZED).generate("q"))
    assert generated.meta["tokens_per_second"] == 60.0


def test_a_model_cut_off_by_num_predict_says_so(monkeypatch):
    capture(monkeypatch, ollama_reply("", thinking="still reasoning", done_reason="length", eval_count=512))

    with pytest.raises(OllamaUnavailable) as caught:
        asyncio.run(ollama_service.generate("q", options=OllamaOptions(num_predict=512)))

    assert caught.value.reason == "truncated"
    assert "num_predict=512" in caught.value.message

    with pytest.raises(ProviderUnavailable) as through_provider:
        asyncio.run(OllamaProvider(profile=BASELINE).generate("q"))

    assert through_provider.value.reason == "truncated"


def test_unfinished_thinking_is_never_shown_as_the_answer():
    assert split_thinking("<think>first I will look at") == ("first I will look at", "")
    assert split_thinking("<think>done</think>Four servers.") == ("done", "Four servers.")
    assert split_thinking("Just an answer.") == ("", "Just an answer.")


# --- prompt selection ----------------------------------------------------------


def test_version_one_is_the_day_24_prompt_unchanged():
    chunks = [chunk()]

    assert cited_prompt_for(PROMPT_V1, "why?", chunks) == cited_prompt("why?", chunks)


def test_version_two_keeps_the_contract_and_puts_the_rules_last():
    prompt = cited_prompt_v2("Does the agent lose its tools?", [chunk()])

    # Same contract: numbered extracts, the same refusal sentence, one JSON object.
    assert "[1] file: README.md" in prompt
    assert DONT_KNOW in prompt
    assert '{"answer": "...", "citations": [{"source": 1, "quote": "..."}]}' in prompt
    # The order a small model follows: extracts, then the question, then rules.
    assert prompt.index("[1] file:") < prompt.index("Does the agent lose") < prompt.index("Rules:")
    assert "Never invent a class, function, file, endpoint" in prompt


def test_an_unknown_prompt_version_is_refused():
    with pytest.raises(ValueError, match="unknown prompt version"):
        cited_prompt_for("v3", "q", [chunk()])

    with pytest.raises(ValueError):
        RagAgent(retriever=Retriever(), provider=Recorder(reply()), prompt_version="v3")


def test_the_agent_hands_the_model_its_profiles_prompt():
    provider = Recorder(reply(QUOTE), prompt_version=PROMPT_V2)
    agent = RagAgent(retriever=Retriever(), provider=provider, settings=OPEN_GATE, check_support=False)
    answer = asyncio.run(agent.ask("Does the agent lose its tools?", mode=ENHANCED))

    assert agent.prompt_version == PROMPT_V2
    assert provider.prompts[0].startswith("You are a technical assistant")
    assert answer.cited.rag_status == ANSWERED


def test_an_explicit_prompt_version_wins_over_the_providers():
    provider = Recorder(reply(QUOTE), prompt_version=PROMPT_V2)
    agent = RagAgent(
        retriever=Retriever(), provider=provider, settings=OPEN_GATE, check_support=False, prompt_version=PROMPT_V1
    )
    asyncio.run(agent.ask("Does the agent lose its tools?", mode=ENHANCED))

    assert provider.prompts[0].startswith("Question:")


# --- unavailable Ollama ----------------------------------------------------------


def test_an_optimised_profile_with_ollama_down_still_fails_loudly(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(ollama_service, "_client", serve(refuse))

    with pytest.raises(ProviderUnavailable) as caught:
        asyncio.run(OllamaProvider(profile=OPTIMIZED).generate("q"))

    assert caught.value.reason == "not_running"


def test_the_endpoint_refuses_rather_than_falling_back_under_any_profile(monkeypatch):
    from app.api.routes import rag_chat as route

    def refuse(request):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(ollama_service, "_client", serve(refuse))
    monkeypatch.setenv("OLLAMA_PROFILE", "optimized")
    no_cloud(monkeypatch)
    monkeypatch.setattr(
        route,
        "agent_for",
        lambda name: RagAgent(retriever=Retriever(), provider=provider_for(name), settings=OPEN_GATE),
    )
    response = client.post("/rag/chat", json={"message": "Does the agent lose its tools?", "provider": "ollama"})

    assert response.status_code == 503
    assert response.json()["provider"] == "ollama"
    assert "answer" not in response.json()


# --- insufficient context ----------------------------------------------------------


def test_below_the_gate_no_profile_is_ever_asked():
    provider = Recorder(reply(QUOTE), prompt_version=PROMPT_V2)
    strict = DEFAULT_SETTINGS.with_overrides(answer_threshold=0.99, query_rewrite=False)
    agent = RagAgent(retriever=Retriever(), provider=provider, settings=strict)
    answer = asyncio.run(agent.ask("Which relational database is used?", mode=ENHANCED))

    assert answer.cited.rag_status == INSUFFICIENT
    assert provider.prompts == []


def test_a_model_that_refuses_keeps_the_contract_sentence():
    provider = Recorder(json.dumps({"answer": DONT_KNOW, "citations": []}), prompt_version=PROMPT_V2)
    agent = RagAgent(retriever=Retriever(), provider=provider, settings=OPEN_GATE, check_support=False)
    answer = asyncio.run(agent.ask("How often is the key rotated?", mode=ENHANCED))

    assert answer.answer == DONT_KNOW
    assert answer.cited.citations == []


# --- citation validation is not bypassed ---------------------------------------------


def test_the_new_prompt_does_not_get_an_invented_quote_past_validation():
    provider = Recorder(
        reply(QUOTE, "The client retries three times with exponential backoff."),
        prompt_version=PROMPT_V2,
    )
    agent = RagAgent(retriever=Retriever(), provider=provider, settings=OPEN_GATE, check_support=False)
    answer = asyncio.run(agent.ask("Does the agent lose its tools?", mode=ENHANCED))

    assert [citation.quote for citation in answer.cited.citations] == [QUOTE]
    assert len(answer.cited.rejected_citations) == 1


# --- Gemini is untouched ---------------------------------------------------------------


def test_gemini_has_no_profile_and_keeps_the_day_24_prompt(monkeypatch):
    monkeypatch.setenv("OLLAMA_PROFILE", "optimized")
    gemini = provider_for("gemini")

    assert isinstance(gemini, GeminiProvider)
    assert not hasattr(gemini, "profile")

    seen: list[str] = []

    async def fake(prompt, **kwargs):
        seen.append(prompt)
        return GeneratedText(reply(QUOTE), 3, 4)

    monkeypatch.setattr("app.services.rag_agent.generate_text_with_usage", fake)
    agent = RagAgent(retriever=Retriever(), settings=OPEN_GATE, check_support=False)
    asyncio.run(agent.ask("Does the agent lose its tools?", mode=ENHANCED))

    assert agent.prompt_version == PROMPT_V1
    assert agent.profile_name == ""
    assert seen[0] == cited_prompt("Does the agent lose its tools?", [chunk()])


# --- the evaluation's own judgement -----------------------------------------------------


def _question(qid: int):
    questions, _ = evaluate_day29.load_questions()

    return next(question for question in questions if question.id == qid)


def test_the_question_set_loads_and_covers_every_category():
    questions, bench = evaluate_day29.load_questions()
    categories = {question.category for question in questions}

    assert len(questions) >= 10
    assert {"architecture", "mcp", "memory", "rag", "insufficient"} <= categories
    assert bench in {question.id for question in questions}


def test_the_rubric_scores_facts_not_wording():
    question = _question(3)

    assert evaluate_day29.score(question, "No. A server that is down takes only its own tools with it.")["score"] == 1.0
    assert evaluate_day29.score(question, "The agent keeps running.")["score"] == 0.0


def test_an_unanswerable_question_rewards_a_refusal_and_punishes_a_number():
    question = _question(13)

    assert evaluate_day29.score(question, DONT_KNOW)["correct"] is True
    assert evaluate_day29.score(question, "The documents do not describe any retry.")["correct"] is True
    invented = evaluate_day29.score(question, "It retries three times with exponential backoff.")
    assert invented["correct"] is False
    assert invented["forbidden"]
    assert evaluate_day29.score(question, "", INSUFFICIENT)["correct"] is True


def test_the_day_28_format_failures_are_recognised():
    leaked = '{\n  "answer": "The chunk_id is built by file_slug.",\n  "citations": [{"source": 1}]\n}'
    escaped = 'Fields: saved_at.\n\n[{"source": 1, "quote": "{\\"saved_at\\": _stamp(moment)}"}]'

    assert evaluate_day29.format_problem(leaked)
    assert evaluate_day29.format_problem(escaped)
    assert not evaluate_day29.format_problem("Four servers: japanese-data, processing, storage and jlpt-vocab.")


def test_names_the_context_never_mentioned_are_listed():
    found = evaluate_day29.unsupported_identifiers(
        "It calls `retry_server()` in mcp_retry.py, then list_tools().",
        CHUNK_TEXT,
        "Does it retry?",
    )

    assert "retry_server" in found
    assert "mcp_retry.py" in found
    assert "list_tools" not in found


# --- the evaluation, end to end, against a stand-in Ollama -------------------------------


class FakeOllama:
    """Just enough of Ollama's API, with a model that can be unloaded."""

    def __init__(self, installed=("qwen3:4b",)) -> None:
        self.installed = list(installed)
        self.loaded: set[str] = set()
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.requests.append((path, body))

        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n, "size": 2_600_000_000} for n in self.installed]})

        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.0-test"})

        if path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "details": {"family": "qwen3", "parameter_size": "4.0B", "quantization_level": "Q4_K_M"},
                    "model_info": {"qwen3.context_length": 262144, "general.finetune": "Thinking-2507"},
                    "parameters": "temperature 0.6",
                    "capabilities": ["completion", "thinking"],
                },
            )

        if path == "/api/ps":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": n, "model": n, "size": 3_000_000_000, "size_vram": 3_000_000_000, "context_length": 4096}
                        for n in sorted(self.loaded)
                    ]
                },
            )

        if path == "/api/generate":
            model = body["model"]

            if body.get("keep_alive") == 0 and "prompt" not in body:
                self.loaded.discard(model)
                return httpx.Response(200, json={"model": model, "response": "", "done": True, "done_reason": "unload"})

            cold = model not in self.loaded
            self.loaded.add(model)
            prompt = body["prompt"]

            if prompt.startswith("You are checking whether statements"):
                text = json.dumps({"unsupported": []})
            elif "relational" in prompt or "rotated" in prompt or "retry" in prompt:
                text = json.dumps({"answer": DONT_KNOW, "citations": []})
            else:
                text = reply(QUOTE)

            return httpx.Response(
                200,
                json={
                    "model": model,
                    "response": text,
                    "done": True,
                    "done_reason": "stop",
                    "eval_count": 80,
                    "eval_duration": 1_000_000_000,
                    "prompt_eval_count": 600,
                    "prompt_eval_duration": 200_000_000,
                    "load_duration": 2_000_000_000 if cold else 10_000_000,
                    "total_duration": 3_300_000_000 if cold else 1_300_000_000,
                },
            )

        return httpx.Response(404, json={"error": "not here"})


def test_the_evaluation_runs_every_profile_on_the_same_context(monkeypatch, tmp_path):
    fake = FakeOllama()
    monkeypatch.setattr(ollama_service, "_client", serve(fake))
    monkeypatch.setattr(evaluate_day29, "RAGRetriever", lambda strategy: Retriever())
    monkeypatch.setattr(evaluate_day29, "RESULTS_FILE", tmp_path / "day29_results.json")
    monkeypatch.setattr(evaluate_day29, "SUMMARY_FILE", tmp_path / "day29_summary.txt")
    no_cloud(monkeypatch)

    options = evaluate_day29.parse_args(
        ["--profiles", "original", "optimized", "--variants", "qwen3:4b-instruct-2507-q8_0", "--only", "3", "13", "--repeats", "2"]
    )
    assert asyncio.run(evaluate_day29.run(options)) == 0

    document = json.loads((tmp_path / "day29_results.json").read_text())

    # Profiles that ran, and the variant that was not installed - skipped, not pulled.
    assert set(document["profiles"]) == {"original", "optimized"}
    assert document["skipped"][0]["model"] == "qwen3:4b-instruct-2507-q8_0"
    assert not any(path == "/api/pull" for path, _ in fake.requests)

    # Same context for every profile.
    by_question: dict[int, set] = {}
    for run in document["results"]:
        by_question.setdefault(run["question_id"], set()).add(tuple(run["retrieved_source_ids"]))
    assert all(len(ids) == 1 for ids in by_question.values())

    # Every field the assignment asks for, on every run.
    required = {
        "question", "profile", "model", "temperature", "num_predict", "num_ctx", "prompt_version",
        "answer", "retrieved_source_ids", "citations_valid", "correctness", "retrieval_ms",
        "generation_ms", "total_ms", "generated_tokens", "tokens_per_second", "error",
    }
    for run in document["results"]:
        assert required <= set(run), required - set(run)

    # The optimised profile's options really went out; the original's did not.
    sent = [body for path, body in fake.requests if path == "/api/generate" and "prompt" in body]
    assert any(body.get("options", {}).get("num_predict") == 512 and body.get("think") is False for body in sent)
    assert any(set(body) == {"model", "prompt", "stream"} for body in sent)

    summary = document["summary"]["optimized"]
    assert summary["bench"]["cold"]["load_ms"] == 2000
    assert summary["bench"]["warm_median_tokens_per_second"] == 80.0
    assert summary["bench"]["memory"]["ollama_ps_size_mb"] is not None
    assert summary["insufficient_handled"] == 1
    assert summary["correctness_mean"] == 1.0
    assert (tmp_path / "day29_summary.txt").read_text().startswith("=" * 10)
