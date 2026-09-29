"""Day 22: the first RAG query, from a question to an answer with sources.

The ten things the assignment asks to prove: the question is embedded, FAISS
is searched, the hits are mapped back to their metadata, top_k is honoured,
the prompt carries the context, both modes work, the sources survive all the
way into the prompt, a question with no answer in the index is handled, and
the control set loads.

Everything runs on a small index built here with the local embedding
backend, so no key and no network are needed; the model is stubbed, because
what is under test is the chain around it, not Gemini's prose.
"""

import asyncio
import json
from pathlib import Path

import pytest

from app.evaluate_rag import (
    QUESTIONS_FILE,
    keywords,
    load_questions,
    mentions_expected,
    metrics,
    retrieved_expected,
    says_unknown,
)
from app.services.chunking import STRUCTURAL, chunk_documents
from app.services.document_loader import Document
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import GeneratedText
from app.services.rag_agent import RagAgent
from app.services.rag_prompt import UNAVAILABLE, plain_prompt, rag_prompt
from app.services.rag_retriever import RAGRetriever, RetrievedChunk
from app.services.vector_index import VectorIndexError, save_index

# The fixtures are written at a realistic length: structural chunking joins
# anything too small to stand on its own, so toy sections of one line each
# would come back as a single chunk per file and the tests would prove
# nothing about sections.
SCHEDULER = """# Scheduler

## Ticking

The digest scheduler ticks once a second, which is cheap because a tick only
asks which tasks are due and goes back to sleep when none of them are. The
shortest interval a task may be created with is ten seconds, and the longest
is a day; both limits are enforced by the schema, so a task that asks for one
second is refused before anything is written to disk rather than quietly
rounded up to something the scheduler can actually manage.

## Restarting

The next run is worked out from the last one rather than from the moment the
scheduler started, so a backend that was down for an hour does one run on the
way back up instead of sixty. Nothing is held in memory between ticks either:
the task file and the run history on disk are the whole state, which is why a
restart continues the count instead of starting it again from zero.
"""

REGISTRY = """# MCP registry

## Routing

Every registered server is asked what it offers, and the routing table is
built from the answers rather than written down anywhere. A tool name alone
is then enough to send a call to the server that offers it, and a server that
changes its tools changes the agent's list by itself the next time discovery
runs, with nothing to keep in step by hand.

## Failures

A server that cannot be reached takes only its own tools with it; the others
keep working, and the call comes back as a failed result rather than as an
exception. A tool that no server offers is reported differently from a server
that is down, because a chain that stops has to be able to say which of the
two happened rather than only that something went wrong.
"""

KANJI = """# Kanji practice

## Stroke order

Stroke order is taught from the top left downwards, and the app shows it as
an animation rather than as a list of rules, because the order of the strokes
is easier to copy than to read. The practice screen keeps the same stroke
data for every character, so a kanji looked up in the dictionary and a kanji
met in an exercise are drawn from one source.
"""


def documents() -> list[Document]:
    return [
        Document("docs", "docs/scheduler.md", "Scheduler", SCHEDULER),
        Document("docs", "docs/registry.md", "MCP registry", REGISTRY),
        Document("docs", "docs/kanji.md", "Kanji practice", KANJI),
    ]


@pytest.fixture
def index_dir(tmp_path) -> Path:
    """A real FAISS index over three small documents, embedded locally."""
    service = EmbeddingService(backend="local")
    chunks = chunk_documents(documents(), STRUCTURAL)
    vectors = asyncio.run(service.embed_all([chunk.text for chunk in chunks]))
    save_index(
        tmp_path / "structural",
        vectors,
        [chunk.as_metadata() for chunk in chunks],
        STRUCTURAL,
        service.name,
    )

    return tmp_path


@pytest.fixture
def retriever(index_dir) -> RAGRetriever:
    return RAGRetriever(strategy=STRUCTURAL, index_dir=index_dir)


def stub_model(monkeypatch, answer: str = "An answer.") -> list[str]:
    """Record what the model was shown, and answer something fixed."""
    from app.services import rag_agent as module

    prompts: list[str] = []

    async def fake(prompt, model=None, temperature=None):
        prompts.append(prompt)
        return GeneratedText(text=answer, input_tokens=100, output_tokens=20)

    monkeypatch.setattr(module, "generate_text_with_usage", fake)

    return prompts


# --- 1. the query is embedded - by whatever embedded the index --------------


def test_the_question_is_embedded_by_the_model_that_built_the_index(retriever):
    assert retriever.embeddings().name == "local-hashing"
    assert retriever.index().embedding_model == "local-hashing"

    vector = asyncio.run(retriever.embeddings().embed("how often does it tick?"))

    assert len(vector) == retriever.index().dimension


def test_a_question_embedded_in_the_wrong_space_is_refused(index_dir):
    """A 4-dimensional question cannot be compared with 768-dimensional
    chunks, and saying so beats returning nonsense."""
    wrong = EmbeddingService(backend="local", dimension=4)
    retriever = RAGRetriever(strategy=STRUCTURAL, index_dir=index_dir, embeddings=wrong)

    with pytest.raises(VectorIndexError) as error:
        asyncio.run(retriever.retrieve("anything"))

    assert "dimensions" in str(error.value)


# --- 2-3. FAISS retrieval, and the metadata behind each hit -----------------


def test_the_search_finds_the_document_the_question_is_about(retriever):
    found = asyncio.run(retriever.retrieve("how often does the scheduler tick?", top_k=3))

    # the nearest chunk is from the document that answers it, and the
    # sentence that answers it is among what came back
    assert found.chunks[0].file == "docs/scheduler.md"
    assert any("once a second" in chunk.text for chunk in found.chunks)
    assert found.chunks[0].score > 0
    assert found.embedding_model == "local-hashing"


def test_every_hit_carries_the_metadata_of_its_chunk(retriever):
    found = asyncio.run(retriever.retrieve("what happens when a server cannot be reached?", top_k=2))

    for chunk in found.chunks:
        assert chunk.chunk_id.startswith("docs-")
        assert chunk.file.startswith("docs/")
        assert chunk.strategy == STRUCTURAL
        assert chunk.position >= 0
        assert chunk.title
        assert chunk.text.strip()


def test_the_hits_are_ordered_by_similarity(retriever):
    found = asyncio.run(retriever.retrieve("routing a tool call to a server", top_k=4))
    scores = [chunk.score for chunk in found.chunks]

    assert scores == sorted(scores, reverse=True)


# --- 4. top_k ---------------------------------------------------------------


@pytest.mark.parametrize("top_k", [1, 2, 5])
def test_top_k_is_what_comes_back(retriever, top_k):
    found = asyncio.run(retriever.retrieve("scheduler", top_k=top_k))

    assert len(found.chunks) == min(top_k, len(retriever.index()))
    assert found.top_k == top_k


def test_top_k_is_not_hardcoded_anywhere_in_the_chain(retriever, monkeypatch):
    stub_model(monkeypatch)
    agent = RagAgent(retriever=retriever, top_k=2)

    assert len(asyncio.run(agent.ask("scheduler")).retrieved_chunks) == 2
    assert len(asyncio.run(agent.ask("scheduler", top_k=3)).retrieved_chunks) == 3


def test_a_meaningless_top_k_is_refused(retriever):
    with pytest.raises(ValueError):
        asyncio.run(retriever.retrieve("scheduler", top_k=0))


# --- 5. the prompt ----------------------------------------------------------


def test_the_prompt_carries_the_question_the_context_and_the_rules(retriever):
    found = asyncio.run(retriever.retrieve("how often does the scheduler tick?", top_k=2))
    prompt = rag_prompt(found.query, found.chunks)

    assert prompt.startswith("Question:\nhow often does the scheduler tick?")
    assert "Relevant context:" in prompt
    assert "[Source: docs/scheduler.md]" in prompt
    # the section is the one the chunker really produced for that text
    assert "[Section: Scheduler" in prompt
    assert "[Score: " in prompt
    assert UNAVAILABLE in prompt
    assert "Sources:" in prompt


def test_the_prompt_says_what_to_do_when_nothing_was_found():
    prompt = rag_prompt("anything", [])

    assert "nothing was found" in prompt
    assert UNAVAILABLE in prompt


def test_the_plain_prompt_has_no_context_at_all():
    prompt = plain_prompt("how often does the scheduler tick?")

    assert "Relevant context" not in prompt
    assert "[Source:" not in prompt
    assert "how often does the scheduler tick?" in prompt


# --- 6-7. the two modes -----------------------------------------------------


def test_rag_off_asks_the_model_alone(retriever, monkeypatch):
    prompts = stub_model(monkeypatch, "From memory.")

    answer = asyncio.run(RagAgent(retriever=retriever).ask("how often does it tick?", use_rag=False))

    assert answer.rag_enabled is False
    assert answer.retrieved_chunks == []
    assert answer.sources == []
    assert answer.retrieval_seconds == 0.0
    assert "[Source:" not in prompts[0]


def test_rag_on_retrieves_first_and_reports_what_it_used(retriever, monkeypatch):
    prompts = stub_model(monkeypatch, "From the documents.")

    answer = asyncio.run(RagAgent(retriever=retriever, top_k=3).ask("how often does it tick?"))

    assert answer.rag_enabled is True
    assert len(answer.retrieved_chunks) == 3
    assert answer.sources
    assert answer.retrieval_seconds > 0
    assert "[Source: docs/scheduler.md]" in prompts[0]


def test_one_agent_answers_the_same_question_both_ways(retriever, monkeypatch):
    prompts = stub_model(monkeypatch)
    agent = RagAgent(retriever=retriever, top_k=2)

    with_rag = asyncio.run(agent.ask("how often does it tick?", use_rag=True))
    without = asyncio.run(agent.ask("how often does it tick?", use_rag=False))

    assert with_rag.question == without.question
    assert with_rag.rag_enabled and not without.rag_enabled
    assert len(prompts[0]) > len(prompts[1])


# --- 8. the sources survive the whole way -----------------------------------


def test_faiss_hit_then_metadata_then_source_then_prompt(retriever, monkeypatch):
    """The correspondence the whole thing rests on, checked end to end."""
    prompts = stub_model(monkeypatch)
    agent = RagAgent(retriever=retriever, top_k=3)

    answer = asyncio.run(agent.ask("what happens when a server cannot be reached?"))

    loaded = retriever.index()

    for chunk in answer.retrieved_chunks:
        # the position reported is the position in the index, and the record
        # at that position is this very chunk
        record = loaded.metadata[chunk.position]
        assert record["chunk_id"] == chunk.chunk_id
        assert record["file"] == chunk.file
        assert record["section"] == chunk.section
        # ...and that file is in the prompt, labelled as its source
        assert f"[Source: {chunk.file}]" in prompts[0]

    # nothing is reported that was not retrieved
    assert set(answer.sources) == {chunk.reference for chunk in answer.retrieved_chunks}
    assert all(source.split(" / ")[0] in prompts[0] for source in answer.sources)


def test_sources_are_listed_once_in_the_order_they_were_found(retriever):
    found = asyncio.run(retriever.retrieve("scheduler ticking and restarting", top_k=4))

    assert found.sources == list(dict.fromkeys(found.sources))
    assert found.sources[0] == found.chunks[0].reference


# --- 9. a question the index cannot answer ----------------------------------


def test_a_question_outside_the_documents_still_retrieves_but_scores_low(retriever, monkeypatch):
    stub_model(monkeypatch, f"The information is {UNAVAILABLE}.")
    agent = RagAgent(retriever=retriever, top_k=3)

    about_documents = asyncio.run(agent.ask("what happens when a server cannot be reached?"))
    about_nothing = asyncio.run(agent.ask("which relational database stores the conversations?"))

    assert about_nothing.retrieved_chunks  # the index always answers with its nearest
    best_known = about_documents.retrieved_chunks[0].score
    best_unknown = about_nothing.retrieved_chunks[0].score
    assert best_unknown < best_known, "an unrelated question should not match as well"
    assert says_unknown(about_nothing.answer)


def test_an_empty_question_is_refused(retriever):
    with pytest.raises(ValueError):
        asyncio.run(RagAgent(retriever=retriever).ask("   "))


# --- 10. the control set ----------------------------------------------------


def test_the_ten_control_questions_load_and_are_about_real_documents():
    questions = load_questions()

    assert len(questions) >= 10
    assert len({question.id for question in questions}) == len(questions)
    assert all(question.question and question.expected for question in questions)
    categories = {question.category for question in questions}
    assert {"factual", "architecture", "mcp", "agent", "code", "configuration"} <= categories
    # one of them is deliberately unanswerable, and says so by expecting no source
    unanswerable = [question for question in questions if not question.expected_sources]
    assert len(unanswerable) == 1
    assert "not available" in unanswerable[0].expected.lower()


def test_every_expected_source_is_a_file_that_is_really_indexed():
    from app.services.document_loader import load_documents

    indexed = {document.file for document in load_documents()}

    for question in load_questions():
        for source in question.expected_sources:
            assert any(source in file for file in indexed), f"{source} is not in the corpus"


def test_a_missing_control_file_is_reported(tmp_path):
    with pytest.raises(SystemExit):
        load_questions(tmp_path / "nothing.json")


# --- the measuring helpers, which must not flatter the results --------------


def test_expected_words_are_counted_not_guessed():
    expected = "The scheduler ticks once a second and the minimum interval is 10 seconds."

    assert mentions_expected("It ticks once a second; the minimum interval is 10 seconds.", expected) > 0.6
    assert mentions_expected("It runs from time to time.", expected) < 0.3
    assert "scheduler" in keywords(expected)
    assert "the" not in keywords(expected)


def test_an_answer_that_admits_it_does_not_know_is_recognised():
    assert says_unknown(f"The information is {UNAVAILABLE}.")
    assert says_unknown("В предоставленном контексте нет информации об этом.")
    assert not says_unknown("The scheduler ticks once a second.")


def test_an_expected_source_counts_only_when_it_was_really_retrieved(retriever, monkeypatch):
    stub_model(monkeypatch)
    answer = asyncio.run(RagAgent(retriever=retriever, top_k=2).ask("how often does it tick?"))

    assert retrieved_expected(answer, ("docs/scheduler.md",)) == ["docs/scheduler.md"]
    assert retrieved_expected(answer, ("docs/nothing.md",)) == []


def test_the_metrics_count_what_they_say_they_count():
    records = [
        {
            "expected_sources": ["a.md"],
            "expected_sources_retrieved": ["a.md"],
            "expected_sources_missing": [],
            "retrieved_chunks": [{}, {}],
            "similarity_scores": [0.8, 0.6],
            "expected_words_in_rag_answer": 0.9,
            "expected_words_in_plain_answer": 0.2,
            "rag_answer_says_unknown": False,
            "plain_answer_says_unknown": False,
            "latency": {
                "retrieval_seconds": 0.1,
                "llm_seconds_with_rag": 2.0,
                "llm_seconds_without_rag": 1.0,
                "total_seconds_with_rag": 2.1,
            },
        },
        {
            "expected_sources": [],
            "expected_sources_retrieved": [],
            "expected_sources_missing": [],
            "retrieved_chunks": [{}],
            "similarity_scores": [0.3],
            "expected_words_in_rag_answer": 0.1,
            "expected_words_in_plain_answer": 0.1,
            "rag_answer_says_unknown": True,
            "plain_answer_says_unknown": False,
            "latency": {
                "retrieval_seconds": 0.1,
                "llm_seconds_with_rag": 1.0,
                "llm_seconds_without_rag": 1.0,
                "total_seconds_with_rag": 1.1,
            },
        },
    ]

    counted = metrics(records, 3.2)

    assert counted["total_questions"] == 2
    assert counted["questions_with_expected_sources"] == 1
    assert counted["expected_source_retrieved"] == 1
    assert counted["average_retrieved_chunks"] == 1.5
    assert counted["average_similarity"] == round((0.8 + 0.6 + 0.3) / 3, 4)
    assert counted["rag_answers_using_expected_words"] == 1
    assert counted["plain_answers_using_expected_words"] == 0
    assert counted["unanswerable_questions"] == 1
    assert counted["unanswerable_handled_by_rag"] == 1
    assert counted["average_llm_seconds_with_rag"] == 1.5


def test_the_control_file_is_in_the_repository():
    assert QUESTIONS_FILE.exists()
    document = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    assert len(document["questions"]) >= 10


# --- the endpoint the app talks to ------------------------------------------


def rag_route(monkeypatch, index_dir, answer_text="An answer."):
    """The real route, over the small test index, with the model stubbed."""
    from fastapi.testclient import TestClient

    from app.api.routes import agent as route_module
    from app.main import app
    from app.services import rag_agent as agent_module

    async def fake(prompt, model=None, temperature=None):
        return GeneratedText(text=answer_text, input_tokens=100, output_tokens=20)

    monkeypatch.setattr(agent_module, "generate_text_with_usage", fake)
    monkeypatch.setattr(
        route_module,
        "rag_agent",
        RagAgent(retriever=RAGRetriever(strategy=STRUCTURAL, index_dir=index_dir)),
    )

    return TestClient(app)


def test_the_endpoint_answers_with_retrieval_and_says_where_from(monkeypatch, index_dir):
    client = rag_route(monkeypatch, index_dir)

    body = client.post(
        "/agent/rag",
        json={"question": "how often does the scheduler tick?", "use_rag": True, "top_k": 3},
    ).json()

    assert body["rag_enabled"] is True
    assert len(body["retrieved_chunks"]) == 3
    assert body["sources"]
    assert body["retrieved_chunks"][0]["file"].startswith("docs/")
    assert body["retrieved_chunks"][0]["score"] > 0
    assert "text" not in body["retrieved_chunks"][0], "the screen gets references, not documents"
    assert body["top_k"] == 3
    assert body["embedding_model"] == "local-hashing"


def test_the_endpoint_answers_without_retrieval_when_asked(monkeypatch, index_dir):
    client = rag_route(monkeypatch, index_dir)

    body = client.post(
        "/agent/rag", json={"question": "how often does the scheduler tick?", "use_rag": False}
    ).json()

    assert body["rag_enabled"] is False
    assert body["retrieved_chunks"] == []
    assert body["sources"] == []
    assert body["retrieval_seconds"] == 0.0


def test_the_same_question_can_be_asked_both_ways(monkeypatch, index_dir):
    client = rag_route(monkeypatch, index_dir)
    question = "what happens when a server cannot be reached?"

    with_rag = client.post("/agent/rag", json={"question": question, "use_rag": True}).json()
    without = client.post("/agent/rag", json={"question": question, "use_rag": False}).json()

    assert with_rag["answer"] == without["answer"]  # the model is stubbed; the modes are not
    assert with_rag["sources"] and not without["sources"]


def test_a_missing_index_is_a_service_error_not_a_crash(monkeypatch, tmp_path):
    client = rag_route(monkeypatch, tmp_path / "empty")

    response = client.post("/agent/rag", json={"question": "anything"})

    assert response.status_code == 503
    assert "index is not available" in response.json()["detail"]


def test_top_k_is_validated_at_the_edge(monkeypatch, index_dir):
    client = rag_route(monkeypatch, index_dir)

    assert client.post("/agent/rag", json={"question": "x", "top_k": 0}).status_code == 422
    assert client.post("/agent/rag", json={"question": "x", "top_k": 99}).status_code == 422
