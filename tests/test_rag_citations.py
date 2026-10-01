"""Day 24: citations, sources and the refusal to answer without evidence.

The ten things the assignment asks to prove, plus the two it spells out
separately: a quote that is not in its chunk must be rejected, and a best
score below the answering threshold must stop the model being called at all.

Everything runs on a real FAISS index built here with the local embedding
backend - no key, no network - and the model is stubbed, because what is
under test is what the backend does with a reply, not Gemini's prose.
"""

import asyncio
import json
from pathlib import Path

import pytest

from app.services import claim_support
from app.services.chunking import STRUCTURAL, chunk_documents
from app.services.document_loader import Document
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import GeneratedText
from app.services.rag_agent import BASELINE, ENHANCED, OFF, RagAgent
from app.services.rag_citations import (
    MAX_QUOTE_CHARS,
    Citation,
    Source,
    flatten,
    parse_reply,
    shorten,
    validate,
)
from app.services.rag_enhanced import EnhancedRetriever
from app.services.rag_prompt import DONT_KNOW, cited_prompt, numbered_context
from app.services.rag_response import (
    ANSWERED,
    DISABLED,
    HIGH,
    INSUFFICIENT,
    INSUFFICIENT_ANSWER,
    LOW,
    MEDIUM,
    NOT_CHECKED,
    SUPPORTED,
    UNSUPPORTED,
    confidence_of,
)
from app.services.rag_retriever import RAGRetriever, RetrievedChunk
from app.services.rag_settings import RagSettings
from app.services.vector_index import save_index

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

KANJI = """# Kanji practice

## Stroke order

Stroke order is taught from the top left downwards, and the app shows it as
an animation rather than as a list of rules, because the order of the strokes
is easier to copy than to read. The practice screen keeps the same stroke
data for every character, so a kanji looked up in the dictionary and a kanji
met in an exercise are drawn from one source.

## Review queue

The review queue is rebuilt every morning from what was answered wrongly the
day before, and a character that has been answered correctly three times in a
row leaves the queue until the month is out. Nothing about the queue is stored
on the server: it is worked out again from the answer history each time.
"""


def documents() -> list[Document]:
    return [
        Document("docs", "docs/registry.md", "MCP registry", REGISTRY),
        Document("docs", "docs/scheduler.md", "Scheduler", SCHEDULER),
        Document("docs", "docs/kanji.md", "Kanji practice", KANJI),
    ]


@pytest.fixture
def index_dir(tmp_path) -> Path:
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


QUESTION = "What happens when one MCP server cannot be reached?"
# The local hashing backend scores much lower than a real embedding model:
# on this index the question below reranks to about 0.24, where a Gemini
# embedding would be past 0.70. The thresholds here are set to the band this
# index actually produces, because what is under test is the behaviour at a
# threshold, not the number.
ANSWERING = 0.15


def settings(**changes) -> RagSettings:
    base = {
        "retrieval_top_k": 6,
        "similarity_threshold": 0.0,
        "final_top_k": 3,
        "query_rewrite": False,
        "answer_threshold": ANSWERING,
    }
    base.update(changes)

    return RagSettings(**base)


def chunk(text: str, chunk_id: str = "c1", file: str = "a.py", section: str = "s", score: float = 0.8):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source="backend",
        file=file,
        title="T",
        section=section,
        strategy=STRUCTURAL,
        position=0,
        score=score,
    )


def reply(answer: str, citations: list[dict]) -> str:
    return json.dumps({"answer": answer, "citations": citations})


def stub_model(monkeypatch, text: str) -> list[str]:
    """Record what the model was shown, and answer something fixed."""
    from app.services import rag_agent as module

    prompts: list[str] = []

    async def fake(prompt, model=None, temperature=None):
        prompts.append(prompt)
        return GeneratedText(text=text, input_tokens=100, output_tokens=20)

    monkeypatch.setattr(module, "generate_text_with_usage", fake)

    return prompts


def no_support_model(monkeypatch) -> None:
    """Keep the support check on words, so a test does not need a model."""

    async def fake(prompt, temperature=None, model=None):
        raise RuntimeError("no model in tests")

    monkeypatch.setattr(claim_support, "generate_text", fake)


def agent_for(retriever: RAGRetriever, config: RagSettings) -> RagAgent:
    return RagAgent(
        retriever=retriever,
        settings=config,
        enhanced=EnhancedRetriever(retriever=retriever, settings=config),
    )


# --- 1. sources -------------------------------------------------------------


def test_a_source_carries_the_four_fields_the_schema_asks_for():
    source = Source.of(chunk("text", chunk_id="agent_15", file="agent.py", section="MCP"))

    assert source.as_dict() == {
        "source": "backend",
        "file": "agent.py",
        "section": "MCP",
        "chunk_id": "agent_15",
    }


def test_sources_are_built_from_the_cited_chunks_not_from_the_model():
    first = chunk("The registry asks every server what it offers.", chunk_id="reg_0")
    second = chunk("A server that cannot be reached takes its tools with it.", chunk_id="reg_1")
    checked = validate(
        reply("It asks.", [{"source": 1, "quote": "The registry asks every server what it offers."}]),
        [first, second],
    )

    assert [source.chunk_id for source in checked.sources] == ["reg_0"]
    assert "reg_1" not in {source.chunk_id for source in checked.sources}


def test_a_source_that_was_not_retrieved_cannot_appear(monkeypatch, retriever):
    no_support_model(monkeypatch)
    stub_model(
        monkeypatch,
        reply("Invented.", [{"source": 1, "chunk_id": "somewhere-else_9", "quote": "x" * 40}]),
    )
    answer = asyncio.run(agent_for(retriever, settings()).ask(QUESTION, mode=ENHANCED))

    for source in answer.cited.sources:
        assert source.chunk_id in {item.chunk_id for item in answer.enhanced.chunks}


# --- 2. citations -----------------------------------------------------------


def test_a_citation_carries_the_four_fields_the_schema_asks_for():
    citation = Citation(source="agent.py", section="MCP", chunk_id="agent_15", quote="a quote")

    assert set(citation.as_dict()) == {"source", "section", "chunk_id", "quote"}


def test_a_citation_is_taken_from_the_chunk_it_points_at():
    source = chunk("The registry asks every server what it offers.", chunk_id="reg_0")
    checked = validate(
        reply("ok", [{"source": 1, "quote": "asks every server what it offers"}]), [source]
    )

    assert len(checked.citations) == 1
    assert checked.citations[0].chunk_id == "reg_0"
    assert checked.citations[0].quote in source.text


def test_the_model_cites_by_number_and_the_backend_resolves_it():
    first = chunk("First chunk text that is long enough.", chunk_id="one")
    second = chunk("Second chunk text that is long enough.", chunk_id="two")
    checked = validate(
        reply("ok", [{"source": 2, "quote": "Second chunk text that is long enough."}]),
        [first, second],
    )

    assert checked.citations[0].chunk_id == "two"


def test_a_bracketed_marker_is_understood_too():
    source = chunk("The registry asks every server what it offers.", chunk_id="reg_0")
    checked = validate(reply("ok", [{"source": "[1]", "quote": "asks every server"}]), [source])

    assert len(checked.citations) == 1


# --- 3. quote validation ----------------------------------------------------


def test_a_quote_that_is_in_the_chunk_is_kept():
    source = chunk("A server that cannot be reached takes only its own tools with it.")
    checked = validate(reply("ok", [{"source": 1, "quote": "takes only its own tools with it"}]), [source])

    assert checked.passed
    assert len(checked.citations) == 1


def test_only_whitespace_is_forgiven():
    source = chunk("A server that cannot\nbe reached takes only its own tools.")
    checked = validate(
        reply("ok", [{"source": 1, "quote": "A server that cannot be reached takes only its own tools."}]),
        [source],
    )

    assert len(checked.citations) == 1, "a reflowed line break is not an invention"


def test_a_changed_word_is_not_forgiven():
    source = chunk("A server that cannot be reached takes only its own tools with it.")
    checked = validate(
        reply("ok", [{"source": 1, "quote": "A server that cannot be reached takes all of its tools."}]),
        [source],
    )

    assert checked.citations == ()
    assert checked.rejected[0].reason.startswith("quote is not a fragment")


def test_a_long_quote_is_cut_to_a_few_sentences():
    long_quote = " ".join(f"Sentence number {number} is here." for number in range(10))
    assert len(shorten(long_quote).split(". ")) <= 3
    assert len(shorten("x" * 900)) <= MAX_QUOTE_CHARS


# --- 4. an invalid citation is rejected -------------------------------------


def test_a_quote_not_in_the_chunk_is_rejected():
    """The check the assignment spells out: quote not in chunk -> no citation."""
    source = chunk("The digest scheduler ticks once a second.")
    checked = validate(
        reply("It ticks every 15 seconds.", [{"source": 1, "quote": "ticks once every 15 seconds"}]),
        [source],
    )

    assert checked.citations == ()
    assert len(checked.rejected) == 1
    assert checked.sources == (), "a rejected citation leaves no source behind"


def test_a_citation_pointing_at_nothing_is_rejected():
    source = chunk("Some text that is long enough to quote from.")
    checked = validate(reply("ok", [{"source": 9, "quote": "Some text that is long enough"}]), [source])

    assert checked.citations == ()
    assert "unknown source" in checked.rejected[0].reason


def test_a_quote_too_short_to_prove_anything_is_rejected():
    source = chunk("The registry asks every server what it offers.")
    checked = validate(reply("ok", [{"source": 1, "quote": "the"}]), [source])

    assert checked.citations == ()
    assert "too short" in checked.rejected[0].reason


def test_the_good_citations_survive_the_bad_ones():
    source = chunk("The registry asks every server what it offers, one by one.")
    checked = validate(
        reply(
            "ok",
            [
                {"source": 1, "quote": "asks every server what it offers"},
                {"source": 1, "quote": "and then writes a report nobody reads"},
            ],
        ),
        [source],
    )

    assert len(checked.citations) == 1
    assert len(checked.rejected) == 1
    assert not checked.passed


def test_a_reply_that_is_not_json_still_becomes_an_answer():
    answer, citations = parse_reply("Just prose, no JSON here.")

    assert answer == "Just prose, no JSON here."
    assert citations == []


def test_json_wrapped_in_a_code_fence_is_still_read():
    answer, citations = parse_reply('```json\n{"answer": "a", "citations": []}\n```')

    assert answer == "a"
    assert citations == []


# --- 5. the metadata behind a source ----------------------------------------


def test_every_cited_chunk_is_one_the_index_really_holds(monkeypatch, retriever):
    no_support_model(monkeypatch)
    quote = "A server that cannot be reached takes only its own tools with it"
    stub_model(monkeypatch, reply("The others keep working.", [{"source": 1, "quote": quote}]))
    answer = asyncio.run(agent_for(retriever, settings()).ask(QUESTION, mode=ENHANCED))
    known = {item.chunk_id: item for item in answer.enhanced.chunks}

    for citation in answer.cited.citations:
        assert citation.chunk_id in known
        assert flatten(citation.quote) in flatten(known[citation.chunk_id].text)


def test_the_prompt_shows_the_metadata_but_asks_for_a_number():
    chunks = [chunk("Some text here for the model.", chunk_id="reg_0", file="registry.py")]
    prompt = cited_prompt("why?", chunks)

    assert "[1] file: registry.py" in prompt
    assert "Cite by NUMBER only" in prompt
    assert "reg_0" not in prompt, "the model is never shown a chunk id to copy"


def test_the_numbering_starts_at_one_and_follows_the_ranking():
    chunks = [chunk("First.", chunk_id="a"), chunk("Second.", chunk_id="b")]

    assert numbered_context(chunks).startswith("[1] ")
    assert "[2] " in numbered_context(chunks)


# --- 6. the relevance threshold ---------------------------------------------


def test_the_answer_threshold_is_a_setting_not_a_number_in_the_code():
    assert RagSettings().answer_threshold == 0.70
    assert settings(answer_threshold=0.9).answer_threshold == 0.9

    with pytest.raises(ValueError):
        RagSettings(answer_threshold=1.5)


def test_a_best_score_below_the_threshold_stops_the_model_being_called(monkeypatch, retriever):
    """The check the assignment spells out: below the bar, no answering flow."""
    prompts = stub_model(monkeypatch, reply("should never be produced", []))
    answer = asyncio.run(
        agent_for(retriever, settings(answer_threshold=0.99)).ask(QUESTION, mode=ENHANCED)
    )

    assert prompts == [], "the model must not be asked"
    assert answer.cited.rag_status == INSUFFICIENT
    assert answer.tokens_used == 0
    assert answer.llm_seconds == 0.0


def test_a_best_score_above_the_threshold_lets_the_answer_through(monkeypatch, retriever):
    no_support_model(monkeypatch)
    prompts = stub_model(monkeypatch, reply("The others keep working.", []))
    answer = asyncio.run(
        agent_for(retriever, settings(answer_threshold=0.0)).ask(QUESTION, mode=ENHANCED)
    )

    assert len(prompts) == 1
    assert answer.cited.rag_status == ANSWERED


def test_the_relevance_is_the_rerank_score_and_both_numbers_are_reported(monkeypatch, retriever):
    no_support_model(monkeypatch)
    stub_model(monkeypatch, reply("ok", []))
    answer = asyncio.run(agent_for(retriever, settings()).ask(QUESTION, mode=ENHANCED))
    record = answer.enhanced.as_dict()

    assert record["best_relevance"] == pytest.approx(answer.enhanced.best_relevance, abs=0.001)
    assert record["best_similarity"] >= record["best_relevance"] or record["best_similarity"] > 0


# --- 7. insufficient context ------------------------------------------------


def test_the_insufficient_answer_says_what_to_do_about_it(monkeypatch, retriever):
    stub_model(monkeypatch, reply("unused", []))
    answer = asyncio.run(
        agent_for(retriever, settings(answer_threshold=0.99)).ask(QUESTION, mode=ENHANCED)
    )
    record = answer.cited.as_dict()

    assert record["answer"] == INSUFFICIENT_ANSWER
    assert "clarify" in record["answer"].lower()
    assert record["sources"] == []
    assert record["citations"] == []
    assert record["confidence"] == LOW
    assert record["rag_status"] == INSUFFICIENT


def test_nothing_surviving_the_filter_is_also_insufficient(monkeypatch, retriever):
    prompts = stub_model(monkeypatch, reply("unused", []))
    answer = asyncio.run(
        agent_for(retriever, settings(similarity_threshold=0.99, answer_threshold=0.0)).ask(
            QUESTION, mode=ENHANCED
        )
    )

    assert prompts == []
    assert answer.cited.rag_status == INSUFFICIENT
    assert answer.enhanced.filtered.kept == ()


# --- 8. a question the index cannot answer ----------------------------------


def test_an_unknown_question_is_refused_rather_than_answered(monkeypatch, retriever):
    """The whole point: low relevance must not become a confident paragraph."""
    prompts = stub_model(monkeypatch, reply("It uses PostgreSQL with a conversations table.", []))
    answer = asyncio.run(
        agent_for(retriever, settings(answer_threshold=0.5)).ask(
            "Which charting library does the billing dashboard use?", mode=ENHANCED
        )
    )

    assert prompts == [], "no random chunks may reach the model"
    assert "PostgreSQL" not in answer.answer
    assert answer.cited.rag_status == INSUFFICIENT
    assert answer.cited.sources == []


def test_the_refusal_still_reports_the_whole_funnel(monkeypatch, retriever):
    stub_model(monkeypatch, reply("unused", []))
    answer = asyncio.run(
        agent_for(retriever, settings(answer_threshold=0.99)).ask(QUESTION, mode=ENHANCED)
    )
    record = answer.enhanced.as_dict()

    assert record["retrieved_count"] > 0
    assert record["answer_threshold"] == 0.99
    assert answer.cited.best_relevance < 0.99


# --- 9. the response schema -------------------------------------------------


def test_every_mode_answers_in_the_same_shape(monkeypatch, retriever):
    no_support_model(monkeypatch)
    stub_model(monkeypatch, reply("An answer.", []))
    agent = agent_for(retriever, settings(answer_threshold=0.0))
    wanted = {"answer", "sources", "citations", "confidence", "rag_status", "citation_support"}

    for mode in (OFF, BASELINE, ENHANCED):
        record = asyncio.run(agent.ask(QUESTION, mode=mode)).cited.as_dict()

        assert set(record) == wanted, mode
        assert isinstance(record["sources"], list)
        assert isinstance(record["citations"], list)


def test_rag_off_reports_itself_as_disabled(monkeypatch, retriever):
    stub_model(monkeypatch, "From memory alone.")
    answer = asyncio.run(agent_for(retriever, settings()).ask(QUESTION, mode=OFF))

    assert answer.cited.rag_status == DISABLED
    assert answer.cited.sources == []
    assert answer.cited.citations == []


def test_baseline_names_its_documents_but_offers_no_quotes(monkeypatch, retriever):
    stub_model(monkeypatch, "An answer from the index.")
    answer = asyncio.run(agent_for(retriever, settings()).ask(QUESTION, mode=BASELINE))

    assert answer.cited.rag_status == ANSWERED
    assert answer.cited.sources, "baseline still says which documents it was given"
    assert answer.cited.citations == []
    assert answer.cited.confidence == LOW, "nothing was quoted, so nothing was checked"


def test_confidence_follows_the_evidence():
    citations = [Citation("a.py", "s", "a_1", "q")]

    assert confidence_of(ANSWERED, 0.90, 0.70, citations, SUPPORTED) == HIGH
    assert confidence_of(ANSWERED, 0.71, 0.70, citations, SUPPORTED) == MEDIUM
    assert confidence_of(ANSWERED, 0.90, 0.70, citations, NOT_CHECKED) == MEDIUM
    assert confidence_of(ANSWERED, 0.90, 0.70, citations, UNSUPPORTED) == LOW
    assert confidence_of(ANSWERED, 0.90, 0.70, [], SUPPORTED) == LOW
    assert confidence_of(INSUFFICIENT, 0.90, 0.70, citations, SUPPORTED) == LOW


def test_the_response_schema_survives_being_written_down(monkeypatch, retriever):
    no_support_model(monkeypatch)
    quote = "A server that cannot be reached takes only its own tools with it"
    stub_model(monkeypatch, reply("The others keep working.", [{"source": 1, "quote": quote}]))
    answer = asyncio.run(agent_for(retriever, settings()).ask(QUESTION, mode=ENHANCED))

    assert json.loads(json.dumps(answer.cited.as_record()))


# --- 10. citation support ---------------------------------------------------


def test_a_claim_the_evidence_does_not_make_is_unsupported(monkeypatch):
    no_support_model(monkeypatch)
    support = asyncio.run(
        claim_support.check(
            "The project stores conversations in a PostgreSQL database with indexed tables.",
            ["A server that cannot be reached takes only its own tools with it."],
        )
    )

    assert support.verdict == UNSUPPORTED
    assert support.checked_by == "lexical"
    assert support.unsupported


def test_a_claim_the_evidence_makes_is_supported(monkeypatch):
    no_support_model(monkeypatch)
    support = asyncio.run(
        claim_support.check(
            "A server that cannot be reached takes only its own tools with it.",
            ["A server that cannot be reached takes only its own tools with it."],
        )
    )

    assert support.verdict == SUPPORTED


def test_support_is_not_claimed_when_there_was_nothing_to_compare(monkeypatch):
    no_support_model(monkeypatch)
    support = asyncio.run(claim_support.check("An answer.", []))

    assert support.verdict == NOT_CHECKED, "no evidence is not the same as supported"


def test_an_existing_citation_is_not_enough_to_call_an_answer_supported(monkeypatch, retriever):
    """A real quote attached to a claim it does not make is still unsupported."""
    no_support_model(monkeypatch)
    quote = "A server that cannot be reached takes only its own tools with it"
    stub_model(
        monkeypatch,
        reply(
            "The scheduler writes every conversation into a PostgreSQL table called messages.",
            [{"source": 1, "quote": quote}],
        ),
    )
    answer = asyncio.run(agent_for(retriever, settings()).ask(QUESTION, mode=ENHANCED))

    assert answer.cited.citations, "the citation itself is valid"
    assert answer.cited.citation_support == UNSUPPORTED
    assert answer.cited.confidence == LOW


def test_the_support_check_says_which_way_it_was_run(monkeypatch, retriever):
    no_support_model(monkeypatch)
    quote = "A server that cannot be reached takes only its own tools with it"
    stub_model(monkeypatch, reply("The others keep working, and so does the call.", [{"source": 1, "quote": quote}]))
    answer = asyncio.run(agent_for(retriever, settings()).ask(QUESTION, mode=ENHANCED))

    assert answer.cited.support_checked_by in ("model", "lexical")


def test_the_model_check_is_used_when_it_answers(monkeypatch, retriever):
    async def fake_support(prompt, temperature=None, model=None):
        return '{"unsupported": []}'

    monkeypatch.setattr(claim_support, "generate_text", fake_support)
    quote = "A server that cannot be reached takes only its own tools with it"
    stub_model(monkeypatch, reply("Only that server's tools go away.", [{"source": 1, "quote": quote}]))
    answer = asyncio.run(agent_for(retriever, settings()).ask(QUESTION, mode=ENHANCED))

    assert answer.cited.support_checked_by == "model"
    assert answer.cited.citation_support == SUPPORTED


# --- the prompt says what the assignment asks it to say ---------------------


def test_the_prompt_forbids_guessing_and_gives_the_exact_refusal():
    prompt = cited_prompt("why?", [chunk("Some text for the model to read.")])

    assert "Do not add facts from general knowledge" in prompt
    assert "must be supported by one of the extracts" in prompt
    assert DONT_KNOW in prompt
    assert "exact" in prompt.lower()


def test_a_prompt_without_extracts_only_asks_for_the_refusal():
    prompt = cited_prompt("why?", [])

    assert DONT_KNOW in prompt
    assert "[1]" not in prompt


# --- the endpoint -----------------------------------------------------------


def rag_route(monkeypatch, index_dir, config: RagSettings, text: str):
    from fastapi.testclient import TestClient

    from app.api.routes import agent as route_module
    from app.main import app
    from app.services import rag_agent as agent_module

    async def fake(prompt, model=None, temperature=None):
        return GeneratedText(text=text, input_tokens=100, output_tokens=20)

    monkeypatch.setattr(agent_module, "generate_text_with_usage", fake)
    no_support_model(monkeypatch)
    retriever = RAGRetriever(strategy=STRUCTURAL, index_dir=index_dir)
    monkeypatch.setattr(
        route_module,
        "rag_agent",
        RagAgent(
            retriever=retriever,
            settings=config,
            enhanced=EnhancedRetriever(retriever=retriever, settings=config),
        ),
    )

    return TestClient(app)


def test_the_endpoint_returns_the_schema(monkeypatch, index_dir):
    quote = "A server that cannot be reached takes only its own tools with it"
    client = rag_route(
        monkeypatch,
        index_dir,
        settings(),
        reply("The others keep working.", [{"source": 1, "quote": quote}]),
    )

    body = client.post("/agent/rag", json={"question": QUESTION, "mode": "enhanced"}).json()

    assert body["rag_status"] == ANSWERED
    assert body["citations"], "the endpoint carries the quotes"
    assert body["cited_sources"][0]["chunk_id"]
    assert body["citations"][0]["quote"] in REGISTRY.replace("\n", " ")
    assert body["confidence"] in (HIGH, MEDIUM, LOW)
    assert body["debug"]["best_relevance"] > 0


def test_the_endpoint_refuses_without_evidence(monkeypatch, index_dir):
    client = rag_route(monkeypatch, index_dir, settings(answer_threshold=0.99), reply("unused", []))

    body = client.post("/agent/rag", json={"question": QUESTION, "mode": "enhanced"}).json()

    assert body["rag_status"] == INSUFFICIENT
    assert body["answer"] == INSUFFICIENT_ANSWER
    assert body["cited_sources"] == []
    assert body["citations"] == []
    assert body["confidence"] == LOW
