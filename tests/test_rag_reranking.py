"""Day 23: the second stage after FAISS.

Twelve things the assignment asks to prove, in the order it asks for them:
the rewrite happens and stays out of the prompt, the threshold is a setting
rather than a number in the code, what is below it is removed, keywords are
scored, the order changes, only final_top_k survives, a question with
nothing relevant sends nothing, both RAG modes still work, metadata and
sources come through intact, and the comparison counts what it says.

Two of them are spelled out separately because they are the point of the
day: a chunk whose score is below the threshold must be gone, and the model
must never see more than final_top_k chunks.

Everything runs on a real FAISS index built here with the local embedding
backend - no key, no network - and the model is stubbed, because what is
under test is the funnel around it.
"""

import asyncio
import json
from pathlib import Path

import pytest

from app.services.chunking import STRUCTURAL, chunk_documents
from app.services.document_loader import Document
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import GeneratedText
from app.services.query_rewriter import FALLBACK, MODEL, NONE, QueryRewriter, keyword_query
from app.services.rag_agent import BASELINE, ENHANCED, OFF, RagAgent, mode_of
from app.services.rag_enhanced import EnhancedRetriever
from app.services.rag_filter import RelevanceFilter
from app.services.rag_prompt import DONT_KNOW
from app.services.rag_response import INSUFFICIENT_ANSWER
from app.services.rag_retriever import RAGRetriever, RetrievedChunk
from app.services.rag_settings import RagSettings
from app.services.reranker import Reranker, overlap, terms
from app.services.vector_index import save_index

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

## Review queue

The review queue is rebuilt every morning from what was answered wrongly the
day before, and a character that has been answered correctly three times in a
row leaves the queue until the month is out. Nothing about the queue is stored
on the server: it is worked out again from the answer history each time.
"""

VOCABULARY = """# Vocabulary lists

## Levels

Words are grouped by the level they belong to, and a list is only ever built
from the levels the learner has actually asked for. A word that appears at two
levels is kept at the lower of them, so a beginner's list never quietly grows
by the same word arriving from somewhere harder.

## Collecting

Collecting runs in the background and writes what it found to disk after every
batch, because a collector that keeps everything in memory loses the lot when
the process restarts. The file it writes is the only state there is.
"""


def documents() -> list[Document]:
    return [
        Document("docs", "docs/scheduler.md", "Scheduler", SCHEDULER),
        Document("docs", "docs/registry.md", "MCP registry", REGISTRY),
        Document("docs", "docs/kanji.md", "Kanji practice", KANJI),
        Document("docs", "docs/vocabulary.md", "Vocabulary lists", VOCABULARY),
    ]


@pytest.fixture
def index_dir(tmp_path) -> Path:
    """A real FAISS index over four small documents, embedded locally."""
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


SCHEDULER_QUESTION = "How often does the digest scheduler tick, and what is the shortest interval?"


def settings(**changes) -> RagSettings:
    """Test settings: the rewrite off unless a test turns it on, so nothing
    reaches for a model that is not there by accident."""
    base = {
        "retrieval_top_k": 8,
        "similarity_threshold": 0.3,
        "final_top_k": 3,
        "query_rewrite": False,
        # Day 24 added a second gate after this one. These tests are about
        # the funnel, so it is opened here and tested on its own elsewhere.
        "answer_threshold": 0.0,
    }
    base.update(changes)

    return RagSettings(**base)


def chunk(score: float, chunk_id: str, text: str = "text", file: str = "a.py", section: str = "s"):
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


def stub_model(monkeypatch, answer: str = "An answer.") -> list[str]:
    """Record what the model was shown, and answer something fixed."""
    from app.services import rag_agent as module

    prompts: list[str] = []

    async def fake(prompt, model=None, temperature=None):
        prompts.append(prompt)
        return GeneratedText(text=answer, input_tokens=100, output_tokens=20)

    monkeypatch.setattr(module, "generate_text_with_usage", fake)

    return prompts


def stub_rewriter(monkeypatch, reply: str) -> list[str]:
    """Answer the rewrite call with a fixed query, and record the prompt."""
    from app.services import query_rewriter as module

    prompts: list[str] = []

    async def fake(prompt, temperature=None, model=None):
        prompts.append(prompt)
        return reply

    monkeypatch.setattr(module, "generate_text", fake)

    return prompts


def enhanced_agent(retriever: RAGRetriever, config: RagSettings) -> RagAgent:
    return RagAgent(
        retriever=retriever,
        settings=config,
        enhanced=EnhancedRetriever(retriever=retriever, settings=config),
    )


# --- 1. query rewrite ------------------------------------------------------


def test_the_question_is_rewritten_into_a_search_query(monkeypatch):
    stub_rewriter(monkeypatch, "MCP architecture, MCP server, MCP client, orchestration")
    rewritten = asyncio.run(QueryRewriter().rewrite("Как работает MCP в проекте?"))

    assert rewritten.used == MODEL
    assert rewritten.query == "MCP architecture, MCP server, MCP client, orchestration"
    assert rewritten.original == "Как работает MCP в проекте?"
    assert rewritten.rewritten is True


def test_the_rewrite_can_be_switched_off(monkeypatch):
    prompts = stub_rewriter(monkeypatch, "should not be used")
    rewritten = asyncio.run(QueryRewriter(enabled=False).rewrite("Как работает MCP?"))

    assert rewritten.used == NONE
    assert rewritten.query == "Как работает MCP?"
    assert rewritten.rewritten is False
    assert prompts == []


def test_a_failed_rewrite_falls_back_to_keywords_instead_of_failing(monkeypatch):
    from app.services import query_rewriter as module

    async def broken(prompt, temperature=None, model=None):
        raise RuntimeError("no model today")

    monkeypatch.setattr(module, "generate_text", broken)
    rewritten = asyncio.run(QueryRewriter().rewrite("How does the MCP registry route a call?"))

    assert rewritten.used == FALLBACK
    assert "registry" in rewritten.query
    assert "how" not in rewritten.query.lower().split()


def test_a_model_that_starts_explaining_is_not_used_as_a_query(monkeypatch):
    stub_rewriter(monkeypatch, "Well, to answer that we should first consider " + "words " * 80)
    rewritten = asyncio.run(QueryRewriter().rewrite("How does routing work?"))

    assert rewritten.used == FALLBACK


def test_the_keyword_fallback_drops_grammar_and_keeps_the_subject():
    assert keyword_query("Как работает MCP в проекте?") == "MCP проекте"
    assert "scheduler" in keyword_query("How often does the scheduler tick?")


def test_the_search_gets_the_rewrite_and_the_model_gets_the_question(monkeypatch, retriever):
    stub_rewriter(monkeypatch, "digest scheduler tick interval seconds")
    prompts = stub_model(monkeypatch)
    agent = enhanced_agent(retriever, settings(query_rewrite=True, similarity_threshold=0.0))
    answer = asyncio.run(agent.ask("Как часто тикает планировщик?", mode=ENHANCED))

    assert answer.enhanced.query.query == "digest scheduler tick interval seconds"
    assert answer.enhanced.retrieval.query == "digest scheduler tick interval seconds"
    # The prompt carries the question as asked, never the rewrite.
    assert "Как часто тикает планировщик?" in prompts[0]
    assert "digest scheduler tick interval seconds" not in prompts[0]


# --- 2. the similarity threshold is a setting ------------------------------


def test_the_threshold_comes_from_settings_not_from_the_code(retriever):
    strict = asyncio.run(
        EnhancedRetriever(retriever=retriever, settings=settings(similarity_threshold=0.9)).retrieve(
            SCHEDULER_QUESTION
        )
    )
    loose = asyncio.run(
        EnhancedRetriever(retriever=retriever, settings=settings(similarity_threshold=0.0)).retrieve(
            SCHEDULER_QUESTION
        )
    )

    assert len(strict.filtered.kept) < len(loose.filtered.kept)
    assert strict.filtered.threshold == 0.9
    assert loose.filtered.threshold == 0.0


def test_one_call_can_use_a_different_threshold_without_changing_the_others(retriever):
    pipeline = EnhancedRetriever(retriever=retriever, settings=settings(similarity_threshold=0.0))
    strict = asyncio.run(pipeline.retrieve(SCHEDULER_QUESTION, settings=settings(similarity_threshold=0.9)))

    assert strict.settings.similarity_threshold == 0.9
    assert pipeline.settings.similarity_threshold == 0.0


def test_a_threshold_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError):
        RelevanceFilter(1.5)

    with pytest.raises(ValueError):
        RagSettings(similarity_threshold=-0.1)


def test_final_top_k_larger_than_retrieval_top_k_is_refused():
    with pytest.raises(ValueError):
        RagSettings(retrieval_top_k=3, final_top_k=10)


# --- 3. filtering: below the threshold means gone --------------------------


def test_a_chunk_below_the_threshold_is_removed():
    kept_high = chunk(0.82, "high")
    kept_edge = chunk(0.70, "edge")
    dropped = chunk(0.69, "low")
    result = RelevanceFilter(0.70).apply([kept_high, kept_edge, dropped])

    assert [item.chunk_id for item in result.kept] == ["high", "edge"]
    assert [item.chunk_id for item in result.dropped] == ["low"]
    assert all(item.score >= 0.70 for item in result.kept)
    assert all(item.score < 0.70 for item in result.dropped)


def test_what_was_dropped_is_reported_with_its_score():
    result = RelevanceFilter(0.70).apply([chunk(0.82, "high"), chunk(0.31, "low")])
    record = result.as_dict()

    assert record["kept"] == 1
    assert record["dropped"] == 1
    assert record["dropped_chunks"] == [
        {"chunk_id": "low", "reference": "a.py / s", "score": 0.31}
    ]


def test_filtering_keeps_the_order_it_was_given():
    ordered = [chunk(0.9, "a"), chunk(0.8, "b"), chunk(0.75, "c")]
    result = RelevanceFilter(0.5).apply(ordered)

    assert [item.chunk_id for item in result.kept] == ["a", "b", "c"]


# --- 4. keyword scoring ----------------------------------------------------


def test_the_query_terms_are_the_words_that_mean_something():
    found = terms("How often does the digest scheduler tick, and what is the shortest interval?")

    assert "digest" in found and "scheduler" in found and "interval" in found
    assert "the" not in found and "does" not in found and "what" not in found


def test_overlap_is_the_share_of_terms_the_text_uses():
    score, matched = overlap("the digest scheduler ticks once a second", ("digest", "scheduler", "kanji"))

    assert score == pytest.approx(2 / 3, abs=0.001)
    assert set(matched) == {"digest", "scheduler"}


def test_a_text_that_uses_none_of_the_terms_scores_zero():
    score, matched = overlap("stroke order is taught from the top left", ("digest", "scheduler"))

    assert score == 0.0
    assert matched == ()


def test_the_label_counts_as_well_as_the_body():
    reranker = Reranker(similarity_weight=0.0, keyword_weight=1.0, section_weight=1.0)
    in_label = reranker.score(chunk(0.5, "a", text="nothing", file="app/services/digest_scheduler.py"), ("digest",))
    in_body = reranker.score(chunk(0.5, "b", text="the digest is written", file="other.py"), ("digest",))

    assert in_label.label_score == 1.0
    assert in_body.label_score == 0.0
    assert in_label.rerank_score > in_body.rerank_score


# --- 5. reranking ----------------------------------------------------------


def test_the_reranker_can_beat_similarity_when_the_words_are_there():
    generic = chunk(0.74, "generic", text="general talk about background jobs", file="README.md")
    exact = chunk(
        0.73,
        "exact",
        text="the digest scheduler tick is one second and the shortest interval is ten",
        file="app/services/digest_scheduler.py",
    )
    result = Reranker().rerank([generic, exact], "digest scheduler tick shortest interval")

    assert [item.chunk.chunk_id for item in result.chunks] == ["exact", "generic"]
    assert result.reordered is True
    assert result.chunks[0].rerank_score > result.chunks[1].rerank_score


def test_the_weights_decide_how_much_the_keywords_matter():
    generic = chunk(0.74, "generic", text="general talk", file="README.md")
    exact = chunk(0.73, "exact", text="digest scheduler tick", file="digest_scheduler.py")
    query = "digest scheduler tick"

    only_similarity = Reranker(similarity_weight=1.0, keyword_weight=0.0).rerank([generic, exact], query)
    only_keywords = Reranker(similarity_weight=0.0, keyword_weight=1.0).rerank([generic, exact], query)

    assert [item.chunk.chunk_id for item in only_similarity.chunks] == ["generic", "exact"]
    assert [item.chunk.chunk_id for item in only_keywords.chunks] == ["exact", "generic"]


def test_every_chunk_reports_the_three_numbers_behind_its_place():
    result = Reranker().rerank([chunk(0.8, "a", text="digest scheduler")], "digest scheduler")
    scored = result.chunks[0]

    assert scored.similarity_score == 0.8
    assert scored.keyword_score > 0
    assert scored.rerank_score == pytest.approx(0.7 * 0.8 + 0.3 * scored.keyword_score, abs=0.001)
    assert set(scored.matched) == {"digest", "scheduler"}


def test_the_same_input_always_gives_the_same_order():
    same = [chunk(0.7, "b", text="x"), chunk(0.7, "a", text="x"), chunk(0.7, "c", text="x")]
    first = Reranker().rerank(same, "nothing matches here")
    second = Reranker().rerank(same, "nothing matches here")

    assert [item.chunk.chunk_id for item in first.chunks] == ["a", "b", "c"]
    assert [item.chunk.chunk_id for item in first.chunks] == [
        item.chunk.chunk_id for item in second.chunks
    ]


def test_a_reranker_with_no_weight_at_all_is_refused():
    with pytest.raises(ValueError):
        Reranker(similarity_weight=0.0, keyword_weight=0.0)


# --- 6. final top-K --------------------------------------------------------


def test_no_more_than_final_top_k_chunks_reach_the_model(monkeypatch, retriever):
    prompts = stub_model(monkeypatch)
    agent = enhanced_agent(retriever, settings(final_top_k=3, similarity_threshold=0.0))
    answer = asyncio.run(agent.ask(SCHEDULER_QUESTION, mode=ENHANCED))

    assert answer.enhanced.filtered.kept  # there was more to choose from
    assert len(answer.enhanced.filtered.kept) > 3
    assert len(answer.retrieved_chunks) == 3
    assert prompts[0].count("] file: ") == 3


@pytest.mark.parametrize("final_top_k", [1, 2, 4])
def test_final_top_k_is_what_comes_out(monkeypatch, retriever, final_top_k):
    stub_model(monkeypatch)
    agent = enhanced_agent(retriever, settings(final_top_k=final_top_k, similarity_threshold=0.0))
    answer = asyncio.run(agent.ask(SCHEDULER_QUESTION, mode=ENHANCED))

    assert len(answer.retrieved_chunks) == final_top_k


def test_fewer_survivors_than_final_top_k_is_not_padded(monkeypatch, retriever):
    stub_model(monkeypatch)
    agent = enhanced_agent(retriever, settings(final_top_k=5, similarity_threshold=0.44))
    answer = asyncio.run(agent.ask(SCHEDULER_QUESTION, mode=ENHANCED))

    assert 0 < len(answer.retrieved_chunks) < 5
    assert len(answer.retrieved_chunks) == len(answer.enhanced.filtered.kept)


# --- 7. nothing relevant ---------------------------------------------------


def test_a_question_the_index_cannot_answer_sends_nothing_to_the_model(monkeypatch, retriever):
    prompts = stub_model(monkeypatch)
    agent = enhanced_agent(retriever, settings(similarity_threshold=0.9))
    answer = asyncio.run(agent.ask("Which relational database stores the conversations?", mode=ENHANCED))

    assert prompts == []
    assert answer.answer == INSUFFICIENT_ANSWER
    assert DONT_KNOW in answer.answer
    assert answer.retrieved_chunks == []
    assert answer.sources == []


def test_the_empty_result_still_reports_the_whole_funnel(monkeypatch, retriever):
    stub_model(monkeypatch)
    agent = enhanced_agent(retriever, settings(similarity_threshold=0.9))
    answer = asyncio.run(agent.ask("Which relational database stores the conversations?", mode=ENHANCED))
    record = answer.enhanced.as_dict()

    assert record["retrieved_count"] > 0
    assert record["filtered_count"] == 0
    assert record["final_count"] == 0
    assert record["threshold"] == 0.9
    assert answer.enhanced.nothing_relevant is True
    assert answer.enhanced.filtered.everything_was_dropped is True


def test_baseline_would_have_sent_those_chunks_anyway(monkeypatch, retriever):
    """The difference the filter makes, stated as a test."""
    prompts = stub_model(monkeypatch)
    question = "Which relational database stores the conversations?"
    baseline = asyncio.run(RagAgent(retriever=retriever, top_k=5).ask(question, mode=BASELINE))

    assert len(baseline.retrieved_chunks) == 5
    assert prompts[0].count("[Source:") == 5


# --- 8. the baseline is untouched ------------------------------------------


def test_baseline_is_day_22_unchanged(monkeypatch, retriever):
    prompts = stub_model(monkeypatch)
    answer = asyncio.run(RagAgent(retriever=retriever, top_k=4).ask(SCHEDULER_QUESTION, mode=BASELINE))

    assert answer.mode == BASELINE
    assert answer.rag_enabled is True
    assert len(answer.retrieved_chunks) == 4
    assert answer.enhanced is None
    assert prompts[0].count("[Source:") == 4


def test_the_day_22_flag_still_decides_when_no_mode_is_given(monkeypatch, retriever):
    stub_model(monkeypatch)
    agent = RagAgent(retriever=retriever)

    assert asyncio.run(agent.ask(SCHEDULER_QUESTION, use_rag=True)).mode == BASELINE
    assert asyncio.run(agent.ask(SCHEDULER_QUESTION, use_rag=False)).mode == OFF
    assert mode_of(True, ENHANCED) == ENHANCED


def test_off_retrieves_nothing_at_all(monkeypatch, retriever):
    prompts = stub_model(monkeypatch)
    answer = asyncio.run(RagAgent(retriever=retriever).ask(SCHEDULER_QUESTION, mode=OFF))

    assert answer.mode == OFF
    assert answer.rag_enabled is False
    assert answer.retrieved_chunks == []
    assert "[Source:" not in prompts[0]


def test_an_unknown_mode_is_refused(retriever):
    with pytest.raises(ValueError):
        asyncio.run(RagAgent(retriever=retriever).ask(SCHEDULER_QUESTION, mode="clever"))


# --- 9. enhanced end to end ------------------------------------------------


def test_the_funnel_only_ever_narrows(monkeypatch, retriever):
    stub_model(monkeypatch)
    agent = enhanced_agent(retriever, settings(retrieval_top_k=8, similarity_threshold=0.3, final_top_k=3))
    found = asyncio.run(agent.ask(SCHEDULER_QUESTION, mode=ENHANCED)).enhanced

    retrieved = len(found.retrieval.chunks)
    kept = len(found.filtered.kept)
    final = len(found.final)

    assert retrieved >= kept >= final
    assert final <= 3
    assert kept + len(found.filtered.dropped) == retrieved


def test_more_is_retrieved_than_is_sent(monkeypatch, retriever):
    stub_model(monkeypatch)
    agent = enhanced_agent(retriever, settings(retrieval_top_k=8, similarity_threshold=0.0, final_top_k=2))
    found = asyncio.run(agent.ask(SCHEDULER_QUESTION, mode=ENHANCED)).enhanced

    assert len(found.retrieval.chunks) > len(found.final)
    assert found.retrieval.top_k == 8


def test_the_debug_record_has_everything_the_assignment_asks_for(monkeypatch, retriever):
    stub_model(monkeypatch)
    agent = enhanced_agent(retriever, settings())
    record = asyncio.run(agent.ask(SCHEDULER_QUESTION, mode=ENHANCED)).enhanced.as_dict()

    for field in (
        "original_query",
        "rewritten_query",
        "retrieval_top_k",
        "retrieved_count",
        "filtered_count",
        "final_count",
        "threshold",
        "chunks",
    ):
        assert field in record, field

    for item in record["chunks"]:
        for field in ("chunk_id", "source", "section", "similarity_score", "keyword_score", "rerank_score"):
            assert field in item, field


def test_the_answer_reports_the_mode_it_was_produced_in(monkeypatch, retriever):
    stub_model(monkeypatch)
    agent = enhanced_agent(retriever, settings())
    record = asyncio.run(agent.ask(SCHEDULER_QUESTION, mode=ENHANCED)).as_dict()

    assert record["mode"] == ENHANCED
    assert "enhanced" in record


# --- 10. metadata survives the second stage --------------------------------


def test_every_surviving_chunk_keeps_its_metadata(monkeypatch, retriever):
    stub_model(monkeypatch)
    agent = enhanced_agent(retriever, settings())
    found = asyncio.run(agent.ask(SCHEDULER_QUESTION, mode=ENHANCED)).enhanced

    for item in found.final:
        original = item.chunk

        assert original.chunk_id
        assert original.file.endswith(".md")
        assert original.strategy == STRUCTURAL
        assert original.position >= 0
        assert original.text
        assert original.score == item.similarity_score


def test_the_scores_the_reranker_adds_do_not_overwrite_the_similarity(monkeypatch, retriever):
    stub_model(monkeypatch)
    agent = enhanced_agent(retriever, settings())
    found = asyncio.run(agent.ask(SCHEDULER_QUESTION, mode=ENHANCED)).enhanced

    for item in found.final:
        assert item.as_dict()["score"] == round(item.chunk.score, 4)
        assert item.as_dict()["similarity_score"] == round(item.chunk.score, 4)


# --- 11. sources survive too -----------------------------------------------


def test_the_sources_are_the_final_chunks_and_nothing_else(monkeypatch, retriever):
    stub_model(monkeypatch)
    agent = enhanced_agent(retriever, settings(final_top_k=2, similarity_threshold=0.0))
    answer = asyncio.run(agent.ask(SCHEDULER_QUESTION, mode=ENHANCED))
    final_references = {item.reference for item in answer.enhanced.final}

    assert set(answer.sources) <= final_references
    assert len(answer.sources) <= 2

    for dropped in answer.enhanced.filtered.dropped:
        assert dropped.reference not in answer.sources or dropped.reference in final_references


def test_the_prompt_names_the_files_the_answer_may_cite(monkeypatch, retriever):
    prompts = stub_model(monkeypatch)
    agent = enhanced_agent(retriever, settings(final_top_k=2, similarity_threshold=0.0))
    answer = asyncio.run(agent.ask(SCHEDULER_QUESTION, mode=ENHANCED))

    for item in answer.enhanced.final:
        assert f"file: {item.chunk.file}" in prompts[0]


# --- 12. the comparison ----------------------------------------------------


def test_the_comparison_metrics_count_what_they_say_they_count():
    from app.compare_rag import metrics

    records = [
        {
            "expected_sources": ["app/services/digest_scheduler.py"],
            "off": {"expected_words": 0.1, "says_unknown": False, "llm_seconds": 9.0},
            "baseline": {
                "expected_sources_retrieved": ["app/services/digest_scheduler.py"],
                "irrelevant_chunks_sent": 2,
                "expected_words": 0.8,
                "says_unknown": False,
                "retrieval_seconds": 0.3,
                "llm_seconds": 6.0,
                "total_seconds": 6.3,
            },
            "enhanced": {
                "expected_sources_retrieved": ["app/services/digest_scheduler.py"],
                "irrelevant_chunks_sent": 0,
                "dropped_count": 4,
                "retrieved_count": 10,
                "filtered_count": 6,
                "final_count": 3,
                "reordered": True,
                "expected_words": 0.9,
                "says_unknown": False,
                "rewrite_seconds": 0.4,
                "retrieval_seconds": 0.3,
                "rerank_seconds": 0.001,
                "llm_seconds": 5.0,
                "total_seconds": 5.7,
            },
        },
        {
            "expected_sources": [],
            "off": {"expected_words": 0.0, "says_unknown": False, "llm_seconds": 8.0},
            "baseline": {
                "expected_sources_retrieved": [],
                "irrelevant_chunks_sent": 5,
                "expected_words": 0.2,
                "says_unknown": False,
                "retrieval_seconds": 0.3,
                "llm_seconds": 6.0,
                "total_seconds": 6.3,
            },
            "enhanced": {
                "expected_sources_retrieved": [],
                "irrelevant_chunks_sent": 0,
                "dropped_count": 10,
                "retrieved_count": 10,
                "filtered_count": 0,
                "final_count": 0,
                "reordered": False,
                "expected_words": 0.6,
                "says_unknown": True,
                "rewrite_seconds": 0.4,
                "retrieval_seconds": 0.3,
                "rerank_seconds": 0.0,
                "llm_seconds": 0.0,
                "total_seconds": 0.3,
            },
        },
    ]
    counted = metrics(records, settings(similarity_threshold=0.7), 12.5)

    assert counted["total_questions"] == 2
    assert counted["questions_with_expected_sources"] == 1
    assert counted["unanswerable_questions"] == 1
    assert counted["source_retrieval_accuracy"] == {"baseline": 1.0, "enhanced": 1.0}
    assert counted["irrelevant_chunks_retrieved"] == 14
    assert counted["irrelevant_chunks_reaching_model"] == {"baseline": 7, "enhanced": 0}
    assert counted["questions_with_zero_relevant_chunks"] == 1
    assert counted["unanswerable_handled"] == {"baseline": 0, "enhanced": 1}
    assert counted["average_retrieved_chunks"] == 10.0
    assert counted["average_filtered_chunks"] == 3.0
    assert counted["average_final_chunks"] == 1.5
    assert counted["questions_reordered_by_reranker"] == 1
    assert counted["comparison_seconds"] == 12.5


def test_the_comparison_runs_all_three_modes_for_one_question(monkeypatch, retriever):
    from app.compare_rag import one
    from app.evaluate_rag import Question

    stub_model(monkeypatch)
    config = settings(similarity_threshold=0.0, final_top_k=2)
    agent = enhanced_agent(retriever, config)
    question = Question(
        id=1,
        category="factual",
        question=SCHEDULER_QUESTION,
        expected="ticks once a second, shortest interval ten seconds",
        expected_sources=("docs/scheduler.md",),
    )
    record = asyncio.run(one(agent, question, config))

    assert set(record) >= {"id", "question", "expected", "expected_sources", "off", "baseline", "enhanced"}
    assert record["enhanced"]["final_count"] == 2
    assert record["enhanced"]["retrieved_count"] >= record["enhanced"]["filtered_count"]
    assert record["baseline"]["retrieved_count"] == 5
    assert record["enhanced"]["expected_sources_retrieved"] == ["docs/scheduler.md"]
    assert json.dumps(record)  # the whole record has to survive being written down


def test_the_comparison_writes_where_the_assignment_says():
    from app.compare_rag import RESULTS_FILE

    assert RESULTS_FILE.name == "day23_results.json"
    assert RESULTS_FILE.parent.name == "rag"


# --- the endpoint the app talks to ------------------------------------------


def rag_route(monkeypatch, index_dir, config: RagSettings | None = None):
    """The real route, over the small test index, with only Gemini stubbed."""
    from fastapi.testclient import TestClient

    from app.api.routes import agent as route_module
    from app.main import app
    from app.services import rag_agent as agent_module

    async def fake(prompt, model=None, temperature=None):
        return GeneratedText(text="An answer.", input_tokens=100, output_tokens=20)

    monkeypatch.setattr(agent_module, "generate_text_with_usage", fake)
    active = config or settings(similarity_threshold=0.0)
    retriever = RAGRetriever(strategy=STRUCTURAL, index_dir=index_dir)
    monkeypatch.setattr(
        route_module,
        "rag_agent",
        RagAgent(
            retriever=retriever,
            settings=active,
            enhanced=EnhancedRetriever(retriever=retriever, settings=active),
        ),
    )

    return TestClient(app)


def test_the_endpoint_answers_in_enhanced_mode_and_reports_the_funnel(monkeypatch, index_dir):
    client = rag_route(monkeypatch, index_dir)

    body = client.post(
        "/agent/rag", json={"question": SCHEDULER_QUESTION, "mode": "enhanced"}
    ).json()

    assert body["mode"] == "enhanced"
    assert body["rag_enabled"] is True
    assert len(body["retrieved_chunks"]) <= 3
    assert body["debug"]["retrieved_count"] >= body["debug"]["filtered_count"]
    assert body["debug"]["filtered_count"] >= body["debug"]["final_count"]
    assert body["debug"]["final_count"] == len(body["retrieved_chunks"])
    assert body["debug"]["original_query"] == SCHEDULER_QUESTION
    assert body["retrieved_chunks"][0]["rerank_score"] > 0


def test_the_endpoint_still_answers_day_22_the_way_it_did(monkeypatch, index_dir):
    client = rag_route(monkeypatch, index_dir)

    body = client.post(
        "/agent/rag", json={"question": SCHEDULER_QUESTION, "use_rag": True, "top_k": 3}
    ).json()

    assert body["mode"] == "baseline"
    assert len(body["retrieved_chunks"]) == 3
    assert body["debug"] is None


def test_the_endpoint_takes_the_threshold_and_the_top_ks_from_the_request(monkeypatch, index_dir):
    client = rag_route(monkeypatch, index_dir)

    body = client.post(
        "/agent/rag",
        json={
            "question": SCHEDULER_QUESTION,
            "mode": "enhanced",
            "retrieval_top_k": 6,
            "similarity_threshold": 0.0,
            "final_top_k": 1,
        },
    ).json()

    assert body["debug"]["retrieval_top_k"] == 6
    assert body["debug"]["threshold"] == 0.0
    assert body["debug"]["final_count"] == 1
    assert len(body["retrieved_chunks"]) == 1


def test_the_endpoint_says_it_does_not_know_rather_than_guessing(monkeypatch, index_dir):
    client = rag_route(monkeypatch, index_dir)

    body = client.post(
        "/agent/rag",
        json={
            "question": "Which relational database stores the conversations?",
            "mode": "enhanced",
            "similarity_threshold": 0.9,
        },
    ).json()

    assert body["answer"] == INSUFFICIENT_ANSWER
    assert body["retrieved_chunks"] == []
    assert body["sources"] == []
    assert body["debug"]["filtered_count"] == 0
    assert body["debug"]["retrieved_count"] > 0


def test_impossible_settings_are_refused_at_the_edge(monkeypatch, index_dir):
    client = rag_route(monkeypatch, index_dir)

    assert (
        client.post(
            "/agent/rag",
            json={"question": "x", "mode": "enhanced", "similarity_threshold": 1.5},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/agent/rag",
            json={"question": "x", "mode": "enhanced", "retrieval_top_k": 2, "final_top_k": 8},
        ).status_code
        == 422
    )
    assert client.post("/agent/rag", json={"question": "x", "mode": "clever"}).status_code == 422
