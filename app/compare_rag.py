"""Day 23: the same ten questions, three ways.

    python -m app.compare_rag
    python -m app.compare_rag --threshold 0.75 --final-top-k 5
    python -m app.compare_rag --only 1 10

Every control question from Day 22 is asked three times - with no retrieval,
with the Day 22 retrieval, and through the second stage - and all three
answers are written to data/rag/day23_results.json next to what was expected
and what each mode actually retrieved.

What is counted, and what is not. Whether the expected file came back is a
fact. How many chunks each stage kept is a fact. Whether a chunk was below
the relevance threshold is a fact, and so is whether such a chunk reached
the model anyway - which is the whole comparison, because that is the number
the filter exists to move. Whether an answer is *good* is not a fact, so it
is not scored here: both answers are kept in full for a person to read, and
the crude word check from Day 22 is reused under the name it earned.
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from app.evaluate_rag import Question, load_questions, mentions_expected, says_unknown
from app.services.document_loader import PROJECT_ROOT
from app.services.rag_agent import BASELINE, ENHANCED, OFF, RagAgent, RagAnswer
from app.services.rag_enhanced import EnhancedRetriever
from app.services.rag_retriever import DEFAULT_STRATEGY, DEFAULT_TOP_K, RAGRetriever
from app.services.rag_settings import DEFAULT_SETTINGS, RagSettings

logger = logging.getLogger("compare-rag")

RESULTS_FILE = PROJECT_ROOT / "data" / "rag" / "day23_results.json"


def expected_found(answer: RagAnswer, expected_sources: tuple[str, ...]) -> list[str]:
    """Which of the expected documents this mode actually put in front of
    the model. Chunks that were retrieved and then dropped do not count -
    the model never saw them."""
    files = {chunk.file for chunk in answer.retrieved_chunks}

    return [source for source in expected_sources if any(source in file for file in files)]


def below(answer: RagAnswer, threshold: float) -> int:
    """How many of the chunks this mode sent to the model were below the
    relevance threshold - the ones the filter exists to keep out."""
    return sum(1 for chunk in answer.retrieved_chunks if chunk.score < threshold)


async def compare(options: argparse.Namespace) -> dict:
    questions = load_questions()

    if options.only:
        questions = [question for question in questions if question.id in options.only]

    settings = DEFAULT_SETTINGS.with_overrides(
        retrieval_top_k=options.retrieval_top_k,
        similarity_threshold=options.threshold,
        final_top_k=options.final_top_k,
        query_rewrite=False if options.no_rewrite else None,
    )
    retriever = RAGRetriever(strategy=options.strategy)
    agent = RagAgent(
        retriever=retriever,
        top_k=options.top_k,
        settings=settings,
        enhanced=EnhancedRetriever(retriever=retriever, settings=settings),
    )
    logger.info("Questions: %d", len(questions))
    logger.info("Index: %s", retriever.directory)
    logger.info("Settings: %s", settings.as_dict())
    logger.info("")

    records = []
    started = time.perf_counter()

    for question in questions:
        logger.info("[%2d/%d] %s", question.id, len(questions), question.question)
        records.append(await one(agent, question, settings))

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "settings": {
            **settings.as_dict(),
            "baseline_top_k": options.top_k,
            "strategy": options.strategy,
            "index": str(retriever.directory),
            "embedding_model": retriever.embeddings().name,
        },
        "metrics": metrics(records, settings, time.perf_counter() - started),
        "results": records,
    }


async def one(agent: RagAgent, question: Question, settings: RagSettings) -> dict:
    """One question, all three modes."""
    off = await agent.ask(question.question, mode=OFF)
    baseline = await agent.ask(question.question, mode=BASELINE)
    enhanced = await agent.ask(question.question, mode=ENHANCED, settings=settings)
    found = enhanced.enhanced
    threshold = settings.similarity_threshold

    record = {
        "id": question.id,
        "category": question.category,
        "question": question.question,
        "expected": question.expected,
        "expected_sources": list(question.expected_sources),
        "off": {
            "answer": off.answer,
            "sources": [],
            "expected_words": mentions_expected(off.answer, question.expected),
            "says_unknown": says_unknown(off.answer),
            "llm_seconds": round(off.llm_seconds, 3),
        },
        "baseline": {
            "answer": baseline.answer,
            "sources": baseline.sources,
            "retrieved_count": len(baseline.retrieved_chunks),
            "scores": [round(chunk.score, 4) for chunk in baseline.retrieved_chunks],
            "expected_sources_retrieved": expected_found(baseline, question.expected_sources),
            "irrelevant_chunks_sent": below(baseline, threshold),
            "expected_words": mentions_expected(baseline.answer, question.expected),
            "says_unknown": says_unknown(baseline.answer),
            "retrieval_seconds": round(baseline.retrieval_seconds, 3),
            "llm_seconds": round(baseline.llm_seconds, 3),
            "total_seconds": round(baseline.total_seconds, 3),
        },
        "enhanced": {
            "answer": enhanced.answer,
            "sources": enhanced.sources,
            "rewritten_query": found.query.query,
            "rewrite_used": found.query.used,
            "retrieved_count": len(found.retrieval.chunks),
            "filtered_count": len(found.filtered.kept),
            "final_count": len(found.final),
            "dropped_count": len(found.filtered.dropped),
            "reordered": found.reranked.reordered,
            "scores": [round(item.rerank_score, 4) for item in found.final],
            "chunks": [item.as_dict() for item in found.final],
            "dropped_by_filter": found.as_dict()["dropped_by_filter"],
            "expected_sources_retrieved": expected_found(enhanced, question.expected_sources),
            "irrelevant_chunks_sent": below(enhanced, threshold),
            "expected_words": mentions_expected(enhanced.answer, question.expected),
            "says_unknown": says_unknown(enhanced.answer),
            "rewrite_seconds": round(found.query.seconds, 3),
            "retrieval_seconds": round(found.retrieval.seconds, 3),
            "rerank_seconds": round(found.reranked.seconds, 3),
            "llm_seconds": round(enhanced.llm_seconds, 3),
            "total_seconds": round(enhanced.total_seconds, 3),
        },
    }
    logger.info(
        "        baseline: %d chunk(s), expected source %s | enhanced: %d -> %d -> %d, expected source %s",
        record["baseline"]["retrieved_count"],
        _verdict(record["baseline"], question),
        record["enhanced"]["retrieved_count"],
        record["enhanced"]["filtered_count"],
        record["enhanced"]["final_count"],
        _verdict(record["enhanced"], question),
    )

    return record


def _verdict(side: dict, question: Question) -> str:
    if not question.expected_sources:
        return "none expected"

    return "yes" if side["expected_sources_retrieved"] else "NO"


def metrics(records: list[dict], settings: RagSettings, seconds: float) -> dict:
    """Both modes, side by side, counting only what can be counted."""
    answerable = [record for record in records if record["expected_sources"]]
    unanswerable = [record for record in records if not record["expected_sources"]]

    return {
        "total_questions": len(records),
        "questions_with_expected_sources": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "threshold": settings.similarity_threshold,
        "source_retrieval_accuracy": {
            "baseline": _share(answerable, "baseline", "expected_sources_retrieved"),
            "enhanced": _share(answerable, "enhanced", "expected_sources_retrieved"),
        },
        "expected_information_found": {
            "off": _words(records, "off"),
            "baseline": _words(records, "baseline"),
            "enhanced": _words(records, "enhanced"),
        },
        "irrelevant_chunks_retrieved": sum(record["enhanced"]["dropped_count"] for record in records),
        "irrelevant_chunks_reaching_model": {
            "baseline": sum(record["baseline"]["irrelevant_chunks_sent"] for record in records),
            "enhanced": sum(record["enhanced"]["irrelevant_chunks_sent"] for record in records),
        },
        "questions_with_zero_relevant_chunks": sum(
            1 for record in records if record["enhanced"]["filtered_count"] == 0
        ),
        "unanswerable_handled": {
            "baseline": sum(1 for record in unanswerable if record["baseline"]["says_unknown"]),
            "enhanced": sum(1 for record in unanswerable if record["enhanced"]["says_unknown"]),
        },
        "average_retrieved_chunks": _mean(records, "enhanced", "retrieved_count"),
        "average_filtered_chunks": _mean(records, "enhanced", "filtered_count"),
        "average_final_chunks": _mean(records, "enhanced", "final_count"),
        "questions_reordered_by_reranker": sum(1 for record in records if record["enhanced"]["reordered"]),
        "average_rewrite_seconds": _mean(records, "enhanced", "rewrite_seconds"),
        "average_retrieval_seconds": {
            "baseline": _mean(records, "baseline", "retrieval_seconds"),
            "enhanced": _mean(records, "enhanced", "retrieval_seconds"),
        },
        "average_rerank_seconds": _mean(records, "enhanced", "rerank_seconds"),
        "average_llm_seconds": {
            "off": _mean(records, "off", "llm_seconds"),
            "baseline": _mean(records, "baseline", "llm_seconds"),
            "enhanced": _mean(records, "enhanced", "llm_seconds"),
        },
        "average_total_seconds": {
            "baseline": _mean(records, "baseline", "total_seconds"),
            "enhanced": _mean(records, "enhanced", "total_seconds"),
        },
        "comparison_seconds": round(seconds, 1),
    }


def _share(records: list[dict], side: str, key: str) -> float:
    if not records:
        return 0.0

    return round(sum(1 for record in records if record[side][key]) / len(records), 3)


def _words(records: list[dict], side: str, floor: float = 0.5) -> int:
    return sum(1 for record in records if record[side]["expected_words"] >= floor)


def _mean(records: list[dict], side: str, key: str) -> float:
    values = [record[side][key] for record in records]

    return round(sum(values) / len(values), 3) if values else 0.0


def report(document: dict) -> None:
    """Baseline against enhanced, question by question, as a person reads it."""
    for record in document["results"]:
        enhanced = record["enhanced"]
        baseline = record["baseline"]
        print("\n" + "=" * 78)
        print(f"Question #{record['id']}  ({record['category']})")
        print(record["question"])
        print(f"Expected sources: {', '.join(record['expected_sources']) or '(none - not in the index)'}")

        print("\n-- Baseline " + "-" * 65)
        print(f"Retrieved sources: {', '.join(baseline['sources']) or '-'}")
        print(f"Below threshold and sent anyway: {baseline['irrelevant_chunks_sent']}")
        print(f"Answer:\n  {_wrapped(baseline['answer'])}")

        print("\n-- Enhanced " + "-" * 65)
        print(f"Rewritten query: {enhanced['rewritten_query']}   [{enhanced['rewrite_used']}]")
        print(
            f"Top-K before filtering: {enhanced['retrieved_count']}"
            f"  ->  after threshold: {enhanced['filtered_count']}"
            f"  ->  after reranking: {enhanced['final_count']}"
        )

        for item in enhanced["chunks"]:
            print(
                f"  [{item['rerank_score']:.3f}] sim {item['similarity_score']:.3f} "
                f"kw {item['keyword_score']:.3f}  {item['file']} / {item['section'] or '-'}"
            )

        for dropped in enhanced["dropped_by_filter"]:
            print(f"  dropped [{dropped['similarity_score']:.3f}] {dropped['reference']}")

        print(f"Answer:\n  {_wrapped(enhanced['answer'])}")

    print("\n" + "=" * 78)
    print("Metrics")
    _print_metrics(document["metrics"])


def _print_metrics(metrics: dict, indent: int = 2) -> None:
    for name, value in metrics.items():
        if isinstance(value, dict):
            print(" " * indent + f"{name}")
            _print_metrics(value, indent + 2)
        else:
            print(" " * indent + f"{name:38} {value}")


def _wrapped(text: str, width: int = 74) -> str:
    import textwrap

    return "\n  ".join(textwrap.wrap(" ".join(text.split()), width=width)[:10])


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.compare_rag",
        description="Ask the ten control questions with no RAG, with Day 22 RAG and with the second stage.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="baseline top-k")
    parser.add_argument("--retrieval-top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--final-top-k", type=int, default=None)
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--only", type=int, nargs="*", help="run only these question ids")
    parser.add_argument("--results", default=str(RESULTS_FILE))

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    options = parse(argv)
    document = asyncio.run(compare(options))
    path = Path(options.results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    report(document)
    print(f"\nWritten: {path}")


if __name__ == "__main__":
    main()
