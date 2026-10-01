"""Day 24: the ten control questions, checked for evidence rather than prose.

    python -m app.evaluate_citations
    python -m app.evaluate_citations --answer-threshold 0.65
    python -m app.evaluate_citations --only 2 10

Every question goes through Enhanced RAG, and what comes back is checked on
the nine things the assignment names: is there an answer, are there sources,
are there citations, does each citation point at a chunk that was really
retrieved, is each quote really in that chunk, do the answer's claims stand
on the quotes, was the expected source retrieved, how relevant was the best
chunk, and was the question refused for want of context.

Two files are written. ``day24_results.json`` keeps every answer in full
with its evidence and everything that was rejected; ``day24_comparison.json``
is the count. Nothing here is estimated: a citation either holds or it does
not, and the summary only adds up what the per-question checks decided.
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from app.evaluate_rag import Question, load_questions
from app.services.document_loader import PROJECT_ROOT
from app.services.rag_agent import ENHANCED, RagAgent, RagAnswer
from app.services.rag_citations import flatten
from app.services.rag_enhanced import EnhancedRetriever
from app.services.rag_response import ANSWERED, INSUFFICIENT, SUPPORTED, UNSUPPORTED
from app.services.rag_retriever import DEFAULT_STRATEGY, RAGRetriever
from app.services.rag_settings import DEFAULT_SETTINGS, RagSettings

logger = logging.getLogger("evaluate-citations")

RAG_DIR = PROJECT_ROOT / "data" / "rag"
RESULTS_FILE = RAG_DIR / "day24_results.json"
COMPARISON_FILE = RAG_DIR / "day24_comparison.json"


def recheck(answer: RagAnswer) -> dict:
    """Validate the citations a second time, from the outside.

    The agent already checked them; doing it again here, from the answer
    rather than from inside the flow, is the point of an evaluation. If the
    two ever disagreed, this file would say so.
    """
    cited = answer.cited
    chunks = {chunk.chunk_id: chunk for chunk in (answer.enhanced.chunks if answer.enhanced else ())}
    checks = []

    for citation in cited.citations if cited else ():
        chunk = chunks.get(citation.chunk_id)
        checks.append(
            {
                "chunk_id": citation.chunk_id,
                "quote": citation.quote,
                "chunk_exists": chunk is not None,
                "chunk_was_retrieved": chunk is not None,
                "quote_in_chunk": bool(chunk and flatten(citation.quote) in flatten(chunk.text)),
            }
        )

    return {
        "citations": checks,
        "valid": sum(1 for item in checks if item["quote_in_chunk"] and item["chunk_exists"]),
        "invalid": sum(1 for item in checks if not (item["quote_in_chunk"] and item["chunk_exists"])),
    }


def expected_found(answer: RagAnswer, expected_sources: tuple[str, ...]) -> list[str]:
    """Which expected documents reached the model.

    Measured against everything the second stage put in the prompt, not only
    the chunks a citation ended up pointing at: retrieval did its job if the
    document was there to be used.
    """
    files = {chunk.file for chunk in (answer.enhanced.chunks if answer.enhanced else ())}

    return [source for source in expected_sources if any(source in file for file in files)]


async def one(agent: RagAgent, question: Question, settings: RagSettings) -> dict:
    """One question, and the nine checks on what came back."""
    answer = await agent.ask(question.question, mode=ENHANCED, settings=settings)
    cited = answer.cited
    validation = recheck(answer)
    found = expected_found(answer, question.expected_sources)
    record = {
        "id": question.id,
        "category": question.category,
        "question": question.question,
        "expected": question.expected,
        "expected_sources": list(question.expected_sources),
        "response": cited.as_record() if cited else {},
        "checks": {
            "has_answer": bool(answer.answer.strip()),
            "has_sources": bool(cited and cited.sources),
            "has_citations": bool(cited and cited.citations),
            "citations_point_at_retrieved_chunks": all(
                item["chunk_was_retrieved"] for item in validation["citations"]
            ),
            "quotes_are_in_their_chunks": all(
                item["quote_in_chunk"] for item in validation["citations"]
            ),
            "citation_support": cited.citation_support if cited else "not_checked",
            "expected_source_retrieved": bool(found) if question.expected_sources else None,
            "best_relevance": round(cited.best_relevance, 4) if cited else 0.0,
            "insufficient_context": bool(cited and cited.rag_status == INSUFFICIENT),
        },
        "validation": validation,
        "expected_sources_retrieved": found,
        "retrieval": answer.enhanced.as_dict() if answer.enhanced else {},
        "latency": {
            "retrieval_seconds": round(answer.retrieval_seconds, 3),
            "llm_seconds": round(answer.llm_seconds, 3),
            "total_seconds": round(answer.total_seconds, 3),
        },
        "tokens_used": answer.tokens_used,
    }
    logger.info(
        "        status=%s · sources=%d · citations=%d (%d valid, %d rejected) · support=%s · best=%.3f",
        record["response"].get("rag_status", "-"),
        len(record["response"].get("sources", [])),
        len(record["response"].get("citations", [])),
        validation["valid"],
        len(record["response"].get("rejected_citations", [])),
        record["checks"]["citation_support"],
        record["checks"]["best_relevance"],
    )

    return record


async def evaluate(options: argparse.Namespace) -> dict:
    questions = load_questions()

    if options.only:
        questions = [question for question in questions if question.id in options.only]

    settings = DEFAULT_SETTINGS.with_overrides(
        retrieval_top_k=options.retrieval_top_k,
        similarity_threshold=options.threshold,
        final_top_k=options.final_top_k,
        answer_threshold=options.answer_threshold,
        query_rewrite=False if options.no_rewrite else None,
    )
    retriever = RAGRetriever(strategy=options.strategy)
    agent = RagAgent(
        retriever=retriever,
        settings=settings,
        enhanced=EnhancedRetriever(retriever=retriever, settings=settings),
        check_support=not options.no_support_model,
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
            "strategy": options.strategy,
            "index": str(retriever.directory),
            "embedding_model": retriever.embeddings().name,
        },
        "metrics": metrics(records, time.perf_counter() - started),
        "results": records,
    }


def metrics(records: list[dict], seconds: float) -> dict:
    """The count, and only the count.

    Every number is a tally of a per-question check that already happened.
    Nothing here re-decides anything, which is why the two files can never
    disagree.
    """
    total = len(records)
    answerable = [record for record in records if record["expected_sources"]]
    answered = [record for record in records if record["response"].get("rag_status") == ANSWERED]
    citations = sum(len(record["response"].get("citations", [])) for record in records)
    valid = sum(record["validation"]["valid"] for record in records)
    invalid = sum(record["validation"]["invalid"] for record in records)
    rejected = sum(len(record["response"].get("rejected_citations", [])) for record in records)

    return {
        "total_questions": total,
        "answered": len(answered),
        "insufficient_context": sum(1 for record in records if record["checks"]["insufficient_context"]),
        "answers_with_sources": sum(1 for record in records if record["checks"]["has_sources"]),
        "answers_with_citations": sum(1 for record in records if record["checks"]["has_citations"]),
        "citations_total": citations,
        "valid_citations": valid,
        "invalid_citations": invalid,
        "citations_rejected_before_answering": rejected,
        "supported_answers": sum(
            1 for record in records if record["checks"]["citation_support"] == SUPPORTED
        ),
        "unsupported_answers": sum(
            1 for record in records if record["checks"]["citation_support"] == UNSUPPORTED
        ),
        "support_not_checked": sum(
            1 for record in records if record["checks"]["citation_support"] == "not_checked"
        ),
        "questions_with_expected_sources": len(answerable),
        "expected_source_retrieved": sum(
            1 for record in answerable if record["checks"]["expected_source_retrieved"]
        ),
        "expected_source_retrieval_rate": (
            round(
                sum(1 for record in answerable if record["checks"]["expected_source_retrieved"])
                / len(answerable),
                3,
            )
            if answerable
            else 0.0
        ),
        "confidence": {
            level: sum(1 for record in records if record["response"].get("confidence") == level)
            for level in ("high", "medium", "low")
        },
        "average_best_relevance": (
            round(sum(record["checks"]["best_relevance"] for record in records) / total, 4)
            if total
            else 0.0
        ),
        "evaluation_seconds": round(seconds, 1),
    }


def report(document: dict) -> None:
    """The evidence, question by question, as a person reads it."""
    for record in document["results"]:
        response = record["response"]
        print("\n" + "=" * 78)
        print(f"Question #{record['id']}  ({record['category']})")
        print(record["question"])
        print(f"\nStatus: {response.get('rag_status')} · confidence: {response.get('confidence')}"
              f" · support: {response.get('citation_support')}"
              f" · best relevance: {record['checks']['best_relevance']}")
        print(f"\nAnswer:\n  {_wrapped(response.get('answer', ''))}")

        print("\nSources:")

        for number, source in enumerate(response.get("sources", []), start=1):
            print(f"  {number}. {source['file']} / {source['section'] or '-'} / {source['chunk_id']}")

        if not response.get("sources"):
            print("  None")

        print("\nCitations:")

        for citation in response.get("citations", []):
            print(f'  > "{citation["quote"]}"')
            print(f"    - {citation['source']} / {citation['chunk_id']}")

        if not response.get("citations"):
            print("  None")

        for item in response.get("rejected_citations", []):
            print(f"  x rejected ({item['reason']}): {item['quote'][:70]}")

    print("\n" + "=" * 78)
    print("Metrics")
    _print(document["metrics"])


def _print(values: dict, indent: int = 2) -> None:
    for name, value in values.items():
        if isinstance(value, dict):
            print(" " * indent + f"{name}")
            _print(value, indent + 2)
        else:
            print(" " * indent + f"{name:38} {value}")


def _wrapped(text: str, width: int = 74) -> str:
    import textwrap

    return "\n  ".join(textwrap.wrap(" ".join(text.split()), width=width)[:10])


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluate_citations",
        description="Ask the ten control questions with Enhanced RAG and check the evidence.",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="similarity filter")
    parser.add_argument("--final-top-k", type=int, default=None)
    parser.add_argument("--answer-threshold", type=float, default=None)
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument(
        "--no-support-model",
        action="store_true",
        help="check citation support on words only, without the extra model call",
    )
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--only", type=int, nargs="*", help="run only these question ids")
    parser.add_argument("--results", default=str(RESULTS_FILE))
    parser.add_argument("--comparison", default=str(COMPARISON_FILE))

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    options = parse(argv)
    document = asyncio.run(evaluate(options))

    results = Path(options.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison = Path(options.comparison)
    comparison.write_text(
        json.dumps(
            {
                "created_at": document["created_at"],
                "settings": document["settings"],
                "metrics": document["metrics"],
                "per_question": [
                    {"id": record["id"], "category": record["category"], **record["checks"]}
                    for record in document["results"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report(document)
    print(f"\nWritten: {results}")
    print(f"Written: {comparison}")


if __name__ == "__main__":
    main()
