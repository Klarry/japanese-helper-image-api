"""Day 28: the same ten questions, the same index, two different generators.

    python -m app.evaluate_day28
    python -m app.evaluate_day28 --providers ollama
    python -m app.evaluate_day28 --answer-threshold 0.60 --only 1 2 3

Every question runs twice. Retrieval, filtering, reranking, the relevance
gate and the citation validation are the same code in both runs - the only
thing that changes is who is handed the finished prompt. That is what makes
the comparison worth anything: the difference in the numbers is the
difference between the two models, not between two pipelines.

Nothing here is estimated or filled in. A run that cannot reach a provider
records the error and carries on to the next question, and the summary only
counts what actually happened.
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.evaluate_rag import Question, load_questions
from app.services import llm_provider
from app.services.document_loader import PROJECT_ROOT
from app.services.llm_provider import GEMINI, OLLAMA, PROVIDERS
from app.services.rag_agent import ENHANCED, RagAgent
from app.services.rag_citations import flatten
from app.services.rag_response import ANSWERED, INSUFFICIENT
from app.services.rag_retriever import DEFAULT_STRATEGY, RAGRetriever
from app.services.rag_settings import DEFAULT_SETTINGS

logger = logging.getLogger("evaluate-day28")

RESULTS_FILE = PROJECT_ROOT / "data" / "evaluation" / "day28_results.json"
RULE = "=" * 62


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))


@dataclass
class Run:
    """One question, one provider, one attempt."""

    question_id: int
    question: str
    provider: str
    model: str = ""
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    rag_status: str = ""
    confidence: str = ""
    retrieval_ms: int = 0
    generation_ms: int = 0
    total_ms: int = 0
    retrieved_count: int = 0
    final_count: int = 0
    best_relevance: float = 0.0
    #: Of the citations that survived Day 24's validation, how many still
    #: hold when checked again here, from the outside.
    valid_citations: int = 0
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "provider": self.provider,
            "model": self.model,
            "answer": self.answer,
            "sources": self.sources,
            "citations": self.citations,
            "rag_status": self.rag_status,
            "confidence": self.confidence,
            "retrieval_ms": self.retrieval_ms,
            "generation_ms": self.generation_ms,
            "total_ms": self.total_ms,
            "retrieved_count": self.retrieved_count,
            "final_count": self.final_count,
            "best_relevance": round(self.best_relevance, 4),
            "valid_citations": self.valid_citations,
            "error": self.error,
        }


def recheck(answer, run: Run) -> int:
    """Validate the surviving citations a second time, independently.

    Day 24 already threw away the quotes that were not in their chunk. This
    checks the ones that were kept, against the chunks that were really
    retrieved - a number the pipeline cannot inflate, because it is
    computed here rather than reported by it.
    """
    if answer.enhanced is None:
        return 0

    by_id = {chunk.chunk_id: chunk.text for chunk in answer.enhanced.chunks}
    held = 0

    for citation in answer.cited.citations if answer.cited else ():
        text = by_id.get(citation.chunk_id, "")

        if text and flatten(citation.quote) in flatten(text):
            held += 1

    return held


async def ask(question: Question, provider_name: str, options) -> Run:
    """One question through the whole pipeline, with one generator."""
    run = Run(question_id=question.id, question=question.question, provider=provider_name)
    settings = DEFAULT_SETTINGS.with_overrides(answer_threshold=options.answer_threshold)
    agent = RagAgent(
        retriever=options.retriever,
        settings=settings,
        provider=llm_provider.provider_for(provider_name),
    )
    run.model = agent.model_name
    started = time.perf_counter()

    try:
        answer = await agent.ask(question.question, mode=ENHANCED, settings=settings)
    except llm_provider.ProviderUnavailable as error:
        # Recorded, not retried elsewhere. A provider that cannot answer is
        # a result about that provider.
        run.error = error.message.splitlines()[0]
        run.total_ms = _ms(time.perf_counter() - started)
        logger.warning("   %-7s error: %s", provider_name, run.error)

        return run
    except Exception as error:  # noqa: BLE001 - one bad question must not end the run
        run.error = f"{type(error).__name__}: {error}"
        run.total_ms = _ms(time.perf_counter() - started)
        logger.warning("   %-7s error: %s", provider_name, run.error)

        return run

    total = time.perf_counter() - started
    cited = answer.cited
    enhanced = answer.enhanced
    run.answer = answer.answer
    run.rag_status = cited.rag_status if cited else ""
    run.confidence = cited.confidence if cited else ""
    run.sources = [source.as_dict() for source in (cited.sources if cited else ())]
    run.citations = [citation.as_dict() for citation in (cited.citations if cited else ())]
    run.retrieval_ms = _ms(answer.retrieval_seconds)
    run.generation_ms = _ms(answer.llm_seconds)
    run.total_ms = _ms(total)
    run.retrieved_count = len(enhanced.retrieval.chunks) if enhanced else 0
    run.final_count = len(enhanced.final) if enhanced else 0
    run.best_relevance = cited.best_relevance if cited else 0.0
    run.valid_citations = recheck(answer, run)

    logger.info(
        "   %-7s %-20s retrieval %5dms · generation %6dms · total %6dms · "
        "chunks %d/%d · citations %d",
        provider_name,
        run.rag_status,
        run.retrieval_ms,
        run.generation_ms,
        run.total_ms,
        run.final_count,
        run.retrieved_count,
        run.valid_citations,
    )

    return run


def summarise(runs: list[Run], provider_name: str) -> dict:
    """What a provider did over the whole set.

    Averages are over the runs that produced an answer: a provider that
    failed on half the questions should not get a flattering mean out of
    the half it skipped, and ``errors`` is reported beside it so the
    omission is visible.
    """
    mine = [run for run in runs if run.provider == provider_name]
    done = [run for run in mine if not run.error]
    answered = [run for run in done if run.rag_status == ANSWERED]
    cited = [run for run in done if run.citations]

    def mean(values: list[int]) -> int:
        return int(round(sum(values) / len(values))) if values else 0

    return {
        "provider": provider_name,
        "model": mine[0].model if mine else "",
        "questions": len(mine),
        "average_retrieval_ms": mean([run.retrieval_ms for run in done]),
        "average_generation_ms": mean([run.generation_ms for run in done]),
        "average_total_ms": mean([run.total_ms for run in done]),
        "answered": len(answered),
        "insufficient_context": len([run for run in done if run.rag_status == INSUFFICIENT]),
        "errors": len(mine) - len(done),
        "answers_with_citations": len(cited),
        "citations_total": sum(len(run.citations) for run in done),
        "citations_valid": sum(run.valid_citations for run in done),
        "citation_validity": round(
            sum(run.valid_citations for run in done)
            / max(1, sum(len(run.citations) for run in done)),
            4,
        ),
        "slowest_total_ms": max([run.total_ms for run in done], default=0),
        "fastest_total_ms": min([run.total_ms for run in done], default=0),
    }


def report(summaries: list[dict]) -> None:
    print(f"\n{RULE}\n=== Summary ===\n{RULE}")

    for summary in summaries:
        print(f"\n{summary['provider']}  ({summary['model']})")
        print(f"  average retrieval:  {summary['average_retrieval_ms']:>7} ms")
        print(f"  average generation: {summary['average_generation_ms']:>7} ms")
        print(f"  average total:      {summary['average_total_ms']:>7} ms")
        print(f"  fastest / slowest:  {summary['fastest_total_ms']} / {summary['slowest_total_ms']} ms")
        print(f"  answered:           {summary['answered']}/{summary['questions']}")
        print(f"  insufficient:       {summary['insufficient_context']}")
        print(f"  errors:             {summary['errors']}")
        print(
            f"  citation validity:  {summary['citations_valid']}/{summary['citations_total']}"
            f"  ({summary['citation_validity']:.0%})"
        )


async def run(options) -> int:
    questions = load_questions()

    if options.only:
        questions = [question for question in questions if question.id in set(options.only)]

    if not questions:
        raise SystemExit("no questions selected")

    options.retriever = RAGRetriever(strategy=options.strategy)
    print(f"\n{RULE}\n=== Day 28 RAG Evaluation ===\n{RULE}")
    print(
        f"{len(questions)} question(s) · providers: {', '.join(options.providers)} · "
        f"threshold {options.answer_threshold}"
    )
    started = time.perf_counter()
    runs: list[Run] = []

    for question in questions:
        print(f"\nQuestion {question.id}: {question.question}")

        for provider_name in options.providers:
            runs.append(await ask(question, provider_name, options))

    seconds = time.perf_counter() - started
    summaries = [summarise(runs, name) for name in options.providers]
    report(summaries)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "strategy": options.strategy,
                "answer_threshold": options.answer_threshold,
                "providers": list(options.providers),
                "questions": len(questions),
                "evaluation_seconds": round(seconds, 1),
                "summary": {summary["provider"]: summary for summary in summaries},
                "results": [item.as_dict() for item in runs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWritten: {RESULTS_FILE}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluate_day28",
        description="The ten control questions, answered through the same index by both providers.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=list(PROVIDERS),
        choices=list(PROVIDERS),
        help=f"which generators to run (default: {' '.join(PROVIDERS)})",
    )
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument(
        "--answer-threshold",
        type=float,
        default=DEFAULT_SETTINGS.answer_threshold,
        help="how relevant the best chunk must be before any model is asked",
    )
    parser.add_argument("--only", nargs="*", type=int, default=[], help="run only these question ids")

    return parser.parse_args(argv)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
