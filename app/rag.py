"""Days 22-23: ask the index a question, from the command line.

    python -m app.rag "What is the architecture of the project?"
    python -m app.rag --mode off "What is the architecture of the project?"
    python -m app.rag --mode enhanced "Как работает MCP в проекте?"
    python -m app.rag --mode enhanced --threshold 0.75 --final-top-k 5 "..."

Three modes, one command, because the only way to see what the second stage
is worth is to run the same question through both and read the two outputs
next to each other. Enhanced prints the whole funnel - what the search was
actually given, how many chunks came back, how many survived the threshold,
and what the reranker did with the rest - since a pipeline that only prints
its answer cannot be argued with.
"""

import argparse
import asyncio
import logging

from app.services.chunking import FIXED, STRUCTURAL
from app.services.rag_agent import BASELINE, ENHANCED, MODES, OFF, RagAgent
from app.services.rag_enhanced import EnhancedRetriever
from app.services.rag_retriever import DEFAULT_STRATEGY, DEFAULT_TOP_K, RAGRetriever
from app.services.rag_settings import DEFAULT_SETTINGS

logger = logging.getLogger("rag")


def show_enhanced(answer) -> None:
    found = answer.enhanced
    settings = found.settings

    print(f"\nOriginal query:\n  {found.query.original}")
    print(f"\nRewritten query:\n  {found.query.query}   [{found.query.used}]")
    print(f"\nRetrieved:\n  {len(found.retrieval.chunks)} (retrieval_top_k={settings.retrieval_top_k})")

    for number, chunk in enumerate(found.retrieval.chunks, start=1):
        mark = " " if chunk.score >= settings.similarity_threshold else "x"
        print(f"  {mark} {number:2}. [{chunk.score:.3f}] {chunk.reference}")

    print(
        f"\nAfter filtering:\n  {len(found.filtered.kept)} "
        f"(threshold={settings.similarity_threshold}, dropped {len(found.filtered.dropped)})"
    )
    print(
        f"\nAfter reranking:\n  {len(found.final)} (final_top_k={settings.final_top_k}, "
        f"weights {settings.similarity_weight}/{settings.keyword_weight}, "
        f"order {'changed' if found.reranked.reordered else 'unchanged'})"
    )

    for number, item in enumerate(found.final, start=1):
        print(
            f"  {number}. rerank {item.rerank_score:.3f} "
            f"= {settings.similarity_weight}·sim {item.similarity_score:.3f} "
            f"+ {settings.keyword_weight}·kw {item.keyword_score:.3f}   {item.reference}"
        )
        print(f"     matched: {', '.join(item.matched) or '-'}")

    if found.nothing_relevant:
        print("\n  Nothing cleared the threshold - the model was not asked.")


def show(answer) -> None:
    print(f"\nQuestion:\n{answer.question}\n")
    print(f"Mode: {answer.mode}")

    if answer.mode == ENHANCED:
        show_enhanced(answer)
    elif answer.mode == BASELINE:
        print("\nRetrieved chunks:")

        for number, chunk in enumerate(answer.retrieved_chunks, start=1):
            print(f"  {number}. [{chunk.score:.3f}] {chunk.reference}")
            print(f"     {chunk.text.strip()[:120].replace(chr(10), ' ')}…")
    else:
        print("Retrieval: off")

    if answer.sources:
        print("\nSources:")

        for source in answer.sources:
            print(f"  - {source}")

    print(f"\nAnswer:\n{answer.answer}\n")
    print(
        f"[retrieval {answer.retrieval_seconds:.3f}s · model {answer.llm_seconds:.3f}s · "
        f"{answer.tokens_used} tokens]"
    )


def settings_from(options: argparse.Namespace):
    """The defaults, with whatever the command line overrode."""
    return DEFAULT_SETTINGS.with_overrides(
        retrieval_top_k=options.retrieval_top_k,
        similarity_threshold=options.threshold,
        final_top_k=options.final_top_k,
        query_rewrite=False if options.no_rewrite else None,
    )


async def run(options: argparse.Namespace) -> None:
    retriever = RAGRetriever(strategy=options.strategy)
    settings = settings_from(options)
    agent = RagAgent(
        retriever=retriever,
        top_k=options.top_k,
        settings=settings,
        enhanced=EnhancedRetriever(retriever=retriever, settings=settings),
    )
    answer = await agent.ask(options.question, mode=options.mode)
    show(answer)


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.rag",
        description="Answer a question about this project, with or without the local document index.",
    )
    parser.add_argument("question")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=BASELINE,
        help="off: no retrieval. baseline: Day 22. enhanced: rewrite, filter and rerank.",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="the same as --mode off (kept from Day 22)",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="baseline only")
    parser.add_argument("--retrieval-top-k", type=int, default=None, help="enhanced: before filtering")
    parser.add_argument("--threshold", type=float, default=None, help="enhanced: similarity cut-off")
    parser.add_argument("--final-top-k", type=int, default=None, help="enhanced: after reranking")
    parser.add_argument("--no-rewrite", action="store_true", help="enhanced: search with the question as asked")
    parser.add_argument("--strategy", choices=(STRUCTURAL, FIXED), default=DEFAULT_STRATEGY)
    options = parser.parse_args(argv)

    if options.no_rag:
        options.mode = OFF

    return options


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(parse(argv)))


if __name__ == "__main__":
    main()
