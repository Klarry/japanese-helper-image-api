"""Day 22: ask the index a question, from the command line.

    python -m app.rag "What is the architecture of the project?"
    python -m app.rag --no-rag "What is the architecture of the project?"
    python -m app.rag --top-k 3 --strategy fixed-size "How is a chunk id built?"

With retrieval it prints what it found before it prints the answer, because
the whole point of the exercise is being able to see where an answer came
from.
"""

import argparse
import asyncio
import logging

from app.services.chunking import FIXED, STRUCTURAL
from app.services.rag_agent import RagAgent
from app.services.rag_retriever import DEFAULT_STRATEGY, DEFAULT_TOP_K, RAGRetriever

logger = logging.getLogger("rag")


def show(answer) -> None:
    print(f"\nQuestion:\n{answer.question}\n")

    if answer.rag_enabled:
        print("Retrieved chunks:")

        for number, chunk in enumerate(answer.retrieved_chunks, start=1):
            print(f"  {number}. [{chunk.score:.3f}] {chunk.reference}")
            print(f"     {chunk.text.strip()[:120].replace(chr(10), ' ')}…")

        print("\nSources:")

        for source in answer.sources:
            print(f"  - {source}")
    else:
        print("Retrieval: off")

    print(f"\nAnswer:\n{answer.answer}\n")
    print(
        f"[retrieval {answer.retrieval_seconds:.3f}s · model {answer.llm_seconds:.3f}s · "
        f"{answer.tokens_used} tokens]"
    )


async def run(options: argparse.Namespace) -> None:
    agent = RagAgent(retriever=RAGRetriever(strategy=options.strategy), top_k=options.top_k)
    answer = await agent.ask(options.question, use_rag=not options.no_rag)
    show(answer)


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.rag",
        description="Answer a question about this project, with or without the local document index.",
    )
    parser.add_argument("question")
    parser.add_argument("--no-rag", action="store_true", help="answer without retrieval")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--strategy", choices=(STRUCTURAL, FIXED), default=DEFAULT_STRATEGY)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(parse(argv)))


if __name__ == "__main__":
    main()
