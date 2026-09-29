"""Day 22: the ten control questions, asked both ways.

    python -m app.evaluate_rag
    python -m app.evaluate_rag --top-k 3
    python -m app.evaluate_rag --only 10

Every question is asked twice - once with retrieval and once without - and
both answers are written down next to what was expected and what was
retrieved, in data/rag/evaluation_results.json.

About the numbers at the end: the only thing measured automatically is what
can be measured honestly. Whether the right document was retrieved is a
fact - the expected source either came back or it did not. Whether an answer
is *good* is not, so the checks here are deliberately crude and named for
what they actually test: whether the words the expectation hangs on appear,
and whether an answer admitted it did not know. The qualitative comparison
is left to a person reading the file, which is why both answers are kept in
full.
"""

import argparse
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from app.services.document_loader import PROJECT_ROOT
from app.services.rag_agent import RagAgent, RagAnswer
from app.services.rag_prompt import UNAVAILABLE
from app.services.rag_retriever import DEFAULT_STRATEGY, DEFAULT_TOP_K, RAGRetriever

logger = logging.getLogger("evaluate-rag")

RAG_DIR = PROJECT_ROOT / "data" / "rag"
QUESTIONS_FILE = RAG_DIR / "evaluation_questions.json"
RESULTS_FILE = RAG_DIR / "evaluation_results.json"

# Words that mean "I could not find it" in either language the answers may
# come back in. Used only to count, never to judge.
_UNKNOWN_MARKERS = (
    UNAVAILABLE,
    "not available in the indexed",
    "no information",
    "does not contain",
    "нет информации",
    "не содержится",
    "отсутствует",
    "не найдено",
)
_WORDS = re.compile(r"[\w/.\-]+", re.UNICODE)
# Words too common to prove anything about an answer.
_NOISE = frozenset(
    "the a an of to in and or is are it its that this for with as by on at from be "
    "и в на с по что это для из не а как the".split()
)


@dataclass(frozen=True)
class Question:
    id: int
    category: str
    question: str
    expected: str
    expected_sources: tuple[str, ...]


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    """The control set, as written by hand from the indexed documents."""
    if not path.exists():
        raise SystemExit(f"{path} is missing - the control questions are part of the repository.")

    document = json.loads(path.read_text(encoding="utf-8"))
    raw = document.get("questions")

    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"{path} holds no questions")

    return [
        Question(
            id=int(item["id"]),
            category=str(item.get("category", "")),
            question=str(item["question"]),
            expected=str(item.get("expected", "")),
            expected_sources=tuple(item.get("expected_sources") or ()),
        )
        for item in raw
    ]


def keywords(expected: str, limit: int = 12) -> list[str]:
    """The words an expectation stands on: long ones, file names, numbers."""
    found = [word.lower() for word in _WORDS.findall(expected)]
    picked = [
        word
        for word in found
        if word not in _NOISE and (len(word) > 4 or any(character.isdigit() for character in word))
    ]

    return list(dict.fromkeys(picked))[:limit]


def mentions_expected(answer: str, expected: str) -> float:
    """How much of the expectation's vocabulary the answer uses, 0 to 1.

    A blunt instrument, and named accordingly: it says whether an answer is
    talking about the same things, not whether it is right.
    """
    words = keywords(expected)

    if not words:
        return 0.0

    text = answer.lower()

    return round(sum(1 for word in words if word in text) / len(words), 3)


def says_unknown(answer: str) -> bool:
    text = answer.lower()

    return any(marker.lower() in text for marker in _UNKNOWN_MARKERS)


def retrieved_expected(answer: RagAnswer, expected_sources: tuple[str, ...]) -> list[str]:
    """Which of the expected documents actually came back."""
    files = {chunk.file for chunk in answer.retrieved_chunks}

    return [source for source in expected_sources if any(source in file for file in files)]


async def evaluate(options: argparse.Namespace) -> dict:
    questions = load_questions()

    if options.only:
        questions = [question for question in questions if question.id in options.only]

    agent = RagAgent(retriever=RAGRetriever(strategy=options.strategy), top_k=options.top_k)
    logger.info("Questions: %d", len(questions))
    logger.info("Index: %s", agent.retriever.directory)
    logger.info("")

    records = []
    started = time.perf_counter()

    for question in questions:
        logger.info("[%2d/%d] %s", question.id, len(questions), question.question)

        without = await agent.ask(question.question, use_rag=False)
        with_rag = await agent.ask(question.question, use_rag=True)
        found = retrieved_expected(with_rag, question.expected_sources)
        record = {
            "id": question.id,
            "category": question.category,
            "question": question.question,
            "expected": question.expected,
            "expected_sources": list(question.expected_sources),
            "answer_without_rag": without.answer,
            "answer_with_rag": with_rag.answer,
            "retrieved_sources": with_rag.sources,
            "retrieved_files": [chunk.file for chunk in with_rag.retrieved_chunks],
            "retrieved_chunks": [chunk.as_dict(with_text=False) for chunk in with_rag.retrieved_chunks],
            "similarity_scores": [round(chunk.score, 4) for chunk in with_rag.retrieved_chunks],
            "expected_sources_retrieved": found,
            "expected_sources_missing": [
                source for source in question.expected_sources if source not in found
            ],
            "expected_words_in_rag_answer": mentions_expected(with_rag.answer, question.expected),
            "expected_words_in_plain_answer": mentions_expected(without.answer, question.expected),
            "rag_answer_says_unknown": says_unknown(with_rag.answer),
            "plain_answer_says_unknown": says_unknown(without.answer),
            "latency": {
                "retrieval_seconds": round(with_rag.retrieval_seconds, 3),
                "llm_seconds_with_rag": round(with_rag.llm_seconds, 3),
                "llm_seconds_without_rag": round(without.llm_seconds, 3),
                "total_seconds_with_rag": round(with_rag.total_seconds, 3),
            },
            "tokens": {
                "with_rag": with_rag.tokens_used,
                "without_rag": without.tokens_used,
            },
        }
        records.append(record)
        logger.info(
            "        retrieved: %s | expected source found: %s",
            ", ".join(record["retrieved_files"][:3]) or "-",
            "yes" if found or not question.expected_sources else "NO",
        )

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "settings": {
            "top_k": options.top_k,
            "strategy": options.strategy,
            "index": str(agent.retriever.directory),
            "embedding_model": agent.retriever.embeddings().name,
        },
        "metrics": metrics(records, time.perf_counter() - started),
        "results": records,
    }


def metrics(records: list[dict], seconds: float) -> dict:
    """What can be counted without pretending to judge."""
    answerable = [record for record in records if record["expected_sources"]]
    unanswerable = [record for record in records if not record["expected_sources"]]
    chunks = [len(record["retrieved_chunks"]) for record in records]
    scores = [score for record in records for score in record["similarity_scores"]]

    return {
        "total_questions": len(records),
        "questions_with_expected_sources": len(answerable),
        "expected_source_retrieved": sum(1 for record in answerable if record["expected_sources_retrieved"]),
        "all_expected_sources_retrieved": sum(
            1 for record in answerable if not record["expected_sources_missing"]
        ),
        "average_retrieved_chunks": round(sum(chunks) / len(chunks), 2) if chunks else 0,
        "average_similarity": round(sum(scores) / len(scores), 4) if scores else 0,
        "best_similarity": round(max(scores), 4) if scores else 0,
        "worst_similarity": round(min(scores), 4) if scores else 0,
        "rag_answers_using_expected_words": sum(
            1 for record in records if record["expected_words_in_rag_answer"] >= 0.5
        ),
        "plain_answers_using_expected_words": sum(
            1 for record in records if record["expected_words_in_plain_answer"] >= 0.5
        ),
        "rag_answers_saying_unknown": sum(1 for record in records if record["rag_answer_says_unknown"]),
        "plain_answers_saying_unknown": sum(1 for record in records if record["plain_answer_says_unknown"]),
        "unanswerable_questions": len(unanswerable),
        "unanswerable_handled_by_rag": sum(1 for record in unanswerable if record["rag_answer_says_unknown"]),
        "average_retrieval_seconds": _mean(records, "retrieval_seconds"),
        "average_llm_seconds_with_rag": _mean(records, "llm_seconds_with_rag"),
        "average_llm_seconds_without_rag": _mean(records, "llm_seconds_without_rag"),
        "average_total_seconds_with_rag": _mean(records, "total_seconds_with_rag"),
        "evaluation_seconds": round(seconds, 1),
    }


def _mean(records: list[dict], key: str) -> float:
    values = [record["latency"][key] for record in records]

    return round(sum(values) / len(values), 3) if values else 0.0


def report(document: dict) -> None:
    """The comparison, question by question, as a person reads it."""
    for record in document["results"]:
        print("\n" + "=" * 78)
        print(f"Question #{record['id']}  ({record['category']})")
        print(record["question"])
        print(f"\nExpected:\n  {record['expected']}")
        print(f"\nWithout RAG:\n  {_wrapped(record['answer_without_rag'])}")
        print(f"\nWith RAG:\n  {_wrapped(record['answer_with_rag'])}")
        print("\nRetrieved sources:")

        for source, score in zip(record["retrieved_sources"], record["similarity_scores"]):
            print(f"  [{score:.3f}] {source}")

        expected = record["expected_sources"]
        found = record["expected_sources_retrieved"]
        print(
            "\nResult: expected source(s) "
            + (
                "none expected (the answer is not in the index)"
                if not expected
                else f"{len(found)}/{len(expected)} retrieved" + (f" - missing {record['expected_sources_missing']}" if record["expected_sources_missing"] else "")
            )
        )
        print(
            f"        expected wording present: RAG {record['expected_words_in_rag_answer']:.0%}"
            f" · no-RAG {record['expected_words_in_plain_answer']:.0%}"
            + ("  · RAG said it does not know" if record["rag_answer_says_unknown"] else "")
        )

    print("\n" + "=" * 78)
    print("Metrics")

    for name, value in document["metrics"].items():
        print(f"  {name:38} {value}")


def _wrapped(text: str, width: int = 74) -> str:
    import textwrap

    return "\n  ".join(textwrap.wrap(" ".join(text.split()), width=width)[:12])


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluate_rag",
        description="Ask the ten control questions with and without retrieval, and write down both answers.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
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
    document = asyncio.run(evaluate(options))
    path = Path(options.results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    report(document)
    print(f"\nWritten: {path}")


if __name__ == "__main__":
    main()
