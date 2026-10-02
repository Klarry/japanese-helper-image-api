"""Day 25: the two long conversations, run and checked.

    python -m app.evaluate_chat
    python -m app.evaluate_chat --only 2
    python -m app.evaluate_chat --answer-threshold 0.60

Each scenario is a real conversation on a throwaway history file: twelve to
fourteen messages that settle a goal, pin constraints and terms, wander off
into the project's documents, change a constraint, and come back to ask what
was decided. What is checked at the end is the thing a chat is actually for -
whether it still knows.

Eight checks per scenario, and they are all facts rather than impressions:
did the goal survive to the last turn, are the constraints still there, the
terms, the decisions; did retrieval run on every single question; did every
answer report its sources; is every source a chunk the index really holds;
and does the closing answer repeat what the memory says rather than the last
thing that was mentioned.

The last one is the only one worth arguing about, so it is kept crude on
purpose: it looks for the constraints' own words in the final answer. That
cannot prove understanding - it can only catch an assistant that has stopped
mentioning them, which is exactly the failure this day is about.
"""

import argparse
import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path

from app.services.chat_session import ChatSession, ChatTurn
from app.services.document_loader import PROJECT_ROOT
from app.services.rag_enhanced import EnhancedRetriever
from app.services.rag_retriever import DEFAULT_STRATEGY, RAGRetriever
from app.services.rag_settings import DEFAULT_SETTINGS

logger = logging.getLogger("evaluate-chat")

CHAT_DIR = PROJECT_ROOT / "data" / "chat"
SCENARIOS_FILE = CHAT_DIR / "day25_scenarios.json"
RESULTS_FILE = CHAT_DIR / "day25_results.json"

#: Asked at the end of every scenario, whatever its own messages were. The
#: answers have to come from task memory: no document says what this
#: conversation decided.
FINAL_QUESTIONS = (
    "What is our current goal?",
    "What constraints and decisions have we confirmed?",
)


def load_scenarios(path: Path = SCENARIOS_FILE) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} is missing - the scenarios are part of the repository.")

    document = json.loads(path.read_text(encoding="utf-8"))
    scenarios = document.get("scenarios")

    if not isinstance(scenarios, list) or not scenarios:
        raise SystemExit(f"{path} holds no scenarios")

    return scenarios


def mentions(text: str, wanted: list[str]) -> list[str]:
    """Which of these words the text actually uses. Case-insensitive, and
    nothing cleverer: it is a presence check, not a judgement."""
    lowered = text.lower()

    return [word for word in wanted if word.lower() in lowered]


def kept(entries, wanted: list[str]) -> list[str]:
    """Which expectations survive somewhere in a memory list."""
    joined = " | ".join(entries).lower()

    return [word for word in wanted if word.lower() in joined]


async def run_scenario(scenario: dict, options: argparse.Namespace) -> dict:
    """One conversation, start to finish, on its own history file."""
    settings = DEFAULT_SETTINGS.with_overrides(
        answer_threshold=options.answer_threshold,
        similarity_threshold=options.threshold,
        final_top_k=options.final_top_k,
    )
    retriever = RAGRetriever(strategy=options.strategy)

    with tempfile.TemporaryDirectory() as folder:
        history = Path(folder) / "chat_history.json"
        session = _session(history, retriever, settings)
        turns: list[ChatTurn] = []
        logger.info("[%d] %s · %d messages", scenario["id"], scenario["name"], len(scenario["messages"]))

        for number, message in enumerate(scenario["messages"], start=1):
            turn = await session.ask(message["text"])
            turns.append(turn)
            logger.info(
                "   %2d. %-11s sources=%d citations=%d memory=%s",
                number,
                message.get("expect", ""),
                len(turn.answer.sources),
                len(turn.answer.citations),
                ", ".join(turn.memory.changes) if turn.memory and turn.memory.changed else "-",
            )

        # The restart, before the closing questions: if the memory does not
        # survive being read back from disk, the answers below will say so.
        session = _session(history, retriever, settings).reload()
        restored = len(session.messages)
        closing = []

        for question in FINAL_QUESTIONS:
            turn = await session.ask(question)
            turns.append(turn)
            closing.append(turn)
            logger.info("   closing: %s -> %s", question, turn.answer.answer[:70])

    return _record(scenario, turns, closing, session, restored)


def _session(history: Path, retriever: RAGRetriever, settings) -> ChatSession:
    from app.services.agent_history_storage import AgentHistoryStorage

    return ChatSession(
        storage=AgentHistoryStorage(str(history)),
        retriever=retriever,
        settings=settings,
        enhanced=EnhancedRetriever(retriever=retriever, settings=settings),
    )


def _record(scenario: dict, turns: list[ChatTurn], closing: list[ChatTurn], session, restored: int) -> dict:
    memory = session.task_memory()
    final_text = " ".join(turn.answer.answer for turn in closing)
    document_turns = [turn for turn in turns if turn.retrieval is not None]
    sources = [source for turn in turns for source in turn.answer.sources]
    indexed = {chunk.chunk_id for turn in document_turns for chunk in turn.retrieval.retrieval.chunks}

    checks = {
        "goal_preserved": bool(
            memory.goal and mentions(memory.goal, scenario.get("goal_contains", []))
        ),
        "constraints_preserved": sorted(
            kept(list(memory.constraints), scenario.get("expected_constraints", []))
        ),
        "terms_preserved": sorted(kept(list(memory.confirmed_terms), scenario.get("expected_terms", []))),
        "decisions_preserved": sorted(
            kept(list(memory.decisions), scenario.get("expected_decisions", []))
        ),
        "rag_ran_for_every_question": len(document_turns) == len(turns),
        "every_answer_reported_sources": all(
            isinstance(turn.answer.sources, list) for turn in turns
        ),
        "every_source_is_a_retrieved_chunk": all(source.chunk_id in indexed for source in sources),
        "final_answer_respects_task_memory": sorted(
            mentions(final_text, list(memory.constraints) + list(memory.decisions))
        ),
    }

    return {
        "id": scenario["id"],
        "name": scenario["name"],
        "messages": len(scenario["messages"]),
        "turns": len(turns),
        "history_restored": restored,
        "task_memory": memory.as_dict(),
        "checks": checks,
        "answers_with_citations": sum(1 for turn in turns if turn.answer.citations),
        "answers_without_sources": sum(1 for turn in turns if not turn.answer.sources),
        "sources_total": len(sources),
        "turns_detail": [
            {
                "question": turn.question,
                "answer": turn.answer.answer,
                "rag_status": turn.answer.rag_status,
                "sources": [source.as_dict() for source in turn.answer.sources],
                "citations": [citation.as_dict() for citation in turn.answer.citations],
                "retrieved": len(turn.retrieval.retrieval.chunks) if turn.retrieval else 0,
                "filtered": len(turn.retrieval.filtered.kept) if turn.retrieval else 0,
                "final": len(turn.retrieval.final) if turn.retrieval else 0,
                "best_relevance": round(turn.answer.best_relevance, 4),
                "memory_changes": turn.memory.changes if turn.memory else [],
                "history_length": turn.history_length,
            }
            for turn in turns
        ],
    }


def metrics(records: list[dict], seconds: float) -> dict:
    return {
        "scenarios": len(records),
        "total_turns": sum(record["turns"] for record in records),
        "goal_preserved": sum(1 for record in records if record["checks"]["goal_preserved"]),
        "constraints_preserved": sum(
            1 for record in records if record["checks"]["constraints_preserved"]
        ),
        "terms_preserved": sum(1 for record in records if record["checks"]["terms_preserved"]),
        "decisions_preserved": sum(1 for record in records if record["checks"]["decisions_preserved"]),
        "rag_ran_for_every_question": sum(
            1 for record in records if record["checks"]["rag_ran_for_every_question"]
        ),
        "every_answer_reported_sources": sum(
            1 for record in records if record["checks"]["every_answer_reported_sources"]
        ),
        "every_source_is_a_retrieved_chunk": sum(
            1 for record in records if record["checks"]["every_source_is_a_retrieved_chunk"]
        ),
        "final_answer_respects_task_memory": sum(
            1 for record in records if record["checks"]["final_answer_respects_task_memory"]
        ),
        "answers_with_citations": sum(record["answers_with_citations"] for record in records),
        "sources_total": sum(record["sources_total"] for record in records),
        "evaluation_seconds": round(seconds, 1),
    }


def report(document: dict) -> None:
    for record in document["results"]:
        print("\n" + "=" * 78)
        print(f"Scenario {record['id']}: {record['name']}  ({record['turns']} turns)")
        print(f"\nTask memory after the conversation:")

        for name, value in record["task_memory"].items():
            print(f"  {name:18} {value}")

        print(f"\nHistory restored from disk: {record['history_restored']} messages")
        print("\nChecks:")

        for name, value in record["checks"].items():
            mark = "ok  " if value else "FAIL"
            print(f"  {mark} {name:38} {value if not isinstance(value, bool) else ''}")

        print("\nClosing answers:")

        for turn in record["turns_detail"][-2:]:
            print(f"  Q: {turn['question']}")
            print(f"  A: {_wrapped(turn['answer'])}")

    print("\n" + "=" * 78)
    print("Metrics")

    for name, value in document["metrics"].items():
        print(f"  {name:38} {value}")


def _wrapped(text: str, width: int = 72) -> str:
    import textwrap

    return "\n     ".join(textwrap.wrap(" ".join(text.split()), width=width)[:6])


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluate_chat",
        description="Run the two long scenarios and check that the chat keeps the task.",
    )
    parser.add_argument("--only", type=int, nargs="*", help="run only these scenario ids")
    parser.add_argument("--answer-threshold", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--final-top-k", type=int, default=None)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--results", default=str(RESULTS_FILE))

    return parser.parse_args(argv)


async def evaluate(options: argparse.Namespace) -> dict:
    scenarios = load_scenarios()

    if options.only:
        scenarios = [scenario for scenario in scenarios if scenario["id"] in options.only]

    started = time.perf_counter()
    records = [await run_scenario(scenario, options) for scenario in scenarios]

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "metrics": metrics(records, time.perf_counter() - started),
        "results": records,
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for name in ("faiss", "faiss.loader", "httpx", "app.services.chat_session"):
        logging.getLogger(name).setLevel(logging.WARNING)

    options = parse(argv)
    document = asyncio.run(evaluate(options))
    path = Path(options.results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    report(document)
    print(f"\nWritten: {path}")


if __name__ == "__main__":
    main()
