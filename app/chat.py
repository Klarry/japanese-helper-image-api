"""Day 25: the mini chat.

    python -m app.chat
    python -m app.chat --debug
    python -m app.chat --new            # start a fresh conversation
    python -m app.chat --answer-threshold 0.60

Every question goes through retrieval, every answer comes back with its
sources, and the conversation - with its task memory - is on disk after each
turn, so closing the terminal is not the same as forgetting.

Two modes, and the plain one is the point. A chat that prints its own
internals at every turn is a demo, not a chat; so the default shows the
answer and its sources and nothing else, and ``/debug`` turns the rest on
when someone wants to see the machinery.

Commands, all of them one word: /debug, /memory, /history, /sources, /new,
/help, /exit.
"""

import argparse
import asyncio
import logging

from app.services.chat_memory import TaskMemory
from app.services.chat_session import ChatSession, ChatTurn
from app.services.rag_enhanced import EnhancedRetriever
from app.services.rag_retriever import DEFAULT_STRATEGY, RAGRetriever
from app.services.rag_settings import DEFAULT_SETTINGS

logger = logging.getLogger("chat")

BANNER = (
    "Mini chat with RAG and task memory. Type a question, or /help for commands.\n"
    "Every answer carries its sources; the conversation is saved after each turn."
)
HELP = """Commands:
  /debug      show task memory, the retrieval funnel and the history size with each answer
  /memory     print the task memory as it stands
  /history    print how much is stored, and the last few turns
  /sources    print the sources of the last answer in full
  /new        end this conversation and its task memory, and start over
  /exit       leave (the conversation stays on disk)"""


def show_answer(turn: ChatTurn) -> None:
    """The plain view: the answer, and where it came from."""
    print(f"\nAssistant:\n{turn.answer.answer}\n")
    print("Sources:")

    if turn.answer.sources:
        for number, source in enumerate(turn.answer.sources, start=1):
            print(f"  {number}. {source.file} / {source.section or '-'} / {source.chunk_id}")
    else:
        print("  None")

    if turn.answer.citations:
        print("\nCitations:")

        for citation in turn.answer.citations:
            print(f'  > "{citation.quote}"')

    print()


def show_debug(turn: ChatTurn) -> None:
    """The machinery, for anyone who wants to see it."""
    found = turn.retrieval
    memory = turn.memory

    print("\n--- debug " + "-" * 56)

    if memory is not None:
        print("Task Memory:")
        print(_indent(_memory_lines(memory.after)))
        print(f"  changed: {', '.join(memory.changes) if memory.changes else 'no'}")

    if found is not None:
        print(
            f"RAG:\n  retrieved: {len(found.retrieval.chunks)}"
            f"\n  filtered: {len(found.filtered.kept)}"
            f"\n  final: {len(found.final)}"
            f"\n  best relevance: {found.best_relevance:.3f}"
            f" (threshold {turn.answer.answer_threshold})"
            f"\n  rewritten query: {found.query.query}"
        )

    print(f"Status: {turn.answer.rag_status} · confidence: {turn.answer.confidence}")
    print(f"History: {turn.history_length} messages ({len(turn.sent_messages)} sent to the model)")
    print(f"[{turn.seconds:.2f}s · {turn.tokens_used} tokens]")
    print("-" * 66 + "\n")


def _memory_lines(memory: TaskMemory) -> str:
    if memory.is_empty:
        return "(nothing settled yet)"

    lines = [f"goal: {memory.goal or '-'}", f"current_state: {memory.current_state}"]

    for caption, entries in (
        ("confirmed_terms", memory.confirmed_terms),
        ("constraints", memory.constraints),
        ("decisions", memory.decisions),
        ("requirements", memory.requirements),
    ):
        if entries:
            lines.append(f"{caption}:")
            lines += [f"  - {entry}" for entry in entries]

    return "\n".join(lines)


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def show_history(session: ChatSession, last: int = 6) -> None:
    messages = session.messages
    print(f"\nHistory: {len(messages)} messages")

    for message in messages[-last:]:
        stamp = message.get("timestamp", "")
        sources = message.get("sources") or []
        tail = f"  [{len(sources)} source(s)]" if sources else ""
        print(f"  {stamp} {message['role']}: {message['content'][:88]}{tail}")

    print()


def show_sources(turn: ChatTurn | None) -> None:
    if turn is None:
        print("\nNothing has been asked yet.\n")
        return

    print()

    for number, source in enumerate(turn.answer.sources, start=1):
        print(f"{number}. file: {source.file}")
        print(f"   section: {source.section or '-'}")
        print(f"   chunk: {source.chunk_id}")

    if not turn.answer.sources:
        print("None")

    print()


async def run(options: argparse.Namespace) -> None:
    settings = DEFAULT_SETTINGS.with_overrides(
        answer_threshold=options.answer_threshold,
        similarity_threshold=options.threshold,
        final_top_k=options.final_top_k,
    )
    retriever = RAGRetriever(strategy=options.strategy)
    session = ChatSession(
        retriever=retriever,
        settings=settings,
        enhanced=EnhancedRetriever(retriever=retriever, settings=settings),
    )

    if options.new:
        session.clear()

    debug = options.debug
    print(BANNER)

    if session.messages:
        print(f"Restored: {len(session.messages)} messages, task memory loaded.")

    last: ChatTurn | None = None

    while True:
        try:
            line = input("\nUser:\n").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not line:
            continue

        if line in ("/exit", "/quit"):
            return

        if line == "/help":
            print(HELP)
            continue

        if line == "/debug":
            debug = not debug
            print(f"debug {'on' if debug else 'off'}")
            continue

        if line == "/memory":
            print("\n" + _memory_lines(session.task_memory()) + "\n")
            continue

        if line == "/history":
            show_history(session)
            continue

        if line == "/sources":
            show_sources(last)
            continue

        if line == "/new":
            session.clear()
            last = None
            print("Conversation and task memory cleared.")
            continue

        try:
            last = await session.ask(line)
        except Exception as error:  # noqa: BLE001 - one bad turn must not end the chat
            print(f"\n[the turn failed: {error}]\n")
            continue

        show_answer(last)

        if debug:
            show_debug(last)


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.chat",
        description="A chat that answers from the project's own documents and remembers the task.",
    )
    parser.add_argument("--debug", action="store_true", help="show the machinery with every answer")
    parser.add_argument("--new", action="store_true", help="start a fresh conversation")
    parser.add_argument("--answer-threshold", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="similarity filter")
    parser.add_argument("--final-top-k", type=int, default=None)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    for name in ("faiss", "faiss.loader", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)

    asyncio.run(run(parse(argv)))


if __name__ == "__main__":
    main()
