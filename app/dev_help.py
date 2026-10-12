"""Day 31: the developer assistant from a terminal.

    python -m app.dev_help "/help"
    python -m app.dev_help "/help How does RAG work in this project?"
    python -m app.dev_help "/help Which Git branch is currently active?" --provider ollama
    python -m app.dev_help "/help Explain the project architecture." --json

The same DevAssistant the /dev/chat endpoint uses - nothing here answers
anything itself.
"""

import argparse
import asyncio
import json
import logging

from app.services.dev_assistant import DevAssistant, NotAHelpCommand
from app.services.llm_provider import GEMINI, PROVIDERS, ProviderUnavailable


def retrieval_line(retrieval: dict | None) -> str:
    """One line on how the documentation was searched, or "" if it was not."""
    if not retrieval:
        return ""

    query = f"{retrieval['rewrite_used']} \"{retrieval['search_query'][:70]}\""
    found = (
        f"best similarity {retrieval['best_similarity']:.3f} (filter {retrieval['similarity_threshold']:.2f}) · "
        f"kept {retrieval['kept']}/{retrieval['retrieved']}"
    )

    if retrieval["retried_with_question"]:
        line = (
            f"search={query} kept nothing (best {retrieval['rewrite_best_similarity']:.3f}) "
            f"-> searched again with the question: {found}"
        )
    else:
        line = f"search={query} · {found}"

    if retrieval["rejected_citations"]:
        line += f" · {retrieval['rejected_citations']} citation(s) rejected"

    return line


def show(result: dict) -> None:
    print(f"\n{result['answer']}\n")

    if result["sources"]:
        print("Sources:")
        for source in result["sources"]:
            print(f"  - {source['file']} · {source['section']} · {source['chunk_id']}")

    if result["citations"]:
        print("Citations:")
        for citation in result["citations"]:
            print(f"  - [{citation['chunk_id']}] \"{citation['quote'][:110]}\"")

    if result.get("retrieval"):
        print(f"Retrieval: {retrieval_line(result['retrieval'])}")

    for tool in result["tools_used"]:
        state = json.dumps(tool["result"], ensure_ascii=False) if tool["ok"] else f"failed: {tool['error']}"
        print(f"Tool: {tool['server']}.{tool['name']} -> {state} ({tool['ms']} ms)")

    print(
        f"\nprovider={result['provider']} · model={result['model'] or '-'} · rag_status={result['rag_status']} "
        f"· timing={json.dumps(result['timing'])}"
    )


async def run(options) -> int:
    assistant = DevAssistant(options.provider)

    try:
        result = (await assistant.ask(" ".join(options.message))).as_dict()
    except NotAHelpCommand as error:
        print(f"{error}. Try: /help")
        return 2
    except ProviderUnavailable as error:
        print(f"{error.provider} is unavailable ({error.reason}) - no other model was asked.\n{error.message}")
        return 3

    if options.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        show(result)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.dev_help", description="Ask the developer assistant.")
    parser.add_argument("message", nargs="+", help='the command, e.g. "/help How does RAG work?"')
    parser.add_argument("--provider", default=GEMINI, choices=list(PROVIDERS))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="show the pipeline's log lines")
    options = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if options.verbose else logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return asyncio.run(run(options))


if __name__ == "__main__":
    raise SystemExit(main())
