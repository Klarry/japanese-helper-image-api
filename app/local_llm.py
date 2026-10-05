"""Day 26: ask the local model from a terminal.

    python -m app.local_llm "What is MCP?"
    python -m app.local_llm --health
    python -m app.local_llm --demo

It checks that Ollama is up before sending anything, because "connection
refused" after a thirty-second wait is a worse answer than "it is not
running, here is how to start it". ``--demo`` runs the three prompts of
increasing difficulty the assignment asks for and writes the real answers
to disk - no fixtures, no canned text; whatever the model actually said is
what lands in the file.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.services import ollama_service
from app.services.ollama_service import OllamaUnavailable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "data" / "local_llm" / "day26_results.json"

#: Three questions, deliberately uneven. The first has a one-line answer,
#: the second needs a distinction held in mind, and the third needs several
#: steps ordered - which is where a 4B model either holds together or does
#: not. Asking it the same question three ways would have proved nothing.
DEMO_PROMPTS = (
    ("simple", "What is MCP?"),
    ("medium", "Explain the difference between MCP client and MCP server."),
    (
        "complex",
        "Explain how an AI agent could use multiple MCP servers "
        "to execute a multi-step task.",
    ),
)

RULE = "-" * 70


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.local_llm",
        description="Ask the local model running under Ollama.",
    )
    parser.add_argument("prompt", nargs="?", help="what to ask")
    parser.add_argument("--model", default="", help=f"override OLLAMA_MODEL (default: {ollama_service.MODEL})")
    parser.add_argument("--health", action="store_true", help="only check that Ollama is up")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the three assignment prompts and write the real answers to data/local_llm/",
    )
    parser.add_argument("--thinking", action="store_true", help="also print the model's reasoning")

    return parser.parse_args(argv)


def show_health(health: dict) -> bool:
    """Print the verdict and say whether it is safe to carry on."""
    if not health["available"]:
        print(f"Ollama: unavailable ({ollama_service.BASE_URL})\n")
        print(health["error"])

        return False

    print(f"Ollama: available ({ollama_service.BASE_URL})")
    print(f"Model: {health['model']}")

    if not health.get("model_installed", True):
        installed = ", ".join(health.get("models") or []) or "none"
        print(f"\nThe model '{health['model']}' is not installed. Installed: {installed}")
        print(f"Pull it with:\n\n    ollama pull {health['model']}")

        return False

    return True


def show_answer(prompt: str, answer, thinking: bool) -> None:
    print(f"\nPrompt:\n{prompt}")

    if thinking and answer.thinking:
        print(f"\nThinking:\n{answer.thinking}")

    print(f"\nResponse:\n{answer.response}")
    print(f"\n[{answer.seconds:.2f}s · {answer.prompt_tokens or 0} in · {answer.response_tokens or 0} out]")


async def run_demo(model: str, thinking: bool) -> int:
    """The three prompts, one after another, saved as they come back."""
    records = []

    for difficulty, prompt in DEMO_PROMPTS:
        print(f"\n{RULE}\n{difficulty.upper()}\n{RULE}")

        try:
            answer = await ollama_service.generate(prompt, model=model)
        except OllamaUnavailable as error:
            print(f"\n{error.message}", file=sys.stderr)

            return 1

        show_answer(prompt, answer, thinking)
        records.append(
            {
                "difficulty": difficulty,
                "prompt": prompt,
                "model": answer.model,
                "response": answer.response,
                "thinking": answer.thinking,
                "seconds": answer.seconds,
                "prompt_tokens": answer.prompt_tokens,
                "response_tokens": answer.response_tokens,
            }
        )

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "base_url": ollama_service.BASE_URL,
                "model": model or ollama_service.MODEL,
                "prompts": len(records),
                "results": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{RULE}\nWritten: {RESULTS_FILE}")

    return 0


async def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)

    if not options.health and not options.demo and not options.prompt:
        print('Nothing to ask. Try: python -m app.local_llm "What is MCP?"', file=sys.stderr)

        return 2

    health = await ollama_service.check_ollama_health(options.model)

    if not show_health(health):
        return 1

    if options.health:
        return 0

    if options.demo:
        return await run_demo(options.model, options.thinking)

    try:
        answer = await ollama_service.generate(options.prompt, model=options.model)
    except OllamaUnavailable as error:
        print(f"\n{error.message}", file=sys.stderr)

        return 1

    show_answer(options.prompt, answer, options.thinking)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
