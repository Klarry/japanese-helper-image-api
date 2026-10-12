"""Day 31: the developer assistant's demo requests, run and checked.

    python -m app.evaluate_day31
    python -m app.evaluate_day31 --provider ollama

Six requests - the five from the assignment and one the documentation cannot
answer - through the real assistant: the real index, the real MCP server,
the real model. Each is checked against something independent of the
assistant:

- a documentation answer must come with at least one source, and every
  citation must hold against the chunk it names (the pipeline checks this
  too; it is checked again here);
- a branch answer must contain the branch ``git branch --show-current``
  prints right now, run by this script, not by the assistant;
- the unanswerable question must not be answered with invented details;
- with --provider ollama, Gemini's text-generation functions are replaced
  by traps for the whole run, so any fallback would fail loudly. (Query
  embeddings still use gemini-embedding-001: the index was built with it.)

Results: data/evaluation/day31_results.json. Nothing is estimated.
"""

import argparse
import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path

from app.services import gemini_service
from app.services.dev_assistant import DevAssistant
from app.services.document_loader import PROJECT_ROOT
from app.services.llm_provider import GEMINI, OLLAMA, PROVIDERS, ProviderUnavailable
from app.services.rag_response import ANSWERED, INSUFFICIENT
from app.services.rag_prompt import DONT_KNOW

RESULTS_FILE = PROJECT_ROOT / "data" / "evaluation" / "day31_results.json"
RULE = "=" * 78

REQUESTS = [
    ("overview", "/help"),
    ("docs", "/help Explain the project architecture."),
    ("docs", "/help How does RAG work in this project?"),
    ("docs", "/help How is the Ollama provider integrated?"),
    ("git", "/help Which Git branch is currently active?"),
    ("unsupported", "/help Which Kubernetes namespace and Helm chart is the backend deployed with?"),
]


def live_branch() -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    return completed.stdout.strip()


def trap_gemini_generation() -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Gemini text generation was called in local mode")

    # Every Gemini text call - generate_text, generate_text_with_usage, and
    # the callers that imported them by name - goes through these two module
    # functions, looked up at call time. embed_texts does not.
    for name in ("_generate", "_post_to_gemini"):
        setattr(gemini_service, name, forbidden)


def check(kind: str, result: dict, branch: str) -> dict:
    checks: dict[str, bool] = {}

    if kind == "overview":
        checks["lists_examples"] = "/help" in result["answer"]
        checks["no_model_call"] = result["model"] is None
    elif kind == "docs":
        checks["answered"] = result["rag_status"] == ANSWERED and DONT_KNOW not in result["answer"]
        checks["has_sources"] = bool(result["sources"])
        checks["citations_hold"] = all(
            any(citation["chunk_id"] == source["chunk_id"] for source in result["sources"])
            for citation in result["citations"]
        )
        checks["no_git_tool"] = not result["tools_used"]
    elif kind == "git":
        tool = result["tools_used"][0] if result["tools_used"] else {}
        checks["tool_called"] = tool.get("name") == "get_git_branch" and tool.get("ok") is True
        checks["live_branch_in_answer"] = bool(branch) and f"`{branch}`" in result["answer"]
        checks["no_rag"] = result["rag_status"] == "disabled"
    elif kind == "unsupported":
        refused = result["rag_status"] == INSUFFICIENT or DONT_KNOW in result["answer"] or "don't know" in result["answer"].lower()
        checks["refused"] = refused
        checks["no_invented_names"] = not any(word in result["answer"].lower() for word in ("namespace:", "chart:", "helm install"))

    return checks


async def run(options) -> int:
    if options.provider == OLLAMA:
        trap_gemini_generation()

    branch = live_branch()
    assistant = DevAssistant(options.provider)
    print(f"\n{RULE}\n=== Day 31 · Developer assistant ===\n{RULE}")
    print(f"provider {options.provider} · git branch --show-current: {branch or '(none)'}\n")
    records = []

    for kind, message in REQUESTS:
        started = time.perf_counter()

        try:
            result = (await assistant.ask(message)).as_dict()
            error = ""
        except ProviderUnavailable as failure:
            result, error = None, f"{failure.provider} unavailable: {failure.reason}"

        seconds = round(time.perf_counter() - started, 2)
        checks = check(kind, result, branch) if result else {"ran": False}
        passed = all(checks.values())
        records.append(
            {"kind": kind, "message": message, "seconds": seconds, "passed": passed, "checks": checks, "error": error, "result": result}
        )
        verdict = "PASS" if passed else "FAIL"
        print(f"[{verdict}] {message}")
        print(f"       {seconds}s · " + " · ".join(f"{name}={'yes' if ok else 'NO'}" for name, ok in checks.items()))

        if result:
            first = result["answer"].strip().splitlines()[0] if result["answer"].strip() else ""
            print(f"       answer: {first[:150]}")
            if result["sources"]:
                print("       sources: " + ", ".join(f"{s['file']}#{s['chunk_id']}" for s in result["sources"][:4]))
            for tool in result["tools_used"]:
                print(f"       tool: {tool['name']} -> {json.dumps(tool['result']) if tool['ok'] else tool['error']}")
        else:
            print(f"       {error}")

        print()

    passed = sum(1 for record in records if record["passed"])
    print(f"{RULE}\n{passed}/{len(records)} requests passed their checks\n{RULE}")
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "provider": options.provider,
                "git_branch_from_git": branch,
                "passed": passed,
                "total": len(records),
                "requests": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Written: {RESULTS_FILE}")

    return 0 if passed == len(records) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.evaluate_day31", description="Run and check the /help demo requests.")
    parser.add_argument("--provider", default=GEMINI, choices=list(PROVIDERS))
    options = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    return asyncio.run(run(options))


if __name__ == "__main__":
    raise SystemExit(main())
