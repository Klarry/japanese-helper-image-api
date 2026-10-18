"""Day 33: the support assistant on six scenarios, checked.

    python -m app.evaluate_day33                    # provider ollama
    python -m app.evaluate_day33 --provider gemini

The scenarios are in data/support/evaluation.json: a sign-in problem after a
password reset, a failing image generation, a "server unavailable" question,
a ticket that does not exist, a question the help centre does not cover, and
a ticket whose user is missing from the CRM. Each runs through the real
assistant: the real support-crm MCP server, the real help-centre index, the
real model.

For each one the results record the question, the ticket, the MCP tools
called, the help-centre chunks shown to the model and the ones it cited, the
answer, the expected behaviour, which mechanical checks held, latency and
errors. The checks are deliberately simple (status, tools, citations, a word
the answer should contain); the answers are written out in full for a person
to judge.

With --provider ollama every Gemini function - text and embeddings - is
replaced by a trap for the whole run: the help-centre index is embedded
locally, so a support answer in local mode should touch Google not at all,
and a trap that fires makes the run fail loudly.

Results: data/evaluation/day33_results_<provider>.json
"""

import argparse
import asyncio
import json
import logging
import time

from app.services import gemini_service
from app.services.document_loader import PROJECT_ROOT
from app.services.llm_provider import OLLAMA, PROVIDERS, ProviderUnavailable
from app.services.support_assistant import SupportAssistant, SupportError

SCENARIOS = PROJECT_ROOT / "data" / "support" / "evaluation.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation"
RULE = "=" * 78


def trap_all_gemini() -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Gemini was called during a local support run")

    for name in ("_generate", "_post_to_gemini", "embed_texts"):
        setattr(gemini_service, name, forbidden)


def check(expect: dict, record: dict) -> dict:
    checks: dict[str, bool] = {}
    result = record.get("result") or {}
    checks["http"] = record["http"] == expect["http"]

    if "error" in expect:
        checks["error"] = record.get("error_kind") == expect["error"]

    if "status" in expect:
        checks["status"] = result.get("status") == expect["status"]

    checks["tools"] = record["tools"] == expect["tools"]

    if "min_citations" in expect:
        checks["citations"] = len(result.get("citations") or []) >= expect["min_citations"]

    if "max_citations" in expect:
        checks["no_citations"] = len(result.get("citations") or []) <= expect["max_citations"]

    if "answer_mentions_any" in expect:
        answer = (result.get("answer") or "").lower()
        checks["mentions"] = any(word.lower() in answer for word in expect["answer_mentions_any"])

    return checks


async def run(options) -> int:
    if options.provider == OLLAMA:
        trap_all_gemini()

    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"]
    assistant = SupportAssistant(options.provider)
    print(f"\n{RULE}\n=== Day 33 · Support assistant · {len(scenarios)} scenarios ===\n{RULE}\nprovider {options.provider}\n")
    records = []

    for number, scenario in enumerate(scenarios, start=1):
        started = time.perf_counter()
        record = {key: scenario[key] for key in ("id", "question", "ticket_id", "expected_behavior")}
        record.update(http=200, error_kind="", error="", result=None, tools=[])

        try:
            answer = await assistant.ask(scenario["question"], scenario["ticket_id"])
            record["result"] = answer.as_dict()
            record["tools"] = [tool.name for tool in answer.tools_used]
        except SupportError as error:
            record.update(http=error.status or 502, error_kind=error.kind, error=error.message)
            record["tools"] = ["get_ticket"] + (["get_user"] if error.kind.startswith("ticket_user") else [])
        except ProviderUnavailable as error:
            # The CRM is asked before the model, so both calls had run.
            record.update(http=503, error_kind="provider_unavailable", error=f"{error.provider}: {error.reason}",
                          tools=["get_ticket", "get_user"])

        record["latency_ms"] = int((time.perf_counter() - started) * 1000)
        record["checks"] = check(scenario["expect"], record)
        record["passed"] = all(record["checks"].values())
        records.append(record)

        verdict = "PASS" if record["passed"] else "FAIL"
        print(f"[{verdict}] {number}. {scenario['id']} · {scenario['ticket_id']} · «{scenario['question']}»")
        print(f"       {record['latency_ms'] / 1000:.1f}s · http {record['http']} · tools {', '.join(record['tools']) or '-'} · "
              + " · ".join(f"{name}={'yes' if ok else 'NO'}" for name, ok in record["checks"].items()))

        result = record["result"]

        if result:
            shown = ", ".join(chunk["chunk_id"].replace("docs-support-", "") for chunk in result["retrieval"]["chunks"]) or "none"
            cited = ", ".join(source["chunk_id"].replace("docs-support-", "") for source in result["sources"]) or "none"
            print(f"       status {result['status']} · docs shown: {shown} · cited: {cited} · facts used: {', '.join(result['ticket_facts_used']) or '-'}")
            print(f"       answer: {' '.join(result['answer'].split())[:220]}")
        else:
            print(f"       {record['error_kind']}: {record['error']}")

        print()

        if record["error_kind"] == "provider_unavailable":
            break

    passed = sum(1 for record in records if record["passed"])
    answered = [record["latency_ms"] for record in records if record["result"]]
    print(f"{RULE}\n{passed}/{len(records)} scenarios passed their checks · median answer "
          f"{sorted(answered)[len(answered) // 2] / 1000 if answered else 0:.1f}s\n{RULE}")
    results = RESULTS_DIR / f"day33_results_{options.provider}.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(
        json.dumps({"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "provider": options.provider,
                    "passed": passed, "total": len(records), "scenarios": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Written: {results}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.evaluate_day33", description="Run the Day 33 support scenarios.")
    parser.add_argument("--provider", default=OLLAMA, choices=list(PROVIDERS))
    options = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    return asyncio.run(run(options))


if __name__ == "__main__":
    raise SystemExit(main())
