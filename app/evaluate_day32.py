"""Day 32: the code review on five synthetic pull requests, scored.

    python -m app.evaluate_day32 --provider ollama
    python -m app.evaluate_day32 --provider ollama --case bug_slot_leak

The diffs are in tests/fixtures/day32/: changes built against the real files
of this project and never applied to it - an intentional bug, a broken
convention, a correct change, untested new logic, and a change whose
architecture the documentation describes. Each runs through the real
reviewer: the real index and filter, the real model.

What is recorded for each case: the diff, the context that was retrieved
(and whether it came from the files the rubric names), every finding, the
rubric, which expectations were met and which were missed, the unexpected
findings and which of them count as false positives, what validation
rejected, latency, provider and model, and parse errors.

The rubric is mechanical and stated in cases.json: same file, an allowed
category, one keyword. It can be fooled both ways, so the findings are
written out in full for a person to read - this script does not claim the
reviewer is right, only what it said.

With --provider ollama, Gemini text generation is replaced by a trap for the
whole run, so a fallback would fail loudly. (The context search still
embeds its short queries with gemini-embedding-001 - the index's model.)

Results: data/evaluation/day32_results_<provider>.json
"""

import argparse
import asyncio
import json
import logging
import time

from app.evaluate_day31 import trap_gemini_generation
from app.services.code_review import CodeReviewer, InvalidReviewOutput, PullRequest
from app.services.document_loader import PROJECT_ROOT
from app.services.llm_provider import OLLAMA, PROVIDERS, ProviderUnavailable

FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "day32"
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation"
RULE = "=" * 78
FALSE_POSITIVE_SEVERITIES = ("critical", "high", "medium")


def load_cases() -> list[dict]:
    return json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))["cases"]


def matches(finding: dict, expectation: dict) -> bool:
    text = " ".join(str(finding.get(key) or "") for key in ("title", "description", "recommendation")).lower()

    return (
        finding.get("file") == expectation["file"]
        and finding.get("category") in expectation["categories"]
        and any(keyword.lower() in text for keyword in expectation["keywords"])
    )


def score(case: dict, findings: list[dict]) -> dict:
    """Which expectations were met, which were missed, and what else was said."""
    used: set[int] = set()
    met, missed = [], []

    for expectation in case["expected"]:
        hit = next((index for index, finding in enumerate(findings) if index not in used and matches(finding, expectation)), None)

        if hit is None:
            missed.append(expectation["label"])
        else:
            used.add(hit)
            met.append({"expectation": expectation["label"], "finding": hit})

    unexpected = [index for index in range(len(findings)) if index not in used]
    false_positives = [index for index in unexpected if findings[index]["severity"] in FALSE_POSITIVE_SEVERITIES]

    return {"met": met, "missed": missed, "unexpected": unexpected, "false_positives": false_positives}


def show_case(number: int, total: int, case: dict, record: dict) -> None:
    print(f"[{number}/{total}] {case['id']} — {case['goal']}")

    if record["error"]:
        print(f"       ERROR {record['error']}\n")
        return

    context = record["retrieved_context"]
    in_prompt = [chunk for chunk in context["chunks"] if chunk["in_prompt"]]
    origins = sorted({chunk["file"] for chunk in in_prompt})
    print(
        f"       context: {len(in_prompt)} chunk(s) · best similarity {context['best_similarity']:.3f} "
        f"(filter {context['similarity_threshold']:.2f}) · hit expected files: {'yes' if record['context_hit'] else 'no'}"
    )

    if origins:
        print(f"       from: {', '.join(origins)}")

    validation = record["validation"]
    print(
        f"       findings: {len(record['findings'])} · rejected {len(validation['rejected'])} · "
        f"parse {validation['parse']} · {record['latency']['total_ms'] / 1000:.1f}s"
    )

    met = {item["finding"] for item in record["score"]["met"]}
    false = set(record["score"]["false_positives"])

    for index, finding in enumerate(record["findings"]):
        mark = "+" if index in met else ("!" if index in false else "·")
        where = finding["file"] + (f":{finding['line']}" if finding["line"] else "")
        print(f"       {mark} {finding['severity']:<8} {finding['category']:<15} {where} — {finding['title'][:70]}")

    for item in validation["rejected"]:
        print(f"       x rejected: {item['reason']} — {item['title'][:60]}")

    result = record["score"]
    print(
        f"       expected found {len(result['met'])}/{len(case['expected'])} · missed {len(result['missed'])} · "
        f"unexpected {len(result['unexpected'])} · false positives {len(result['false_positives'])}\n"
    )


async def run(options) -> int:
    if options.provider == OLLAMA:
        trap_gemini_generation()

    cases = [case for case in load_cases() if not options.case or case["id"] in options.case]
    reviewer = CodeReviewer(options.provider)
    print(f"\n{RULE}\n=== Day 32 · AI code review · {len(cases)} synthetic pull requests ===\n{RULE}")
    print(f"provider {options.provider}\n")
    records = []

    for number, case in enumerate(cases, start=1):
        diff = (FIXTURES / case["diff"]).read_text(encoding="utf-8")
        pr = PullRequest(
            repository="local/japanese-helper-image-api",
            number=number,
            base_branch="main",
            head_branch=case["head_branch"],
            title=case["title"],
            diff=diff,
        )
        record = {"id": case["id"], "goal": case["goal"], "title": case["title"], "diff_file": case["diff"], "diff": diff,
                  "expected": case["expected"], "provider": options.provider, "error": ""}
        started = time.perf_counter()

        try:
            result = (await reviewer.review(pr)).as_dict()
        except InvalidReviewOutput as error:
            record.update(error=f"invalid_model_output: {error}", model=error.model, parse="failed",
                          latency={"total_ms": int((time.perf_counter() - started) * 1000)})
        except ProviderUnavailable as error:
            record.update(error=f"{error.provider} unavailable: {error.reason}",
                          latency={"total_ms": int((time.perf_counter() - started) * 1000)})
        else:
            chunks = result["context"]["chunks"]
            record.update(
                model=result["model"],
                summary=result["summary"],
                retrieved_context=result["context"],
                context_hit=any(chunk["in_prompt"] and chunk["file"] in case["context_files"] for chunk in chunks),
                findings=result["findings"],
                architecture_notes=result["architecture_notes"],
                recommendations=result["recommendations"],
                limitations=result["limitations"],
                validation=result["validation"],
                parse=result["validation"]["parse"],
                score=score(case, result["findings"]),
                files=result["files"],
                generation=result["generation"],
                latency=result["timing"],
            )

        records.append(record)
        show_case(number, len(cases), case, record)

        if record["error"].startswith(f"{options.provider} unavailable"):
            break

    scored = [record for record in records if not record["error"]]
    expected = sum(len(record["expected"]) for record in scored)
    found = sum(len(record["score"]["met"]) for record in scored)
    false_positives = sum(len(record["score"]["false_positives"]) for record in scored)
    parse_errors = sum(1 for record in records if record.get("parse") == "failed")
    repaired = sum(1 for record in records if record.get("parse") == "repaired")
    latencies = [record["latency"]["total_ms"] for record in scored]
    summary = {
        "cases": len(records),
        "reviewed": len(scored),
        "expected_issues": expected,
        "expected_found": found,
        "false_positives": false_positives,
        "parse_errors": parse_errors,
        "parse_repaired": repaired,
        "median_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else None,
    }
    print(RULE)
    print(
        f"expected issues found {found}/{expected} · false positives {false_positives} · "
        f"parse errors {parse_errors} (repaired {repaired}) · median {summary['median_latency_ms']} ms"
    )
    print(RULE)
    results = RESULTS_DIR / f"day32_results_{options.provider}.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(
        json.dumps({"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "provider": options.provider,
                    "summary": summary, "cases": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Written: {results}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.evaluate_day32", description="Run the code review on the Day 32 cases.")
    parser.add_argument("--provider", default=OLLAMA, choices=list(PROVIDERS))
    parser.add_argument("--case", action="append", help="run only this case id (repeatable)")
    options = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    return asyncio.run(run(options))


if __name__ == "__main__":
    raise SystemExit(main())
