"""Day 34: the project file assistant from a terminal.

    python -m app.project_task "Find all usages of SupportAssistant and explain the request flow"
    python -m app.project_task "Compare the support assistant docs with the code and prepare a patch" --provider gemini
    python -m app.project_task --rerun output/runs/<run_id>.json     # the same task and provider again, compared
    python -m app.project_task --apply README.md --proposal output/proposals/README.md --expected-sha256 <sha>
    python -m app.project_task --apply README.md --proposal ... --expected-sha256 <sha> --confirm

The same FileAssistant the /project-files endpoints use. Without --confirm,
--apply only shows the diff.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from app.services.file_assistant import COMPLETED, PARTIAL, ApplyRefused, FileAssistant, apply_proposal
from app.services.llm_provider import OLLAMA, PROVIDERS, ProviderUnavailable
from app.services.project_files import ProjectFiles

MAX_DIFF_LINES = 80


def _diff(text: str) -> None:
    lines = text.splitlines()

    for line in lines[:MAX_DIFF_LINES]:
        print(f"    {line}")

    if len(lines) > MAX_DIFF_LINES:
        print(f"    … {len(lines) - MAX_DIFF_LINES} more lines in the patch file")


def show(result: dict) -> None:
    plan = result.get("plan") or {}
    print(f"\nTask: {result['task']}")
    print(f"Workflow: {result['workflow'] or '-'} · subject: {result['subject'] or '-'} · plan: {plan.get('source', '-')}"
          f" · search terms: {', '.join(plan.get('search_terms') or []) or '-'}")
    print(f"\nTool calls ({len(result['tool_calls'])}):")

    for number, call in enumerate(result["tool_calls"], start=1):
        outcome = call["summary"] if call["ok"] else f"refused: {call['error']}"
        print(f"  {number:2d}. [{call['planned_by']}] {call['tool']} {json.dumps(call['arguments'], ensure_ascii=False)[:90]}"
              f" -> {outcome} ({call['ms']} ms)")

    stages = [f"{call['stage']} {call['ms'] / 1000:.1f}s" + ("" if call["ok"] else f" ({call['error']})")
              for call in result["model_calls"]]
    print(f"\nModel calls: {', '.join(stages) or '-'}")
    print(f"\nFiles inspected ({len(result['files_inspected'])}):")

    for read in result["files_inspected"]:
        print(f"  - {read['path']} ({read['lines']} lines, read by {read['read_by']}, sha256 {read['sha256'][:12]})")

    print("\nArtifacts:")

    for artifact in result["artifacts"] or [{"path": "none", "kind": "", "bytes": 0, "created": False}]:
        print(f"  - {artifact['path']}" + (f" ({artifact['kind']}, {artifact['bytes']} bytes, "
                                          f"{'created' if artifact['created'] else 'updated'})" if artifact["kind"] else ""))

    for diff in result["diffs"]:
        print(f"\nDiff {diff['path']} (+{diff['added']} -{diff['removed']}), proposal {diff['proposal']}:")
        _diff(diff["diff"])
        print(f"  Apply after review: {diff['apply']}")

    for error in result["errors"]:
        print(f"\n! {error['stage']}: {error['kind']} - {error['message']}")

    for note in result["notes"]:
        print(f"  note: {note}")

    used = result["limits"].get("used", {})
    print(f"\nStatus: {result['status']} · provider {result['provider']} · model {result['model'] or '-'} · "
          f"{used.get('tool_calls', 0)} tool calls · {used.get('files', 0)} files · {used.get('model_calls', 0)} model calls · "
          f"{used.get('seconds', 0)} s")
    print(f"Run record: {result['run_record'] or '-'}")
    print(f"\n{result['summary']}\n")


def compare(previous: dict, result: dict) -> None:
    """What a rerun kept and what it did not - files, saved results, status."""
    def same(key, pick):
        return "same" if pick(previous.get(key) or []) == pick(result.get(key) or []) else "different"

    print("Compared with the earlier run:")
    print(f"  status: {previous.get('status')} -> {result['status']}")
    print(f"  files inspected: {same('files_inspected', lambda items: sorted(item['path'] for item in items))}")
    print(f"  artifacts (path and sha256): {same('artifacts', lambda items: sorted((item['path'], item['sha256']) for item in items))}")
    print(f"  diffs: {same('diffs', lambda items: sorted((item['path'], item['diff']) for item in items))}\n")


async def run(options) -> int:
    if options.apply:
        try:
            result = await apply_proposal(options.apply, options.proposal, options.expected_sha256, options.confirm)
        except ApplyRefused as error:
            print(f"Refused ({error.kind}): {error.message}")
            return 2

        print(f"\n{result['path']} <- {result['proposal']} (+{result['added']} -{result['removed']})")
        _diff(result["diff"])
        print(f"\n{result['message']}" + (f" sha256 {result['sha256']}" if result["applied"] else ""))
        return 0

    previous = None
    task, provider = " ".join(options.task), options.provider

    if options.rerun:
        files = ProjectFiles()

        if not files.in_output(options.rerun):
            print(f"{options.rerun} is not a run record in {files.output_rel}/")
            return 2

        previous = json.loads((files.root / options.rerun).read_text(encoding="utf-8"))
        task, provider = previous["task"], previous["provider"]

    if not task.strip():
        print("Give a task, or --rerun a run record, or --apply a proposal.")
        return 2

    try:
        result = (await FileAssistant(provider).run(task)).as_dict()
    except ProviderUnavailable as error:
        print(f"{error.provider} is unavailable ({error.reason}) - no other model was asked.\n{error.message}")
        return 3

    if options.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        show(result)

    if previous:
        compare(previous, result)

    return 0 if result["status"] == COMPLETED else 1 if result["status"] == PARTIAL else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.project_task", description="Give the project file assistant a task.")
    parser.add_argument("task", nargs="*", help="the goal, in words; no file names needed")
    parser.add_argument("--provider", default=OLLAMA, choices=list(PROVIDERS))
    parser.add_argument("--rerun", metavar="RECORD", help="run the task of an earlier run record again and compare")
    parser.add_argument("--apply", metavar="PATH", help="apply a saved proposal to this project file")
    parser.add_argument("--proposal", help="the proposal file in the output folder (with --apply)")
    parser.add_argument("--expected-sha256", default="", help="the sha256 the proposal was made against (with --apply)")
    parser.add_argument("--confirm", action="store_true", help="with --apply: write the file (otherwise only the diff is shown)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="show the log lines")
    options = parser.parse_args(argv)

    if options.apply and not (options.proposal and options.expected_sha256):
        parser.error("--apply needs --proposal and --expected-sha256")

    logging.basicConfig(level=logging.INFO if options.verbose else logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return asyncio.run(run(options))


if __name__ == "__main__":
    raise SystemExit(main())
