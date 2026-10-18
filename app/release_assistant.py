"""Day 35: the AI Release Assistant from a terminal.

    python -m app.release_assistant --base-ref main --provider ollama
    python -m app.release_assistant --repo android --base-ref v1.0 --run-checks android-unit-tests
    python -m app.release_assistant                       # no base: the working tree against HEAD, said so in the report
    python -m app.release_assistant --repo ../japaneseHelper --base-ref HEAD~5 --output-dir output/release-android

Prepares a release *draft*: a changelog, release notes, a review of the changes and a
report, saved in ``output/release/<run_id>/``. It never commits, tags, builds or
publishes, and it does not touch the repository it reads. The same ReleaseAssistant
the ``POST /release/prepare`` endpoint uses.

Exit codes: 0 completed (or nothing to release), 1 partial (something could not be done,
see the errors), 2 the request was refused, 3 the model was unavailable and nothing was reviewed.
The readiness verdict is printed but is not an exit code: it is a judgment for a person.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from app.services.llm_provider import OLLAMA, PROVIDERS
from app.services.release_assistant import COMPLETED, FAILED, NO_CHANGES, PARTIAL, ReleaseAssistant, ReleaseError, ReleaseRequest
from app.services.release_config import load_config

MAX_LISTED = 10


def _lines(title: str, items: list[str], limit: int = MAX_LISTED) -> None:
    if not items:
        return

    print(f"\n{title}:")

    for item in items[:limit]:
        print(f"  - {item}")

    if len(items) > limit:
        print(f"  … {len(items) - limit} more in run.json")


def show(run: dict) -> None:
    repo, scope, readiness = run["repository"], run["scope"], run["readiness"]
    print(f"\nRelease preparation {run['run_id']} - status: {run['status']} · readiness: {readiness['verdict']}")
    print(f"Repository: {repo['name']} · branch {repo['branch'] or 'detached'} @ {repo['head_short']} · {run['request']['provider']}")
    print(f"Scope: {scope['description']}")
    inspection = run["inspection"]
    print(f"\n[1] Inspect: git tools {', '.join(inspection['mcp_tools']['project-git'])}")
    print(f"    file tools {', '.join(inspection['mcp_tools']['project-files'])}")
    indexes = inspection["assistant_project"]["rag_indexes"]
    print(f"    RAG indexes: {', '.join(str(i['strategy']) + ' (' + str(i.get('chunks')) + ' chunks)' for i in indexes) or 'none'}")

    for version in inspection.get("version_files", []):
        facts = version["facts"]

        if facts:
            print(f"    {version['path']}: package {facts.get('namespace', '-')} · versionName {facts.get('version_name', '-')} · versionCode {facts.get('version_code', '-')}")

    start = run["git_status_at_start"]
    print(f"    git status at start: {'clean' if start['clean'] else 'dirty ' + json.dumps(start['counts'])}")
    analyzed = [item for item in run["files"] if item["analysis"] == "analyzed"]
    withheld = [item for item in run["files"] if item["analysis"] == "withheld"]
    print(f"\n[2] Collect: {len(analyzed)} files analyzed, {len(withheld)} withheld, {len(run['other_changes']['untracked'])} untracked and {len(run['other_changes']['uncommitted_not_analyzed'])} uncommitted not analyzed")

    for item in withheld:
        print(f"    withheld {item['path']} ({item['withheld']})")

    review = run["review"]

    if review:
        print(f"\n[3] Review ({review['provider']}, {review.get('model') or '-'}): {review['files_reviewed']} of {len(analyzed)} files reviewed")

        for batch in review["batches"]:
            print(f"    batch {batch['batch']}: {len(batch['files'])} files · {batch['status']} · {batch['findings']} findings · {batch['ms'] / 1000:.1f} s" + (f" · {batch['error']}" if batch["error"] else ""))

    docs = run["documentation"]

    if docs:
        print(f"    documentation: {docs.get('terms', 0)} names searched, {docs.get('total_flagged', 0)} documents mention changed files")

    print("\n[4] Checks:")

    for check in run["checks"] or [{"id": "(none configured)", "status": "-", "exit_code": None, "seconds": 0, "command": "", "reason": ""}]:
        print(f"    {check['id']}: {check['status']}" + (f" (exit {check['exit_code']}, {check['seconds']} s)" if check["exit_code"] is not None else "") + (f" - {check['reason']}" if check["reason"] and check["status"] != "not_run" else ""))

        for line in check.get("errors", [])[:3]:
            print(f"        {line}")

    findings = run["findings"]
    print(f"\nFindings ({len(findings)}):")

    for item in findings[:MAX_LISTED]:
        where = f"{item['file']}:{item['line']}" if item.get("file") and item.get("line") else item.get("file") or "-"
        print(f"  {item['id']} [{item['severity']}/{item['kind']}] {item['title']} ({where}) · {item['source']} · {item['validation_status']}")

    draft = run["draft"]
    print(f"\n[5] Draft: {draft.get('status', '-')} · changelog {len(draft.get('changelog', []))} entries · release notes {len(draft.get('release_notes', []))} · discarded {len(draft.get('rejected', []))}")

    if draft.get("reason"):
        print(f"    {draft['reason']}")

    for entry in draft.get("changelog", [])[:MAX_LISTED]:
        print(f"    {entry['category']}: {entry['text']} ({', '.join(entry['files'])})")

    _lines("Blockers", readiness["blockers"])
    _lines("Warnings", readiness["warnings"])
    _lines("Not verified", readiness["unverified"])
    _lines("Questions for a person", run["unresolved"])
    _lines("Limitations", run["limitations"], 6)

    for error in run["errors"]:
        print(f"\n! {error['stage']}: {error['kind']} - {error['message']}")

    print("\nArtifacts:")

    for artifact in run["artifacts"] or [{"path": "none written", "bytes": 0, "sha256": ""}]:
        print(f"  - {artifact['path']}" + (f" ({artifact['bytes']} bytes, sha256 {artifact['sha256'][:12]})" if artifact.get("sha256") else ""))

    llm = run["llm"]
    print(f"\nProvider {llm.get('provider')} ({'local' if llm.get('local') else 'cloud'}) · models {', '.join(llm.get('models', [])) or '-'} · {run['seconds']} s · {len(run['tool_calls'])} MCP calls")
    print(f"Readiness: {readiness['verdict']} - {readiness['summary']}\n")


def build_request(options) -> ReleaseRequest:
    config = load_config()
    chosen = options.repo
    named = chosen in config.repositories

    return ReleaseRequest(
        base_ref=options.base_ref or "",
        provider=options.provider,
        repository=chosen if named else "custom",
        repository_path="" if named else str(Path(chosen).expanduser()),
        include_uncommitted=options.include_uncommitted,
        run_checks=options.run_checks is not None,
        checks=tuple(options.run_checks or ()),
        output_dir=options.output_dir or "",
        max_review_batches=options.max_review_batches,
    )


async def run(options) -> int:
    try:
        result = await ReleaseAssistant(options.provider).prepare(build_request(options))
    except ReleaseError as error:
        print(f"Refused ({error.kind}): {error.message}")
        return 2

    document = result.as_dict()

    if options.json:
        print(json.dumps(document, ensure_ascii=False, indent=2))
    else:
        show(document)

    if result.status == FAILED:
        return 3 if any(item["kind"] == "provider_unavailable" for item in result.errors) else 1

    return 0 if result.status in (COMPLETED, NO_CHANGES) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.release_assistant", description="Prepare a release draft: review, changelog, release notes, report.")
    parser.add_argument("--base-ref", metavar="REF", help="the start of the release range: a branch, tag or commit. Without it the working tree is analyzed against HEAD and no range is guessed")
    parser.add_argument("--provider", default=OLLAMA, choices=list(PROVIDERS))
    parser.add_argument("--repo", default="backend", help="a repository named in data/release/release_config.json (backend, android) or a folder")
    parser.add_argument("--include-uncommitted", action="store_true", help="with --base-ref: also analyze uncommitted changes of the working tree")
    parser.add_argument("--run-checks", nargs="*", metavar="ID", help="run the configured checks (all of them, or the ids given); nothing runs without this flag")
    parser.add_argument("--output-dir", metavar="DIR", help="where the run folders go; must be inside output/ (default: output/release)")
    parser.add_argument("--max-review-batches", type=int, metavar="N", help="at most N model calls for the review (default 3)")
    parser.add_argument("--json", action="store_true", help="print the whole run record as JSON")
    parser.add_argument("--verbose", action="store_true", help="show the log lines")
    options = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if options.verbose else logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return asyncio.run(run(options))


if __name__ == "__main__":
    raise SystemExit(main())
