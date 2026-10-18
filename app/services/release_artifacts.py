"""Day 35: the four release artifacts and the run record, as text.

Pure functions from a run (the dict ``Run.as_dict()`` makes) to strings; the
assistant writes them through the project-files MCP server. Nothing here reads
a file, runs Git or asks a model, so what an artifact says is exactly what the
run observed:

    release_report.md      base ref, branch, Git status, changed files, verified changes, review
                           findings, validation checks and their actual results, blockers and open
                           questions, overall readiness
    CHANGELOG.md           Added / Changed / Fixed / Removed / Security - entries with evidence only
    release_notes.md       short, user-facing; verified user-visible changes only
    review_findings.json   findings in the Day 32 code-review shape, plus id / source / validation_status
    run.json               the whole run record (written last, by the assistant)

Every artifact is labelled a draft. Nothing in them is a promise: where there is
nothing verified to say, they say so.
"""

import json

from app.services.release_safety import REASONS

REPORT = "release_report.md"
CHANGELOG = "CHANGELOG.md"
NOTES = "release_notes.md"
FINDINGS = "review_findings.json"
RECORD = "run.json"
NAMES = (REPORT, CHANGELOG, NOTES, FINDINGS)
CATEGORY_ORDER = ("Added", "Changed", "Fixed", "Removed", "Security")


def _cell(value) -> str:
    return str(value if value not in (None, "") else "-").replace("|", "\\|").replace("\n", " ")


def _code(value) -> str:
    return "`" + str(value).replace("`", "'").replace("\n", " ") + "`"


def _list(items, empty: str = "None.") -> str:
    return "\n".join(f"- {item}" for item in items) if items else empty


def _label(doc: dict) -> str:
    scope = doc["scope"]

    if scope.get("base_ref"):
        return f"{scope['base_ref']}...{scope['head_short']}"

    return f"working tree against HEAD {scope['head_short']}"


def _upstream(repo: dict) -> str:
    upstream = repo.get("upstream")

    if not upstream:
        return "no upstream"

    return f"{upstream['name']}, ahead {upstream['ahead']}, behind {upstream['behind']}"


def _draft_note(doc: dict) -> str:
    draft = doc["draft"]

    if draft.get("status") == "failed":
        return f"The model could not produce a draft: {draft.get('reason', 'unknown reason')}."

    if draft.get("status") == "skipped":
        return f"No draft was generated: {draft.get('reason', '')}."

    if doc["status"] == "no_changes":
        return "There were no changes to describe."

    if draft.get("rejected") and not draft.get("changelog") and not draft.get("release_notes"):
        return "The model's entries were all discarded because the change set did not prove them (see run.json)."

    return ""


# --- the report ------------------------------------------------------------------------------------


def render_report(doc: dict) -> str:
    repo, scope, readiness = doc["repository"], doc["scope"], doc["readiness"]
    start = doc["git_status_at_start"]
    lines = [
        f"# Release report (draft) - {repo['title']}",
        "",
        f"> Prepared by the AI Release Assistant · run {_code(doc['run_id'])} · {doc['started_at']} · status **{doc['status']}**.",
        "> A draft for a person to read. This run committed, tagged, built and published nothing, and changed no file of the repository.",
        "",
        f"## Readiness: {_code(readiness['verdict'])}",
        "",
        readiness["summary"],
        "",
        "**Blockers**",
        "",
        _list(readiness["blockers"]),
        "",
        "**Warnings**",
        "",
        _list(readiness["warnings"]),
        "",
        "**Not verified**",
        "",
        _list(readiness["unverified"]),
        "",
        "## Scope",
        "",
        f"- Repository: {_code(repo['name'])} ({repo['title']})",
        f"- Branch: {_code(repo['branch'] or 'detached HEAD')} · HEAD {_code(repo['head_short'])} · upstream: {_upstream(repo)}",
    ]

    if scope.get("base"):
        base = scope["base"]
        lines.append(f"- Base ref: {_code(scope['base_ref'])} → {_code(base['short'])}; merge base {_code(base['merge_base_short'])}")
    else:
        lines.append("- Base ref: **not supplied** - the release range was not guessed")

    lines += [f"- Analyzed: {scope['description']}"]
    counts = start.get("counts", {})
    lines.append(
        "- Git status at start: "
        + ("clean" if start.get("clean") else f"**dirty** - {counts.get('staged', 0)} staged, {counts.get('unstaged', 0)} unstaged, {counts.get('untracked', 0)} untracked")
        + " (nothing was reset, cleaned, stashed or committed)"
    )

    other = doc["other_changes"]

    if other.get("uncommitted_not_analyzed"):
        lines.append(f"- Uncommitted, not analyzed: {', '.join(_code(path) for path in other['uncommitted_not_analyzed'][:8])}")

    if other.get("untracked"):
        lines.append(f"- Untracked, not analyzed: {', '.join(_code(item['path']) for item in other['untracked'][:8])}")

    lines += ["", "## Changed files", ""]

    if doc["files"]:
        lines += ["| File | Status | + / - | Origin | Analysis | Review |", "| --- | --- | --- | --- | --- | --- |"]

        for item in doc["files"]:
            lines.append(
                f"| {_code(item['path'])} | {item['status']} | +{item['additions']} / -{item['deletions']} | {item['origin']}{' (also edited, uncommitted; not analyzed)' if item.get('also_edited_uncommitted') else ''} | "
                f"{item['analysis']}{' (' + item['withheld'] + ')' if item['withheld'] else ''} | {item['review']} |"
            )
    else:
        lines.append("No changed files in the selected scope.")

    withheld = [item for item in doc["files"] if item["analysis"] == "withheld"]

    if withheld:
        lines += ["", "**Withheld from the model and from this report** (listed by path only):", ""]
        lines += [f"- {_code(item['path'])}: {REASONS.get(item['withheld'], item['withheld'])}" for item in withheld]

    commits = doc["commits"]
    lines += ["", f"## Commits ({commits.get('range', '-')})", ""]

    if commits.get("items"):
        note = "These commits are the release range." if commits.get("used_as_evidence") else "No base reference: shown for information only, not analyzed."
        lines += [note, ""]
        lines += [f"- {_code(item['short'])} {item['subject']}" for item in commits["items"][:15]]

        if commits.get("total", 0) > min(15, len(commits["items"])):
            lines.append(f"- … {commits['total']} commits in total")
    else:
        lines.append("No commits in the range.")

    lines += ["", "## Verified changes", ""]
    draft = doc["draft"]
    entries = draft.get("changelog", [])

    if entries:
        lines += ["Each entry below is kept only because its evidence is a changed line of the named file or a commit subject of the range.", ""]
        lines += ["| Category | Entry | Files | Evidence |", "| --- | --- | --- | --- |"]

        for item in entries:
            lines.append(f"| {item['category']} | {_cell(item['text'])} | {_cell(', '.join(item['files']))} | {_cell(item['evidence_kind'].replace('_', ' ') + ': ' + item['evidence'])} |")
    else:
        lines.append(_draft_note(doc) or "No entry could be verified.")

    if draft.get("rejected"):
        proposed = sum((draft.get("proposed") or {}).values())
        lines += ["", f"The model proposed {proposed} entries; {len(draft['rejected'])} were discarded because the change set did not prove them:", ""]
        lines += [f"- {_cell(item['text'])} - {item['reason']}" for item in draft["rejected"][:8]]

    if draft.get("adjusted"):
        lines += ["", "Adjusted: " + "; ".join(f"{item['list']} #{item['index']}: {item['change']}" for item in draft["adjusted"][:6])]

    lines += ["", "## Review findings", ""]
    review = doc["review"]
    findings = doc["findings"]

    if review.get("batches"):
        reviewed = review.get("files_reviewed", 0)
        lines.append(f"{reviewed} of {len([f for f in doc['files'] if f['analysis'] == 'analyzed'])} analyzed files were reviewed by {review.get('model') or review.get('provider')} in {len(review['batches'])} batch(es), with project context from the RAG index.")
        lines.append("")

    if findings:
        lines += ["| ID | Severity | Kind | Category | Where | Finding | Validation |", "| --- | --- | --- | --- | --- | --- | --- |"]

        for item in findings:
            where = f"{item['file']}:{item['line']}" if item.get("file") and item.get("line") else item.get("file")
            lines.append(f"| {item['id']} | {item['severity']} | {item['kind']} | {item['category']} | {_cell(where)} | {_cell(item['title'])} | {item['validation_status']} |")

        lines += [""]

        for item in findings:
            lines += [f"**{item['id']} - {item['title']}** ({item['source']})", "", item["description"], ""]

            if item.get("evidence"):
                lines += [f"Evidence: {_code(item['evidence'][:300])}", ""]

            if item.get("recommendation"):
                lines += [f"Recommendation: {item['recommendation']}", ""]
    else:
        lines.append("No findings. This does not mean there are no problems; see the limitations below.")
        lines.append("")

    if review.get("architecture_notes"):
        lines += ["**Architecture notes**", "", _list(review["architecture_notes"]), ""]

    if review.get("not_reviewed"):
        lines += ["**Not reviewed**", "", _list(f"{_code(item['path'])}: {item['reason']}" for item in review["not_reviewed"][:12]), ""]

    lines += ["## Documentation check", ""]
    documentation = doc["documentation"]

    if documentation.get("flags"):
        lines += [f"{documentation.get('total_flagged', 0)} documents mention a changed file while they did not change themselves. These are questions for a person, not defects:", ""]
        lines += [f"- {_code(item['document'])} line {item['line']} mentions {', '.join(_code(path) for path in item['changed_files']) or ', '.join(_code(term) for term in item['mentions'])}" for item in documentation["flags"]]
    elif documentation:
        lines.append(f"Searched {documentation.get('terms', 0)} names from the changed files in {', '.join(documentation.get('documents', [])) or 'the documentation'}: no unchanged document mentions them." if documentation.get("terms") else documentation.get("note", "No documentation search was needed."))
    else:
        lines.append("No documentation check was made.")

    inspected = doc["inspection"].get("existing_release_documents", [])

    if inspected:
        lines += ["", f"Existing release documents (not modified): {', '.join(_code(path) for path in inspected)}"]

    lines += ["", "## Validation checks", ""]

    if doc["checks"]:
        lines += ["| Check | Status | Exit | Seconds | Command |", "| --- | --- | --- | --- | --- |"]

        for check in doc["checks"]:
            lines.append(f"| {check['id']} | **{check['status']}** | {_cell(check['exit_code'])} | {check['seconds'] if check['status'] not in ('not_run',) else '-'} | {_code(check['command'])} |")

        for check in doc["checks"]:
            if check["reason"] and check["status"] != "not_run":
                lines += ["", f"{check['id']}: {check['reason']}"]

            if check["errors"]:
                lines += ["", f"Error lines of {check['id']}:", ""] + [f"    {line}" for line in check["errors"][:8]]

        tree = doc.get("tree_after_checks") or {}

        if tree.get("compared"):
            lines += ["", "After the checks the working tree " + ("**was changed by them**: " + json.dumps(tree.get("differences", {}), ensure_ascii=False)[:300] if tree.get("changed") else "was the same as before.")]
    else:
        lines.append("No check is configured for this repository.")

    lines += ["", "## Blockers and unresolved questions", "", "**Blockers**", "", _list(readiness["blockers"]), "", "**Questions for a person**", "", _list(doc["unresolved"]), "", "## Limitations", "", _list(doc["limitations"]), ""]

    if doc["errors"]:
        lines += ["## Errors", "", _list(f"{item['stage']} / {item['kind']}: {item['message']}" for item in doc["errors"]), ""]

    lines += ["## Run", ""]
    llm = doc["llm"]
    lines += [
        f"- Provider: {llm.get('provider')} ({'local' if llm.get('local') else 'cloud'}) · models: {', '.join(llm.get('models', [])) or '-'}",
        f"- Model calls: {', '.join(call['stage'] + ' ' + str(round(call['ms'] / 1000, 1)) + 's' for call in llm.get('calls', [])) or 'none'}",
        f"- Artifacts in {_code(doc['run_dir'] or doc['output_dir'])}: {', '.join(_code(name) for name in (*NAMES, RECORD))}",
        "",
    ]

    return "\n".join(lines)


# --- changelog and release notes -------------------------------------------------------------------


def render_changelog(doc: dict) -> str:
    entries = doc["draft"].get("changelog", [])
    lines = [
        "# Changelog (draft)",
        "",
        f"> Generated by the AI Release Assistant, run {_code(doc['run_id'])}, from {_label(doc)}. A draft for a person to review - nothing was published.",
        "> Every entry is backed by a changed line or a commit subject; the evidence is listed in release_report.md.",
        "",
        f"## [Unreleased] - {_label(doc)}",
        "",
    ]

    if not entries:
        lines.append("_No changelog entry could be verified from this change set._" + (f" {_draft_note(doc)}" if _draft_note(doc) else ""))
        return "\n".join(lines) + "\n"

    for category in CATEGORY_ORDER:
        chosen = [item for item in entries if item["category"] == category]

        if chosen:
            lines += [f"### {category}", ""] + [f"- {item['text']} ({', '.join(_code(path) for path in item['files'])})" for item in chosen] + [""]

    return "\n".join(lines).rstrip() + "\n"


def render_notes(doc: dict) -> str:
    notes = doc["draft"].get("release_notes", [])
    repo = doc["repository"]
    versions = [fact for entry in doc["inspection"].get("version_files", []) for fact in [entry["facts"]] if fact.get("version_name")]
    lines = [f"# Release notes (draft) - {repo['title']}", ""]

    if versions:
        facts = versions[0]
        lines += [f"Version in the build file: {facts['version_name']}" + (f" (versionCode {facts['version_code']})" if facts.get("version_code") else "") + " - read from the build script; this tool does not change it.", ""]

    lines += ["> A draft for a person to review. Not published, and not a promise about any version.", "", "## What's new", ""]

    if notes:
        lines += [f"- {item['text']}" for item in notes]
    else:
        lines.append("_No user-visible change could be verified in this change set._" + (f" {_draft_note(doc)}" if _draft_note(doc) else ""))

    return "\n".join(lines) + "\n"


# --- the findings file -----------------------------------------------------------------------------


def render_findings(doc: dict) -> str:
    review = doc["review"]
    body = {
        "run_id": doc["run_id"],
        "schema": "Day 32 code-review response (summary, findings, architecture_notes, recommendations, limitations, provider, model, files); each finding also carries id, source and validation_status",
        "repository": doc["repository"]["name"],
        "scope": {"mode": doc["scope"]["mode"], "base_ref": doc["scope"]["base_ref"], "description": doc["scope"]["description"]},
        "status": doc["status"],
        "provider": review.get("provider") or doc["request"]["provider"],
        "model": review.get("model", ""),
        "summary": " ".join(review.get("summaries", [])).strip(),
        "findings": [
            {
                "id": item["id"],
                "category": item["category"],
                "severity": item["severity"],
                "kind": item["kind"],
                "file": item.get("file"),
                "line": item.get("line"),
                "title": item["title"],
                "description": item["description"],
                "recommendation": item.get("recommendation", ""),
                "evidence": item.get("evidence", ""),
                "sources": item.get("sources", []),
                "source": item["source"],
                "validation_status": item["validation_status"],
            }
            for item in doc["findings"]
        ],
        "architecture_notes": review.get("architecture_notes", []),
        "recommendations": review.get("recommendations", []),
        "limitations": doc["limitations"],
        "files": [
            {"path": item["path"], "status": item["status"], "additions": item["additions"], "deletions": item["deletions"], "analysis": item["analysis"], "withheld": item["withheld"], "reviewed": item["review"]}
            for item in doc["files"]
        ],
        "batches": review.get("batches", []),
        "context_sources": review.get("context_sources", []),
    }

    return json.dumps(body, ensure_ascii=False, indent=2) + "\n"


def render(doc: dict) -> dict[str, str]:
    """name -> text of the four artifacts (run.json is written by the assistant, last)."""
    return {
        REPORT: render_report(doc),
        CHANGELOG: render_changelog(doc),
        NOTES: render_notes(doc),
        FINDINGS: render_findings(doc),
    }


def render_record(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=2, default=str) + "\n"
