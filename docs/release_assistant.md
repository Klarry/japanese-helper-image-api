# AI Release Assistant (Day 35)

The task this solves: **preparing a release draft of Japanese Helper** (the FastAPI backend and the Android app). Before a release somebody has to read the Git changes, look for problems, check whether the documentation still matches, and write a changelog, release notes and a status report. The assistant does that reading and writing. A person still decides whether to release, and does the release.

```
{base_ref, provider, repository, include_uncommitted, run_checks, checks}
  → inspect  (MCP project-git, project-files): branch, status, upstream, tools, RAG indexes, version files
  → collect  (MCP project-git): committed / uncommitted / untracked changes, diff, recent commits
  → filter   (release_safety): secrets, generated and binary files and credential-looking diffs never reach a prompt
  → review   (Day 32 CodeReviewer + RAG): findings tied to changed files and to lines of the diff
  → docs     (MCP project-files): documents that mention changed files, flagged "needs a human check"
  → checks   (only if asked): commands listed in data/release/release_config.json, no shell
  → draft    (model): changelog + release notes, every entry tied to changed files, unsupported ones dropped
  → save     (MCP project-files): output/release/<run_id>/ - report, CHANGELOG, release notes, findings, run.json
```

Nothing in this flow commits, stashes, resets, cleans, tags, builds a release artifact or publishes. There is no such step to call.

## What is reused

| Need | Existing component |
|---|---|
| Review of the changes | `app/services/code_review.py` (Day 32): the same `CodeReviewer`, categories, severities and validated findings |
| Project documentation as context | the RAG pipeline and the `data/index/*` FAISS indexes, through the reviewer's retriever |
| Reading Git | the `project-git` MCP server (`mcp_servers/project_git.py`), extended in Day 35 with read-only tools |
| Reading and writing files | the `project-files` MCP server (Day 34) and its rules in `app/services/project_files.py` |
| Models | `app/services/llm_provider.py`: Ollama by default; the named provider is never replaced by the other one |
| HTTP guard | the `/code-review` rule: `CODE_REVIEW_TOKEN` as a bearer token, or this machine only |

New in Day 35: `app/services/release_assistant.py` (the orchestration), `release_draft.py` (the release prompt, parsing and evidence checks), `release_checks.py` (opt-in checks), `release_artifacts.py` (the five files), `release_safety.py`, `release_config.py`, `app/api/routes/release.py`, `app/release_assistant.py` (CLI).

## Tools added to `project-git`

| Tool | What it does |
|---|---|
| `get_git_branch()` | unchanged (Day 31) |
| `get_git_status()` | branch, HEAD, upstream, counts of staged / unstaged / untracked / conflicted files, the files |
| `resolve_git_ref(ref)` | whether a reference exists, and the commit it points at |
| `get_git_changes(base_ref)` | files changed in `base...HEAD`, uncommitted files, untracked files, each with status and line counts |
| `get_git_diff(base_ref, scope, paths, max_chars)` | the diff of `committed`, `uncommitted` or `combined` changes, per file, size-limited |
| `get_recent_commits(base_ref, limit)` | commit subjects of the range (or the latest ones when no base was given) |

Only `rev-parse`, `branch`, `status`, `rev-list`, `merge-base`, `diff` and `log` are ever run. The repository to read comes from `PROJECT_GIT_ROOT`, set by the orchestrator for the server process; the model cannot change it.

## The release range

- **With `base_ref`**: the committed changes `base_ref...HEAD`. Uncommitted and untracked files are listed but not analyzed, because they may not belong to the release. `include_uncommitted` adds the uncommitted ones on purpose.
- **Without `base_ref`**: the uncommitted changes of the working tree against HEAD are analyzed and the report says so. No range is guessed. The latest commits are shown for information only and are never used as evidence for the changelog.
- A dirty working tree is reported, never cleaned. An unknown base reference stops the run with `invalid_base_ref` (HTTP 422) before any model is called.

## What a model may see

The prompt receives diffs, commit subjects and short documentation excerpts. It never receives:

- files whose name says they hold secrets (`.env*`, `*.pem`, `google-services.json`, `local.properties`, keystores, …);
- generated, build, cache and lock files and binary files (listed by path, never shown);
- a diff that looks like it contains a credential (private key, `AIza…`, `sk-…`, `ghp_…`, AWS, Slack, JWT, bearer tokens, `password = "…"` with a real-looking value);
- a diff larger than the per-file limit.

The filter is conservative: a false alarm withholds one file's diff and says so in the report. Everything written to an artifact goes through `redact` as well. Repository text is treated as untrusted data: the release prompt says so, and instructions inside a diff or a document are not followed.

## Review

The changed files are reviewed in up to three batches (about 12 000 characters each, source files first) by the Day 32 `CodeReviewer` with the chosen provider and the RAG context. A finding is kept only if its file is a changed file and its evidence is a changed line; the others are dropped and counted. If RAG is unavailable the reviewer records it and the report lists it as a limitation. Files that did not fit into the batches are listed as not reviewed.

Besides the model's findings the assistant adds deterministic ones, labelled by source: a documentation page that mentions a changed path or module name ("needs a human check", never claimed as a defect), source changed with no test changed, and a failed check.

## Checks

`data/release/release_config.json` lists the commands the assistant may run, as argument lists. Nothing runs unless `run_checks` / `--run-checks` is given. A check runs without a shell, with a small environment allowlist (no API keys), a timeout and an output tail kept. Each check ends as `passed`, `failed` (with its exit status and error lines), `could_not_run` (missing tool, for example no JDK) or `not_run` (not requested, or its precondition is missing). A failed check is reported, not repaired. The report never says that a check passed unless it was executed and exited with 0. After the checks the working tree is compared with its state before, and a change is a warning.

## Readiness

The verdict is rule-based, not the model's opinion:

- `blocked`: a failed check, a critical or high defect found by the review, or merge conflicts;
- `not_verified`: no check passed (none was run, or none could run), or the model was unavailable;
- `ready_with_warnings`: the checks that ran passed, but warnings remain (uncommitted files, withheld files, documents to check, unreviewed files);
- `ready`: every configured check passed and nothing needs attention;
- `nothing_to_release`: no changes in the selected scope.

It is a hint for a person. The CLI exit code reports whether the run itself finished, not whether to release.

## Artifacts

Each run writes into its own folder, `output/release/<run_id>/` (the folder is created with a new unique name; an existing file is never overwritten):

| File | Content |
|---|---|
| `release_report.md` | scope, git state at start, files, review, documentation, checks, findings, readiness, limitations, questions for a person |
| `CHANGELOG.md` | Added / Changed / Fixed / Removed / Security; only entries tied to changed files and to the diff or commit subjects |
| `release_notes.md` | only entries about user-visible changes, in the paths the repository configures as user-facing; otherwise it says there are none |
| `review_findings.json` | the findings in the Day 32 schema, plus the source of each |
| `run.json` | run ID, request, refs, git status at start, files, tool calls, checks, provider and models, errors, limitations, artifact paths with sha256 |

Artifacts go only below `output/` (git-ignored) through the file tools; `--output-dir` / the configuration may name another folder inside `output/`. Existing user-written release documents in the analyzed repository (`CHANGELOG.md`, `RELEASE_NOTES.md`) are read for context and never changed. A later run creates a new folder and a new report.

## Providers

`ollama` is the default and the preferred provider. If Ollama is not running the run ends with status `failed`, `run.json` and the report still record it, the API answers 503 and the CLI exits with 3. It does not switch to Gemini. With `provider: gemini` the same prompt runs on Gemini and the report says it is a cloud model. The release prompt is separate from the chat, support and code-review prompts.

## Using it

```
.venv-mac/bin/python -m app.release_assistant --base-ref main --provider ollama
.venv-mac/bin/python -m app.release_assistant --base-ref v1.0 --run-checks backend-compile
.venv-mac/bin/python -m app.release_assistant --repo android --base-ref HEAD~5 --run-checks android-unit-tests android-detekt
.venv-mac/bin/python -m app.release_assistant                    # no base: the working tree against HEAD
```

```
curl -s -X POST http://127.0.0.1:8000/release/prepare \
  -H 'Content-Type: application/json' \
  -d '{"base_ref": "main", "provider": "ollama"}'
```

Request fields: `base_ref`, `provider` (`ollama` or `gemini`), `repository` (a name from the configuration), `include_uncommitted`, `run_checks`, `checks` (ids). Unknown fields are refused. Answers: 200 with the run record, 401/403 (not allowed), 422 (unknown repository, base reference or check id, refused output folder), 503 (busy, an MCP server that will not start, or the model unavailable).

CLI exit codes: 0 completed or nothing to release, 1 partial or failed for another reason, 2 the request was refused, 3 the model was unavailable.

## Configuration

`data/release/release_config.json` is edited by a person and never by the model or by the repository being released. For each repository: its path, the documentation files to compare with, the paths that count as user-facing, the files that hold the version. For each check: an id, the repository, the argument list, a timeout, and what must exist for it to run. `{python}` stands for the interpreter that runs the assistant.

## Tests

`tests/test_day35_release_assistant.py` (120 tests) builds temporary Git repositories and runs the real Git MCP code in-process with a scripted model, so no network and no Ollama are needed. They cover: a clean repository, a dirty tree, no relevant changes, a missing and an unknown base reference, RAG unavailable, review findings (and invented ones being dropped), failed checks and checks that could not run, existing output files, path restrictions, an unavailable local model with no cloud fallback, no invented test results, and the structure of the artifacts. A trap fails the test if any Gemini function is called while Ollama is selected.

`tests/test_day35_integration.py` is the real run: Ollama, the RAG index and the MCP servers against a real repository. It is skipped unless `RUN_DAY35_INTEGRATION=1`.

```
RUN_DAY35_INTEGRATION=1 OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M \
  .venv-mac/bin/python -m pytest tests/test_day35_integration.py -v
```

## Limitations

- The review is a language model's: it can miss problems, and its severities are judgments. Findings are checked against the diff, not run.
- The documentation check is a search for changed paths and names in the documents. It points a person to pages, it does not decide whether a page is wrong.
- Release notes need paths marked as user-facing in the configuration. The backend has none, so its notes say that no user-visible change was found.
- The filter withholds large and credential-looking diffs; those files are not reviewed. The review covers at most three batches by default (`--max-review-batches`).
- The version is read from the files named in the configuration (the Android `app/build.gradle.kts`); the assistant never changes it.
- With the local provider the RAG query embedding still uses `gemini-embedding-001` (the structural index was built with it); only titles, paths and identifiers are sent. The model that writes and reviews stays local.
- The Gradle checks need a JDK and the Android SDK on the machine that runs the assistant. The backend test check runs the whole suite and takes minutes.
- One run at a time over HTTP (a second request gets 503 `busy`).
