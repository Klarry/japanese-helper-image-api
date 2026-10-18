# Project file assistant (Day 34)

`POST /project-files/task` takes a goal in words and works on the project's files itself: it decides what to search for and which files to read, reads them through MCP, and saves a report or a documentation patch. The caller names no files.

```
{task, provider}
  → plan (model): which workflow, the subject, search terms, the first tool calls
  → inspect (MCP project-files): search_in_files, search_project_files, read_project_file
        up to 2 rounds in which the model asks for more; agent rules fill the gaps
  → analyse (model): numbered extracts of the files read
  → check: every quoted line against the files as read
  → save (MCP project-files): write_project_file into output/; get_file_diff for a documentation change
```

Two workflows:

- **usage_report**: where a component or API is used and how a request flows through it.
- **docs_update**: documentation compared with the implementation. Verified discrepancies become a proposed file and a unified diff.

Anything else (changing code, running commands) is answered with `status: unsupported`.

## Tools: the `project-files` MCP server

`mcp_servers/project_files.py`, registered with the MCP client as `SERVERS["project-files"]` (like `project-git` and `support-crm`, not with the learning agent). Every rule lives in `app/services/project_files.py`; the server is a thin wrapper.

| Tool | What it does |
|---|---|
| `search_project_files(pattern, directory)` | Files whose path matches a glob (`*.md`) or contains a fragment |
| `search_in_files(query, file_pattern, directory, ignore_case)` | Lines that contain the text (literal, not a regex): path, line number, excerpt |
| `read_project_file(path, start_line, end_line)` | One text file, or a range of its lines, with the sha256 of the whole file |
| `write_project_file(path, content, expected_sha256, confirmed)` | Create or replace a file: freely in the output folder, elsewhere only with `confirmed` and the expected sha256 |
| `get_file_diff(path, proposed_content)` | A unified diff from the file as it is to the proposal; writes nothing |

There is no tool that runs anything or deletes anything. The model is shown only the three read tools; a call it names to any other tool is dropped and noted in the result. All writes are made by the assistant's own code.

## The rules

- **One root.** Paths are relative to `PROJECT_FILES_ROOT` (this repository by default). Absolute paths, `~`, `..` and backslashes are refused.
- **Symlinks.** A link is followed only to a file inside the root that could be read itself. A link that leads outside the root or to a secret is refused and is not listed. Writes never go through a link.
- **Ignore rules.** In a Git repository the files are the ones `git ls-files --cached --others --exclude-standard` lists; outside one, a walk of the tree. On top of that: `.git`, virtual environments, `node_modules`, caches, `build`, `dist`, `data/index` (`PROJECT_FILES_EXCLUDE`) and the output folder itself, so an earlier report never becomes evidence.
- **Secrets.** `.env*` (except `.env.example`), keys, keystores, `*credential*`, `*secret*`, `google-services.json` and similar names are never listed, searched, read or written. A file that contains a private key is refused by content.
- **Limits.** Files over `PROJECT_FILES_MAX_FILE_BYTES` (262144) are refused, not cut. Search results stop at `PROJECT_FILES_MAX_RESULTS` (200) and say so. Binary and non-UTF-8 files are skipped and counted.
- **Writes.** The output folder (`PROJECT_FILES_OUTPUT_DIR`, `output/`, git-ignored) is the assistant's to write. Any other file needs `confirmed: true`, and an existing one also needs the sha256 the change was prepared against: a file edited after it was read is not overwritten (`hash_mismatch`). Writes are atomic (temporary file and `os.replace`), keep CRLF line endings and a BOM, and log the path and hash, never the content.
- **Untrusted content.** File contents, comments and documentation reach the model as data, and every prompt says so. Nothing read is executed. An instruction hidden in a file has no write tool to use.

## Workflow A: usages

1. The search terms are the subject and its other spellings (`UserService` ↔ `user_service`). Every term is searched; a module-style name is also looked up as a file name.
2. The model asks for the files to read: the endpoint, the code in between, the subject's own file. If it reads no code, the files with the most matches are read.
3. The model explains the flow from numbered extracts. Every step must quote one line it was shown. A step whose quote is on a nearby line is reported at the real line ("cited as line N"); a step whose quote is not there, or comes from a file the model was not shown, is listed under **Not verified**.
4. The report, `output/<subject>_usage_report.md`, has the matches by file with line numbers and kind (import, call, definition, config, reference, docs, test), the checked flow, the unverified steps, the model's stated assumptions and the files inspected with their sha256.

The list of usages comes from the search, not from the model.

## Workflow B: documentation against the code

1. Documentation files that mention the subject are read and cut at their headings. The sections with the most mentions are compared, at most `PROJECT_FILES_MAX_SECTIONS` (4).
2. The names a section states (settings, paths, endpoints, flags) are searched in the code, and the code files that hold them are read.
3. For each section the model reports findings: a quote from the section, a quote from the code, the problem, a replacement.
4. Each finding is checked. The documentation quote must be in that section, and the code quote in the extracts the model was shown. A replacement that brings in a name, number or path that neither the section nor the extracts contain is kept out of the patch ("evidence only").
5. Accepted replacements become `output/proposals/<path>`. The diff comes from `get_file_diff` and is validated: applied to the original, it must give the proposal back. The results go to `output/documentation.patch` and `output/documentation_review.md`.

If nothing is verified, the review says so and no patch is written. The original documentation is never changed by the task.

## Applying a proposal

```
POST /project-files/apply   {"path": "README.md", "proposal": "output/proposals/README.md", "expected_sha256": "<base_sha256>", "confirm": false}
```

Without `confirm`, nothing is written and the reply is the diff. With `confirm: true`, the file is replaced only if it still has `expected_sha256`; otherwise the reply is 409 `hash_mismatch`. The proposal must be in the output folder, and the target must not be.

## Status and response codes

`status` in the body says how far the task got:

- `completed`: done;
- `partial`: something was saved, but a limit or an error stopped part of the work (see `errors`);
- `failed`: nothing was saved;
- `unsupported`: not one of the two workflows.

| Code | When |
|---|---|
| 200 | the task ran: `completed`, `partial`, `unsupported`, or `failed` for a reason other than an unavailable dependency |
| 422 | malformed request (an empty task, an unknown provider, extra fields such as a file list) |
| 503 | the MCP server or the named model was unavailable and nothing was produced — never a fallback to the other provider |

`/apply`: 200, 403 (`write_not_authorized`, `protected_path`, `secret_file`, `outside_root`), 404, 409 (`hash_mismatch`), 422, 503.

## Limits of a task

| Variable | Default | |
|---|---|---|
| `PROJECT_FILES_MAX_TOOL_CALLS` | 40 | 6 of them are kept for saving |
| `PROJECT_FILES_MAX_FILES` | 12 | files read |
| `PROJECT_FILES_MAX_INPUT_CHARS` | 250000 | characters read |
| `PROJECT_FILES_MAX_MODEL_CALLS` | 8 | |
| `PROJECT_FILES_MAX_SECONDS` | 600 | |
| `PROJECT_FILES_MAX_OUTPUT_CHARS` | 60000 | size of a saved report |
| `PROJECT_FILES_MAX_ROUNDS` | 2 | rounds in which the model asks for more files |
| `PROJECT_FILES_NUM_CTX` / `PROJECT_FILES_NUM_PREDICT` | 12288 / 2048 | context and answer size for the local model |

A task that reaches a limit stops inspecting, still saves what it found, and ends `partial` with the limit named in `errors`. The local model runs with the Day 29 `optimized` profile at temperature 0, answering in JSON.

## Reproducibility

Every run writes `output/runs/<run_id>.json`: the task, the provider and model, the plan, every tool call (who chose it, its arguments with file contents replaced by their size, the result in one line, the time), every model call, the files inspected with their sha256, the artifacts with their sha256, the diffs, the errors and the limits used. No file contents and no secrets are written there.

```
.venv-mac/bin/python -m app.project_task "<the task>"
.venv-mac/bin/python -m app.project_task --rerun output/runs/<run_id>.json   # same task and provider, then compared
.venv-mac/bin/python -m app.project_task --apply README.md --proposal output/proposals/README.md --expected-sha256 <sha>
.venv-mac/bin/python -m app.project_task --apply README.md --proposal output/proposals/README.md --expected-sha256 <sha> --confirm

curl -s http://127.0.0.1:8000/project-files/task -H 'Content-Type: application/json' \
  -d '{"task": "Find all usages of SupportAssistant and explain how a request reaches it.", "provider": "ollama"}'

.venv-mac/bin/python -m pytest tests/test_day34_file_assistant.py                 # temporary project, scripted model
RUN_DAY34_INTEGRATION=1 .venv-mac/bin/python -m pytest tests/test_day34_integration.py -v   # this repository, the real model
```

`--rerun` reports whether the files inspected, the saved artifacts (by sha256) and the diffs came out the same.

## Known limitations

- **Text search.** Search is literal: it finds names and phrases, not meaning. The RAG index is not used here; the assistant reads the files as they are now.
- **Small model.** Checks confirm that a quoted line exists where it is cited. They do not confirm that the explanation around it is right.
- **Sections, not whole documents.** Workflow B compares the sections that mention the subject most. Text that describes earlier days is history and is meant to be left alone, but the model can still flag it.
- **One process per call.** Every tool call starts the MCP server (about 0.6–1.2 s), the same client as the other servers.
- **Access.** No end-user authentication; `/apply` can change documentation with confirmation. For a trusted network (Day 30).
