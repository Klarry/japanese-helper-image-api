# Architecture

Japanese Helper has two parts. The Android client is written in Kotlin: ViewModel → Repository → Retrofit API. The Python FastAPI backend is in this repository. The project started as a Japanese-learning app: image search, kanji word sets and descriptions through Gemini. Over time it grew an AI agent with memory, MCP tools, a RAG pipeline over the project's own files, a local LLM through Ollama, and a developer assistant.

## Repository layout

| Path | What it holds |
|---|---|
| `app/main.py` | The FastAPI app. It registers the routers and starts the digest scheduler in its lifespan. |
| `app/api/routes/` | HTTP endpoints, one module per area: `agent.py`, `rag_chat.py`, `local_llm.py`, `dev_assistant.py`, … |
| `app/schemas/` | Pydantic request and response models |
| `app/services/` | The logic: model clients, the agent, memory, RAG, MCP client, providers |
| `app/core/config.py` | Settings read from the environment. `GEMINI_API_KEY` is required at import time. |
| `mcp_servers/` | MCP servers. Each runs as its own subprocess over stdio. |
| `data/` | Runtime state: JSON files, the FAISS index in `data/index/`, and the extra corpus in `data/documents/` (Android sources, PDF overviews) |
| `docs/` | This documentation: architecture, the generated API reference (`api.md`) and per-feature guides |
| `deploy/` | The systemd unit (`gemini-image-api.service`) for the VPS |
| `tests/` | pytest. External services are replaced by stand-ins. |

## Models and providers

- **Gemini.** `app/services/gemini_service.py` is the only client for the hosted model. Text uses `gemini-3.5-flash`; embeddings use `gemini-embedding-001`.
- **Ollama.** `app/services/ollama_service.py` is the only client for the local model. The default is `qwen3:4b`; `OLLAMA_MODEL` overrides it. Ollama listens on `localhost:11434` and is reached only by the backend.
- **Providers.** `app/services/llm_provider.py` puts both behind one interface: `GeminiProvider` and `OllamaProvider`, each with `generate(prompt)`. `provider_for(name)` refuses an unknown name instead of picking a default. No path ever falls back from one provider to the other.
- **Profiles.** `app/services/ollama_profiles.py` names the ways the local model is asked: `original`, `baseline`, `optimized-params`, `optimized`. A profile sets temperature, `num_predict`, `num_ctx`, thinking and the prompt version. `OLLAMA_PROFILE` selects one.

## The agent (`/agent/chat`)

`app/services/japanese_learning_agent.py` owns one request from start to finish: context, prompt, Gemini call, history update and response. Its parts:

- **History.** `agent_history_storage.py` keeps one JSON file with branches and checkpoints. It is also the storage the mini chat and the local chat reuse.
- **Memory.** `agent_memory.py` has three layers: short-term (the conversation), working (the current task) and long-term (the learner). `agent_memory_router.py` decides which layer a message belongs to.
- **Task state.** `agent_task_state.py` and `agent_task_tracker.py` form a state machine for multi-step tasks. Invariants live in `agent_invariants.py`.
- **Tools.** `agent_tool_planner.py` lets the model ask for MCP tools. `mcp_registry.py` discovers which registered server offers which tool. `mcp_client.py` starts the server and calls the tool.

## MCP servers

| Server | Module | Tools |
|---|---|---|
| japanese-data | `mcp_servers/japanese_data.py` | `search` (JLPT vocabulary API) |
| processing | `mcp_servers/processing.py` | `summarize` |
| storage | `mcp_servers/storage.py` | `save_to_file` |
| jlpt-vocab | `mcp_servers/jlpt_vocab.py` | `get_japanese_word_info`, `create_periodic_digest`, `get_latest_digest` |
| project-git | `mcp_servers/project_git.py` | `get_git_branch` (developer assistant), `get_git_status`, `resolve_git_ref`, `get_git_changes`, `get_git_diff`, `get_recent_commits` (release assistant; read-only) |
| support-crm | `mcp_servers/support_crm.py` | `get_ticket`, `get_user`, `get_user_tickets` (support assistant only) |
| project-files | `mcp_servers/project_files.py` | `search_project_files`, `search_in_files`, `read_project_file`, `write_project_file`, `get_file_diff` (file assistant only) |

The first four make up the agent's registry. `project-git` is registered with the MCP client (`SERVERS["project-git"]`) but not with the agent: the learning agent has no use for a branch name.

## RAG

- **Indexing.** `python -m app.index_documents` loads the corpus (`document_loader.py`), cuts it into chunks (`chunking.py`, fixed or structural), embeds them (`embedding_service.py`) and writes a FAISS index plus metadata to `data/index/<strategy>/`. The corpus is `README.md`, `docs/*.md`, `app/**/*.py`, `mcp_servers/*.py` and `data/documents/`. Every chunk keeps its file, section and `chunk_id`.
- **Query.** For one question the pipeline runs in this order:
  1. Optional query rewrite (`query_rewriter.py`).
  2. FAISS top-k search (`rag_retriever.py`).
  3. Similarity filter and rerank (`rag_enhanced.py`, `reranker.py`).
  4. Relevance gate. Below the threshold the answer is "I don't know" and no model is called.
  5. Cited prompt (`rag_prompt.py`).
  6. Citation validation (`rag_citations.py`): every quote must appear verbatim in the chunk it cites, otherwise it is dropped. Only whitespace and Markdown inline markup (backticks, asterisks) are ignored; the kept quote is the chunk's own text.
  7. Claim-support check (`claim_support.py`).
- **Endpoints.** `POST /rag/chat` takes `provider: gemini | ollama` and passes it to all three model calls.

## Local LLM service (`/local-chat`)

`app/api/routes/local_llm.py` puts guards from `local_chat_guard.py` in front of the model. The checks are:

- rate limit per client address → 429;
- message size and context budget → 413;
- at most N generations at once → 503 busy;
- timeout.

`GET /local-llm/health?strict=true` returns 503 unless the model can actually answer. `python -m app.serve` starts the app on the network interface. Details are in `docs/day30_private_service.md`.

## Developer assistant (`/dev/chat`, `/help`)

`app/services/dev_assistant.py` answers `/help` commands:

- Questions about the documentation go through the RAG agent above, with sources. If the rewritten search query keeps nothing at the similarity filter, the question itself is searched once more against the same threshold (`retry_with_question`).
- Questions about the current branch go to the `get_git_branch` MCP tool.
- A combined question uses both.
- `/help` alone lists what the assistant can do.

The CLI is `python -m app.dev_help "/help …"`.

## Code review (`/code-review`, Day 32)

`app/services/code_review.py` reviews a pull request's unified diff. It parses the diff into files and numbered
lines (`code_review_diff.py`) and adds context from the index, then asks the provider for a JSON review. The
context has two parts: chunks of the changed files that contain lines of the diff (exact text) and a vector search
through the RAG pipeline above with the query rewrite turned off. Findings are validated against the diff:
category, severity, file, evidence and line. `.github/workflows/ai-code-review.yml` runs
`app/github_review.py` on a self-hosted runner. The script fetches the diff from GitHub, calls the endpoint and
keeps one comment per pull request up to date. Details are in `docs/code_review.md`.

## Support assistant (`/support/chat`, Day 33)

`app/services/support_assistant.py` works in three steps:

1. Reads the ticket and its user through the `support-crm` MCP server. The user is the one the ticket names, verified to match.
2. Searches the help centre (`docs/support`, a separate index with local embeddings) through the RAG pipeline above.
3. Answers with a support-specific prompt through the named provider. Citations are validated as in `/rag/chat`.

Details are in `docs/support_assistant.md`.

## Project file assistant (`/project-files/task`, Day 34)

`app/services/file_assistant.py` takes a goal in words and chooses the files itself:

1. The model plans: the workflow (usages of a component, or documentation against the code), the subject, the first searches.
2. The files are searched and read through the `project-files` MCP server. The model asks for more in up to two rounds; agent rules fill the gaps. The model is shown only the read tools.
3. The model analyses numbered extracts. Every quoted line is checked against the files as read.
4. The report, or the proposed documentation and its diff, is written to `output/`. A project file changes only through `/project-files/apply`, with confirmation and the expected sha256.

The rules for paths, secrets, symlinks, limits and writes are in `app/services/project_files.py`. Details are in `docs/project_files_assistant.md`.

## AI Release Assistant (`/release/prepare`, Day 35)

`app/services/release_assistant.py` prepares a release draft and changes nothing in the repository it reads:

1. Inspects the repository through `project-git` (branch, status, refs, changes) and `project-files` (version files, documentation list), and reads the RAG index metadata.
2. Collects the change set for `base...HEAD` (or the working tree when no base is given), separating committed, uncommitted and untracked files. `release_safety.py` withholds secrets, generated and binary files and credential-looking diffs before anything reaches a prompt.
3. Reviews it with the Day 32 `CodeReviewer` (RAG context included) in up to three batches. Findings must point at a changed file and a changed line.
4. Searches the documentation for changed paths and names, runs the checks from `data/release/release_config.json` only if asked, and computes a rule-based readiness verdict.
5. Drafts the changelog and release notes with a release-specific prompt (`release_draft.py`); entries without evidence in the diff or the commit subjects are dropped.
6. Saves the report, changelog, release notes, findings and `run.json` to `output/release/<run_id>/` through `project-files`.

The local model is never replaced by Gemini. Details are in `docs/release_assistant.md`.

## Storage

There is no database. State is kept in JSON files under `data/`: conversation history, long-term memory, user profile, invariants, token usage, digest tasks and results, pipeline output and local-chat conversations. Each storage module treats a missing or broken file as empty rather than failing.
