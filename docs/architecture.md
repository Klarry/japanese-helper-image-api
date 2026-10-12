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
| project-git | `mcp_servers/project_git.py` | `get_git_branch` (developer assistant only) |

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

## Storage

There is no database. State is kept in JSON files under `data/`: conversation history, long-term memory, user profile, invariants, token usage, digest tasks and results, pipeline output and local-chat conversations. Each storage module treats a missing or broken file as empty rather than failing.
