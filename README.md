# Japanese Helper — Gemini Image API

FastAPI backend serving the Android client.

## Endpoints

| Method | Path | Request | Response |
| --- | --- | --- | --- |
| POST | `/image-search` | `{"query": "..."}` | raw JPEG bytes, `Content-Type: image/jpeg` |
| POST | `/description` | `{"meaning": "..."}` | `{"uncontrolled": "...", "controlled": "..."}` |
| POST | `/kanji-word-set` | `{"kanji": "...", "experimentType": "DIRECT\|STEP_BY_STEP\|PROMPT\|EXPERTS"}` | `{"prompt": "...", "words": ["..."], "cost": 0, "value": 0}` |

## Production entry point

```
app.main:app
```

The FastAPI instance is created in `app/main.py`. Root `main.py` is a
compatibility shim that re-exports the same object, so `main:app` also works;
new deployments should use `app.main:app`.

Production command:

```
/root/gemini-image-api/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Run it from the project root — `app/core/config.py` calls `load_dotenv()`, which
reads `.env` relative to the working directory.

## Deployment

The canonical unit file is `deploy/gemini-image-api.service`. To install or
update it on the server:

```
sudo cp deploy/gemini-image-api.service /etc/systemd/system/gemini-image-api.service
sudo systemctl daemon-reload
sudo systemctl restart gemini-image-api
```

Routine deploys afterwards:

```
git pull
sudo systemctl restart gemini-image-api
```

## Configuration

`.env` in the project root (never committed):

```
GEMINI_API_KEY=...
LOG_LEVEL=INFO                    # optional, defaults to INFO
AGENT_COMPRESSION_ENABLED=false   # optional, defaults to false
```

Paths for the agent's files (`AGENT_HISTORY_FILE_PATH`,
`AGENT_LONG_TERM_MEMORY_FILE_PATH`, `AGENT_USAGE_LOG_FILE_PATH`) and the sizes
of its memories (`AGENT_RECENT_MESSAGES_KEPT`, `AGENT_SUMMARY_UPDATE_THRESHOLD`,
`AGENT_FACTS_LIMIT`, `AGENT_MEMORY_ENTRY_LIMIT`) are all overridable the same
way, and all have working defaults.

`GEMINI_API_KEY` is read server-side only, in `app/core/config.py`, and is never
returned to a client or written to logs.

### Agent history compression

`AGENT_COMPRESSION_ENABLED` decides how much of the conversation `/agent/chat`
sends to Gemini. It is a flag rather than a request field because it exists to
be measured: run a dialogue with it off, run the same dialogue with it on, and
compare.

* `false` (default) - the whole conversation is sent on every turn, exactly as
  the agent has always worked. The prompt grows with the dialogue.
* `true` - only a running summary plus the messages still kept verbatim are
  sent. The newest 6 messages are always kept as they are; once 10 messages
  have aged out past that window they are folded into the summary by one extra
  Gemini call. `AGENT_RECENT_MESSAGES_KEPT` and
  `AGENT_SUMMARY_UPDATE_THRESHOLD` override those two numbers.

A client can pick the mode per request instead: `POST /agent/chat` accepts an
optional `compression_enabled` boolean, and the env flag is only the fallback
for requests that leave it out. The response carries a `compression` block
(`enabled`, `summary_tokens`, `messages_sent`) describing the context that
request sent - the summary that went with it, if any, and how many messages
went along word for word. It deliberately describes the request rather than
the conversation as it stands afterwards: on the turn that folds older
messages away, the tokens were still spent sending them, so the status and the
usage beside it always refer to the same request. The Android screen shows the
two together.

Note what this means for a short experiment: with the defaults, the first
summary is only written once sixteen messages exist - the eighth turn - and
the saving shows from the ninth. A dialogue shorter than that sends the same
context in both modes, and the status says so (`summary_tokens: 0`). Lower
`AGENT_SUMMARY_UPDATE_THRESHOLD` and `AGENT_RECENT_MESSAGES_KEPT` to see it
sooner.

### Context strategies

`PUT /agent/strategy` chooses how much of the conversation the next requests
send. The choice is persisted with the conversation, so it survives a restart.
A single request can override it with a `strategy` field; a client that sends
neither still falls back to the compression flag above, so nothing that
existed before this needs to change.

| strategy | what it puts in front of the model |
| --- | --- |
| `full` | the whole conversation - the baseline |
| `summary` | the running summary plus the messages still kept verbatim |
| `sliding_window` | only the newest `AGENT_RECENT_MESSAGES_KEPT` (6) messages |
| `sticky_facts` | a key-value memory of goals, constraints, preferences and decisions, plus that same window |
| `branching` | the current branch's whole conversation |

The last three never read or write a summary. `sliding_window` and
`sticky_facts` window the *context*, not the file: older messages stay on disk
so the same dialogue can be replayed under another strategy, but they are not
sent. `sticky_facts` refreshes its memory from every learner message with one
extra Gemini call, recorded separately as `facts_tokens` so the strategy's own
cost is visible next to what it saves.

### Branches and checkpoints

Branches exist under every strategy; `branching` is simply the strategy for
when the point is which branch is being talked on.

- `POST /agent/checkpoint` captures the current branch as it stands. It stores
  the conversation itself rather than a position in it, so a strategy that
  rewrites or shortens the branch afterwards cannot move the checkpoint.
- `POST /agent/branch` forks a branch from a checkpoint and deliberately does
  not switch to it - which is what makes forking a second branch from the same
  checkpoint straightforward. Two branches forked from one checkpoint start
  identical and then never touch each other.
- `PUT /agent/branch` switches. Each branch owns its messages and its facts,
  and `GET /agent/history` always means the branch being talked on.
- `GET /agent/context` returns exactly what the next request would send - built
  by the same code that builds the real prompt - plus the strategy, the current
  branch, and the branches and checkpoints available.

Branches, checkpoints, facts and the chosen strategy all live in
`data/agent_history.json` and come back after a restart.

### Memory layers

The strategies above all answer the same question - how much of the
conversation to send. `layered_memory` answers a different one: what kind of
thing is being remembered. Memory is split into three layers, stored apart and
sent apart, because they have different lifetimes.

| Layer | What it holds | How long it lives | Where it is kept |
| --- | --- | --- | --- |
| `short_term` | the current conversation, message by message | the dialogue | `data/agent_history.json`, in the branch |
| `working` | the current task: goals, requirements, constraints, decisions | the task | `data/agent_history.json`, beside the branches |
| `long_term` | the learner: profile, preferences, important decisions, knowledge | across dialogues and restarts | `data/agent_long_term_memory.json` |

Under `layered_memory` the prompt carries each layer as its own labelled
section - `LONG-TERM MEMORY`, `WORKING MEMORY`, `SHORT-TERM MEMORY`, in that
order - rather than one merged history, and an empty layer is left out
entirely. `GET /agent/context` shows the exact text.

What goes where is decided per message by `MemoryRouter`, through the same
`gemini_service` every other feature uses: wording like "remember for a long
time" routes to long-term, "for the current task" to working, and an ordinary
request ("explain this grammar") writes to neither - it is already the
conversation. The router names only the layers that change; a layer it does
not name keeps exactly what it had. Like summarising and fact-keeping it is
best-effort: a failed routing keeps the previous memory and is retried on the
next message rather than failing an answer that already worked. Its extra
Gemini call is recorded as `memory_tokens`, separately from the answer's own
usage.

- `GET /agent/memory` returns all three layers, each under its own key.
- `PUT /agent/memory/short_term` replaces the conversation;
  `PUT /agent/memory/working` and `PUT /agent/memory/long_term` update those
  layers, leaving out a field to keep its current value.
- `DELETE /agent/memory/{layer}` empties one layer and leaves the other two
  exactly as they are.

`DELETE /agent/history` ends the conversation, so it clears the two layers
scoped to one - short-term and working - and deliberately leaves long-term
memory alone. That is why long-term memory has its own file: clearing a
dialogue must never mean forgetting the learner.

Only `layered_memory` writes to these layers. Under every other strategy they
are read but never written, so nothing that worked before behaves differently.

The summary and the recent messages are persisted together in
`data/agent_history.json`, so a restart resumes the conversation either way,
and `GET /agent/history` returns both.

Every request's token usage is appended to `data/agent_token_usage.json`
(`AGENT_USAGE_LOG_FILE_PATH`) and is readable through `GET /agent/usage`. Each
record carries the flag it was produced under, and what the summarising itself
cost, so neither side of the comparison hides anything.

## Layout

```
main.py                       compatibility shim -> app.main:app
app/main.py                   FastAPI() + router registration
app/api/routes/               HTTP endpoints
app/schemas/                  Pydantic request/response models
app/services/                 Gemini calls, image handling, prompts (incl. kanji word set)
app/core/config.py            environment and constants
deploy/                       systemd unit
```
