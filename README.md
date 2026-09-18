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
`AGENT_FACTS_LIMIT`, `AGENT_MEMORY_ENTRY_LIMIT`, `AGENT_USER_PROFILE_FILE_PATH`,
`AGENT_PROFILE_PREFERENCES_LIMIT`, `AGENT_TASK_TRACKING_ENABLED`,
`AGENT_INVARIANTS_FILE_PATH`) are all overridable the same way, and all have
working defaults.

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

### User profile

The memory layers answer "what has been said". The profile answers "how this
learner wants to be answered": `preferred_language`, `japanese_level`,
`explanation_style`, `answer_format`, `translation_language`, and a free-form
list of extra `preferences`. That is a setting rather than something the
conversation mentioned, so it is kept out of all three layers, in its own file
`data/agent_user_profile.json` (`AGENT_USER_PROFILE_FILE_PATH`).

- `GET /agent/profile` returns it, `PUT /agent/profile` updates it (fields left
  out keep their current value; send an empty string to unset one), and
  `DELETE /agent/profile` unsets everything.
- Every `/agent/chat` request reads it and, if anything is set, sends it as a
  `USER PROFILE` block - under *every* strategy, since it is not a strategy and
  does not belong to one. The learner never repeats their level or format.
- An unset profile sends no block at all, so an agent without one gets exactly
  the prompt it got before.

The block sits after the context and immediately before the new message: it is
the last thing the model reads before the question, and it stays out of the text
counted as `history_tokens`, because history is what the conversation produced
and the profile is a setting.

Keeping it apart cuts both ways - clearing the conversation or any memory layer
leaves the profile alone, and setting a profile puts no words into a
conversation the learner never had. Same request, two profiles, two different
answers: the only difference in what reaches Gemini is that block.

### Task state machine

Memory says what was said and the profile says how to answer; neither says what
the learner and the agent are in the middle of. A task has stages, and they only
ever go one way:

```
planning -> execution -> validation -> done
```

Those three moves, plus starting a task (`idle -> planning`) and staying put, are
the only legal ones. Skipping validation, going back to planning halfway through
execution, or reopening a finished task are all refused - and refused whole: a
proposal that jumps to `done` describes a task that ended, so keeping its step
text while rejecting its stage would leave the two contradicting each other.

Alongside the stage, `current_step` says what is being done now and
`expected_action` what should happen next. All three travel with the conversation
in `data/agent_history.json`, beside the branches and never inside a transcript,
so a task survives the app being closed and ends when the conversation is cleared.

Where the task has got to is *proposed* by `TaskTracker` - one call through the
same `gemini_service`, recorded separately as `task_tokens` - and accepted only
if the state machine allows that move, so a confused answer can never put the
task somewhere impossible. Like every other extra call here it is best-effort: a
failed update leaves the task exactly where it was.

- `GET /agent/task` returns the stage, both step fields, and `allowed_next` - the
  machine reported rather than documented, so a client never guesses - plus
  `plan`, `validation_passed`, `next_requirement` and the last refused move.
- `DELETE /agent/task` ends the task and leaves the conversation, the memory
  layers and the profile untouched.

While a task is running its state is sent as a `TASK STATE` block, last of all,
immediately before the new message: "carry on from this exact step, do not start
over" is the final thing the model reads. An idle task sends nothing. Tracking
can be switched off with `AGENT_TASK_TRACKING_ENABLED=false`, which is how the
earlier days' token comparisons are re-run without the extra call.

#### Controlled transitions

The order above is only half a rule: a move can be legal in shape and still
wrong in fact. Execution may only begin once a plan has actually been approved,
and `done` may only be reached once a validation has actually passed. So every
change of stage - asked for through the API or proposed by the tracker - goes
through one check, `check_transition`, which asks both questions: is this edge
in the table, and is the current stage's own work finished.

A refused move changes nothing about where the task is. It answers with `409`
and the four things a refusal owes the caller:

```json
{
  "message": "The task is in 'execution' and cannot move to 'done': from 'execution' the only next stage is 'validation', and the task has not been through validation yet.",
  "current_stage": "execution",
  "requested_stage": "done",
  "required_next": ["validation"],
  "unmet_condition": "the task has not been through validation yet"
}
```

The same refusal is kept on the task as `blocked` until the next move that
works, which is what makes it visible rather than silent: the screen keeps
showing why the task did not advance, and the refusal is added to the
`TASK STATE` block so the next answer explains it to the learner in words
instead of quietly doing nothing.

- `POST /agent/task/transition` - `{"task_stage": ..., "current_step": ...,
  "expected_action": ...}`. An unknown stage is rejected as `422` before it
  reaches the machine; a disallowed one as `409`.
- `POST /agent/task/plan` - `{"plan": "..."}`, the condition for leaving
  planning. `409` unless the task is planning: a plan approved from anywhere
  else would be a way round the very condition it exists to satisfy.
- `POST /agent/task/validation` - `{"passed": true|false, "notes": "..."}`, the
  condition for reaching done. A failed check is recorded too - "checked, not
  right yet" is not "not checked", and neither opens the way to `done`.

The agent reaches the same two records through the tracker, which reports a plan
the learner approved or a check that came out right alongside the move it
proposes. Both are only accepted in the stage that produces them, so the model
cannot approve a plan retrospectively to unlock a stage it is already past.

### Invariants

The layers above say what was said, how to answer and where the work has got to.
None of them says what must never happen - so "move the Gemini call into the app"
is a perfectly coherent suggestion an agent will happily help with.

Invariants are their own layer: project rules in four categories -
`architecture`, `technology_stack`, `technical_decisions`, `business_rules` - in
their own file `data/agent_invariants.json` (`AGENT_INVARIANTS_FILE_PATH`), apart
from the conversation and from every memory layer. They are not memory: nothing
said in a conversation writes them, and clearing a conversation, a layer, the
task or the profile cannot forget them. They change deliberately, through the
API, because that is what a rule is.

The file is seeded with this project's own rules the first time it is read, so a
fresh install starts out constrained - ViewModel → Repository → API, FastAPI,
Gemini, JSON storage, Kotlin, the LLM called only from the backend, no SQLite, no
second LLM integration. A rule deleted afterwards stays deleted.

- `GET /agent/invariants` returns every rule with its id and category.
- `POST /agent/invariants` adds one (the id is generated from the category);
  `PUT /agent/invariants/{id}` adds or changes that rule, keeping its place in
  the list; `DELETE /agent/invariants/{id}` removes it, and an unknown id is a
  404 rather than a silent success.

Every request carries them as an `INVARIANTS` block, grouped by category, ahead
of the profile and the task state - hardest rule first. The block carries the
protocol as well as the rules: if the request would break one, name the rule,
explain why it exists, and offer a permitted alternative instead of helping to
work around it. What the backend guarantees is that the rules and that protocol
reach the model on every single request; the refusal itself is the model's, so it
is worth reading a live answer rather than assuming one.

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
