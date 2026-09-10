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
