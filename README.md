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
LOG_LEVEL=INFO        # optional, defaults to INFO
```

`GEMINI_API_KEY` is read server-side only, in `app/core/config.py`, and is never
returned to a client or written to logs.

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
