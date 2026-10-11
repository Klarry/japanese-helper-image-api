# Day 30 · The local LLM as a private network service

The model that answers is the one Days 26-29 set up - Ollama on this
machine, reached only through `app/services/ollama_service.py`. Day 30 does
not add a second backend or a second client: it puts guards in front of the
existing `POST /local-chat`, makes the health check say whether the *model*
is ready, and starts the same FastAPI app on the network interface.

## Architecture

```
phone / laptop on the LAN
        │  HTTP  :8000
        ▼
FastAPI  app.main:app  (one worker)
  POST /local-chat
    rate limit (per client address) → 429 + Retry-After
    message length                  → 413 input_too_large
    context budget (est. tokens)    → 413 context_too_large
    conversation already answering  → 409 conversation_busy
    generation slots (max N)        → 503 busy
    timeout                         → 503 timeout
        │  ollama_service.generate(...)   ← the Day 26 client, unchanged
        ▼
Ollama  127.0.0.1:11434   ← loopback only, never exposed
```

No route in this path imports Gemini (a test parses the imports), and no
failure is answered by another model: every refusal is the same JSON shape
with `provider: "ollama"` and no `response` field.

## Configuration

| Variable | Default | What it limits |
|---|---|---|
| `SERVICE_HOST` / `SERVICE_PORT` | `0.0.0.0` / `8000` | where `python -m app.serve` listens |
| `LOCAL_CHAT_RATE_LIMIT` | `10` | chat requests per client address per window |
| `LOCAL_CHAT_RATE_WINDOW_SECONDS` | `60` | the sliding window |
| `LOCAL_CHAT_MAX_MESSAGE_CHARS` | `4000` | the message, in characters |
| `LOCAL_CHAT_MAX_CONCURRENT` | `2` | generations running at once |
| `LOCAL_CHAT_TIMEOUT_SECONDS` | `120` | one generation, then the request is cancelled |
| `LOCAL_CHAT_DEFAULT_NUM_CTX` | `4096` | context window when the profile sets none |
| `LOCAL_CHAT_OUTPUT_RESERVE_TOKENS` | `1024` | tokens kept free for the answer when the profile sets no `num_predict` |
| `LOCAL_CHAT_RECENT_MESSAGES` | `8` | earlier messages a conversation may bring into the prompt |
| `LOCAL_CHAT_HISTORY_DIR` | `data/local_chat` | one file per conversation (Day 7 history storage) |
| `LOCAL_CHAT_TRUSTED_PROXIES` | empty | proxy addresses whose `X-Forwarded-For` is believed |
| `OLLAMA_MODEL`, `OLLAMA_PROFILE`, `OLLAMA_NUM_CTX`, `OLLAMA_NUM_PREDICT`, … | Day 26 / 29 | which model and how it generates |

The generation settings are Day 29's: the active profile decides
temperature, `num_predict` and thinking. What Day 30 changes is that
`num_ctx` is always sent - the profile's value, or 4096 - instead of being
left to the server's default.

## Start it

```bash
# 1. Ollama (stays on 127.0.0.1:11434 - do NOT set OLLAMA_HOST=0.0.0.0)
ollama serve                      # or open the Ollama app
lsof -nP -iTCP:11434 -sTCP:LISTEN # should show 127.0.0.1:11434

# 2. The backend, on the network interface, one worker
cd "$HOME/Desktop/AI Challenge/Week 1/japanese-helper-image-api"
.venv-mac/bin/python -m app.serve
# same as: .venv-mac/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# Day 29's fast configuration, if wanted:
OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M OLLAMA_PROFILE=optimized .venv-mac/bin/python -m app.serve
```

`app.serve` prints the LAN address it found. By hand: `ipconfig getifaddr en0`
(macOS, Wi-Fi) or `hostname -I` (Linux).

## Reach it from another device

```bash
curl -s "http://192.168.10.35:8000/local-llm/health?strict=true" | python3 -m json.tool
curl -s http://192.168.10.35:8000/local-chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What is MCP? One sentence."}' | python3 -m json.tool

# a conversation: the second message sees the first
curl -s http://192.168.10.35:8000/local-chat -H 'Content-Type: application/json' \
  -d '{"message": "My name is Klara.", "conversation_id": "demo"}'
curl -s http://192.168.10.35:8000/local-chat -H 'Content-Type: application/json' \
  -d '{"message": "What is my name?", "conversation_id": "demo"}'
```

The Android app's "Local LLM" chip (Day 27) uses the same endpoint, so the
phone on the same Wi-Fi is another client.

A home address like `192.168.x.x` is reachable only inside that network.
It is not public, and nothing here makes it so.

## Response

```json
{
  "response": "…", "model": "qwen3:4b", "provider": "ollama", "seconds": 2.1,
  "request_id": "7c5dc73d2d80", "conversation_id": null,
  "timing": {"generation_ms": 2104, "total_ms": 2110},
  "context": {"estimated_prompt_tokens": 12, "prompt_budget_tokens": 3072,
              "history_messages_used": 0, "history_messages_dropped": 0},
  "usage": {"prompt_tokens": 14, "response_tokens": 40, "tokens_per_second": 58.3,
            "options": {"options": {"num_ctx": 4096}}}
}
```

The first four fields are Day 27's, so the app is unaffected. Every response
and every refusal carries `request_id`, also sent as the `X-Request-ID`
header. The same id is on the server's log line:

```
[LocalService] request_id=7c5dc73d2d80 endpoint=/local-chat client=192.168.10.40 model=qwen3:4b outcome=ok status=200 total_ms=2110 generation_ms=2104 prompt_tokens=14 response_tokens=40 history=0
```

Log lines carry ids, sizes and times. They never carry what was said.

## Limits, and what they are not

- **Rate limit.** A sliding window per client address, in memory. It is
  per process and resets when the process restarts. `X-Forwarded-For` is
  ignored unless the request comes from an address listed in
  `LOCAL_CHAT_TRUSTED_PROXIES`, because a header any client can write is
  not an identity.
- **Input vs context.** 4000 *characters* is checked first. The context
  budget is in *tokens*, estimated: kana and kanji count as one token each,
  anything else as one token per three characters. This is deliberately
  high, and it is not a tokenizer. Ollama's real count comes back as
  `usage.prompt_tokens`. 3500 characters of Japanese pass the first check
  and fail the second.
- **History.** With a `conversation_id`, earlier turns are added newest
  first while they fit `num_ctx − reserve`. The rest are dropped and
  counted in `history_messages_dropped`. The current message is never cut:
  if it alone does not fit, the request gets 413.
- **Concurrency.** At most `LOCAL_CHAT_MAX_CONCURRENT` generations at once.
  The next request gets 503 `busy` immediately rather than waiting in a
  queue. Ollama calls are async (`httpx.AsyncClient`), so a running
  generation never blocks the event loop. The slot is released in a
  `finally`, so timeouts and errors cannot leak it. Two messages to the
  same conversation at once give 409 for the second.
- **Timeout.** `LOCAL_CHAT_TIMEOUT_SECONDS` cancels the request to Ollama.
  The client gets 503 `timeout`.
- **Health.** `GET /local-llm/health` returns 200 with `ready` in the body,
  which is what the app reads. `?strict=true` returns 503 unless Ollama
  answers *and* has the model, so a monitor or `curl -f` cannot mistake
  "FastAPI is up" for "the model is up".

`/rag/chat` (Day 28) and the Gemini endpoints are not behind these guards.
Their behaviour is unchanged.

## Firewall

- **macOS:** System Settings → Network → Firewall → Options → allow incoming
  connections for the Python that runs uvicorn, and block the rest. The
  macOS application firewall works per app, not per port or address. A
  per-address rule needs `pf`.
- **Linux (VPS / home server):**
  ```bash
  sudo ufw default deny incoming
  sudo ufw allow from 192.168.10.0/24 to any port 8000 proto tcp
  sudo ufw deny 11434/tcp           # belt and braces: Ollama stays private
  ```

## Security limitations

There is no authentication. Anyone who can reach port 8000 can use the
model, up to the rate limit. That is acceptable on a trusted home network
and nowhere else. Before any public exposure, put the service behind HTTPS
(a reverse proxy such as Caddy or nginx), add authentication, allow only
known addresses at the firewall, and set `LOCAL_CHAT_TRUSTED_PROXIES` to the
proxy's address. Never publish Ollama's 11434.

## Tests and the probe

```bash
.venv-mac/bin/python -m pytest -q tests/test_day30_service.py              # mocked Ollama
DAY30_BASE_URL=http://192.168.10.35:8000 \
  .venv-mac/bin/python -m pytest -q -m integration tests/test_day30_network.py   # real service
.venv-mac/bin/python -m app.evaluate_day30 --base-url http://192.168.10.35:8000
.venv-mac/bin/python -m app.evaluate_day30 --base-url http://192.168.10.35:8000 --only unavailable  # with Ollama stopped
```

### Results so far

- `tests/test_day30_service.py`: 29 passed. Ollama here is a stand-in
  (httpx `MockTransport`, or an async fake that holds a generation open),
  so these tests prove the guards and the concurrency, not a real model.
- Full suite: 880 passed, 4 skipped. The skipped ones are integration tests
  that need a running model or service.
- The probe was also run in the development VM over real HTTP against
  `uvicorn` with a **stand-in Ollama** that answers after 1.5 s:
  - 413 for the oversized message;
  - 1 + 5 answers of about 1.6 s each;
  - 2 × 200 and 1 × 503 `busy` for 3 requests at once;
  - 429 with `Retry-After: 49` for the 11th request in the minute;
  - with nothing on the Ollama port, health `?strict=true` → 503 and chat →
    503 `not_running`.

  These numbers are about the service, not about qwen3:4b.
- Over the LAN, against the real model: *pending - fill in from
  `data/evaluation/day30_results.json` after the run on the Mac.*
