"""Day 30: probe the running service over the network and write down what happened.

    python -m app.evaluate_day30 --base-url http://192.168.10.35:8000
    python -m app.evaluate_day30 --base-url http://192.168.10.35:8000 --only unavailable

Real HTTP to a real service - nothing is mocked here. Phases, in an order
chosen so each limit shows up where it should:

    health       GET /local-llm/health?strict=true
    oversized    one message over the character limit         -> 413
    one          one chat request
    sequential   five chat requests, one after another
    parallel     max_concurrent + 1 at the same moment         -> the extra one 503 busy
    rate_limit   requests until one is refused with 429         (the quota is spent by now)
    unavailable  (only with --only unavailable, Ollama stopped) -> 503, no other model

Every request records its HTTP status, the reason the service gave, the time
the client waited, and the generation time the service measured. Results go
to data/evaluation/day30_results.json. Nothing is estimated.
"""

import argparse
import asyncio
import json
import platform
import socket
import statistics
import time
from pathlib import Path

import httpx

RESULTS_FILE = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "day30_results.json"
RULE = "=" * 78
QUESTIONS = [
    "What is MCP? Answer in one sentence.",
    "Name one benefit of running a language model locally. One sentence.",
    "What does an HTTP 429 status mean? One sentence.",
    "What is a context window in a language model? One sentence.",
    "Translate 'study' into Japanese. One word.",
    "What is rate limiting? One sentence.",
]


async def call(http: httpx.AsyncClient, phase: str, index: int, method: str, path: str, **kwargs) -> dict:
    started = time.perf_counter()
    record = {"phase": phase, "index": index, "method": method, "path": path}

    try:
        response = await http.request(method, path, **kwargs)
    except httpx.TimeoutException:
        record.update(status=None, reason="client_timeout", ms=int((time.perf_counter() - started) * 1000))
        return record
    except httpx.RequestError as error:
        record.update(status=None, reason=f"unreachable: {type(error).__name__}", ms=int((time.perf_counter() - started) * 1000))
        return record

    record["ms"] = int((time.perf_counter() - started) * 1000)
    record["status"] = response.status_code

    try:
        body = response.json()
    except ValueError:
        body = {}

    if response.status_code == 200:
        record["reason"] = "ok"
    elif path.startswith("/local-llm/health"):
        record["reason"] = "not_ready"
    else:
        record["reason"] = body.get("reason", "") or f"http_{response.status_code}"
    record["request_id"] = body.get("request_id") or response.headers.get("x-request-id", "")
    record["model"] = body.get("model", "")
    record["provider"] = body.get("provider", "")
    record["generation_ms"] = (body.get("timing") or {}).get("generation_ms")
    record["retry_after"] = response.headers.get("retry-after")
    record["answer_chars"] = len(body.get("response") or "")

    if path.startswith("/local-llm/health"):
        record["health"] = body

    return record


def line(record: dict) -> str:
    status = record["status"] if record["status"] is not None else "---"
    extra = []

    if record.get("generation_ms") is not None:
        extra.append(f"generation {record['generation_ms']}ms")

    if record.get("retry_after"):
        extra.append(f"Retry-After {record['retry_after']}s")

    if record.get("request_id"):
        extra.append(f"id {record['request_id']}")

    return (
        f"  {record['phase']:<11} #{record['index']:<2} {status:>4}  {record['reason']:<18} "
        f"{record['ms']:>7} ms  " + " · ".join(extra)
    )


async def run(options) -> int:
    timeout = httpx.Timeout(options.timeout, connect=10)
    records: list[dict] = []

    def keep(record: dict) -> dict:
        records.append(record)
        print(line(record))
        return record

    print(f"\n{RULE}\n=== Day 30 · local LLM service probe ===\n{RULE}")
    print(f"target {options.base_url} · client {socket.gethostname()} · phases {', '.join(options.only or ['all'])}\n")

    async with httpx.AsyncClient(base_url=options.base_url, timeout=timeout) as http:
        health = keep(await call(http, "health", 1, "GET", "/local-llm/health", params={"strict": "true"}))
        service = (health.get("health") or {}).get("service") or {}
        max_chars = int(service.get("max_message_chars", 4000))
        parallel = int(service.get("max_concurrent", 2)) + 1

        def chat(phase: str, index: int, message: str):
            return call(http, phase, index, "POST", "/local-chat", json={"message": message})

        if options.only == ["unavailable"]:
            keep(await chat("unavailable", 1, QUESTIONS[0]))
        elif health["status"] != 200:
            # Nothing else would measure anything but "Ollama is down" - and
            # every request would still spend the rate-limit quota.
            reason = ((health.get("health") or {}).get("error") or "the model is not ready").splitlines()[0]
            print(f"\nStopped: the service is up but the model is not ready ({reason}).")
            print("Start Ollama (open the app, or `ollama serve`), check `ollama list`, and run this again.")
            print("To record the 'Ollama is down' case on purpose, use --only unavailable.")
            return 1
        else:
            keep(await chat("oversized", 1, "a" * (max_chars + 1)))
            keep(await chat("one", 1, QUESTIONS[0]))

            for index in range(1, 6):
                keep(await chat("sequential", index, QUESTIONS[index % len(QUESTIONS)]))

            for record in await asyncio.gather(
                *(chat("parallel", index, QUESTIONS[index % len(QUESTIONS)]) for index in range(1, parallel + 1))
            ):
                keep(record)

            for index in range(1, options.rate_attempts + 1):
                record = keep(await chat("rate_limit", index, QUESTIONS[0]))

                if record["status"] == 429:
                    break

    summary = summarise(records)
    report(summary)
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(
        json.dumps(
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "base_url": options.base_url,
                "client": {"hostname": socket.gethostname(), "system": platform.system()},
                "service": service,
                "health": health.get("health"),
                "summary": summary,
                "requests": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWritten: {options.out}")

    return 0


def summarise(records: list[dict]) -> dict:
    phases: dict[str, dict] = {}

    for record in records:
        phase = phases.setdefault(record["phase"], {"requests": 0, "statuses": {}, "ms": []})
        phase["requests"] += 1
        key = str(record["status"]) if record["status"] is not None else record["reason"]
        phase["statuses"][key] = phase["statuses"].get(key, 0) + 1
        phase["ms"].append(record["ms"])

    for phase in phases.values():
        times = phase.pop("ms")
        phase["median_ms"] = int(statistics.median(times))
        phase["min_ms"] = min(times)
        phase["max_ms"] = max(times)

    chats = [record for record in records if record["path"] == "/local-chat"]

    return {
        "phases": phases,
        "chat_requests": len(chats),
        "successful": sum(1 for record in chats if record["status"] == 200),
        "failed": sum(1 for record in chats if record["status"] != 200),
        "timeouts": sum(1 for record in chats if record["reason"] in {"timeout", "client_timeout"}),
        "rate_limited": sum(1 for record in chats if record["status"] == 429),
        "busy": sum(1 for record in chats if record["reason"] == "busy"),
        "too_large": sum(1 for record in chats if record["status"] == 413),
        "unavailable": sum(1 for record in chats if record["reason"] in {"not_running", "model_not_installed"}),
        "providers_seen": sorted({record["provider"] for record in chats if record.get("provider")}),
        "median_success_ms": int(statistics.median([r["ms"] for r in chats if r["status"] == 200]))
        if any(r["status"] == 200 for r in chats)
        else None,
    }


def report(summary: dict) -> None:
    print(f"\n{RULE}\n=== Summary ===\n{RULE}")
    print(f"{'phase':<12}{'requests':>9}  {'statuses':<28}{'median':>9}{'min':>9}{'max':>9}")

    for name, phase in summary["phases"].items():
        statuses = ", ".join(f"{key}×{count}" for key, count in phase["statuses"].items())
        print(f"{name:<12}{phase['requests']:>9}  {statuses:<28}{phase['median_ms']:>7}ms{phase['min_ms']:>7}ms{phase['max_ms']:>7}ms")

    print()
    for key in ("chat_requests", "successful", "failed", "timeouts", "rate_limited", "busy", "too_large", "unavailable", "providers_seen", "median_success_ms"):
        print(f"  {key:<18} {summary[key]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m app.evaluate_day30", description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=300, help="seconds the client waits for one answer")
    parser.add_argument("--rate-attempts", type=int, default=3, help="requests sent in the rate_limit phase at most")
    parser.add_argument("--only", nargs="*", default=[], choices=["unavailable"], help="run only the given phase")
    parser.add_argument("--out", type=Path, default=RESULTS_FILE)
    options = parser.parse_args(argv)
    options.base_url = options.base_url.rstrip("/")

    return options


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
