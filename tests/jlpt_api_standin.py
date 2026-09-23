"""A local stand-in for jlpt-vocab-api.vercel.app - for tests only.

The tests exercise the real path: the agent, the MCP client, the MCP server
in a real subprocess, and the backend's real HTTP client for the API. The
only thing replaced is the far end of that HTTP request, so the suite does
not depend on the internet. It serves ``GET /api/words?word=...`` and ``GET /api/words/random`` in the
live API's exact response shapes; the entries below are copied verbatim
from the live API. The random endpoint here is not random - it walks the
pool in order, so a test can count what came back.
"""

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socket import socket
from urllib.parse import parse_qs, urlparse

LIVE_ENTRIES = {
    "学習": [{"word": "学習", "meaning": "study, learning", "furigana": "がくしゅう", "romaji": "gakushū", "level": 3}],
    "学": [
        {
            "word": "学",
            "meaning": "learning, scholarship, erudition, knowledge",
            "furigana": "がく",
            "romaji": "gaku",
            "level": 3,
        }
    ],
    "勉強": [
        {"word": "勉強", "meaning": "to study", "furigana": "べんきょうする", "romaji": "benkyōsuru", "level": 5},
        {
            "word": "勉強",
            "meaning": "study, diligence, discount, reduction",
            "furigana": "べんきょう",
            "romaji": "benkyō",
            "level": 3,
        },
    ],
    "申し訳": [
        {"word": "申し訳", "meaning": "apology, excuse", "furigana": "もうしわけ", "romaji": "mōshiwake", "level": 3}
    ],
}

POOL = [entry for entries in LIVE_ENTRIES.values() for entry in entries]


@contextmanager
def jlpt_api(status: int = 200, raw_body: str | None = None) -> Iterator[tuple[str, list[dict]]]:
    """Serve the API on a free local port. Yields its base URL and the list
    of requests it received (path and query), so a test can see exactly
    what reached "the API". ``status``/``raw_body`` make it misbehave."""
    received: list[dict] = []
    served: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server's naming
            url = urlparse(self.path)
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            received.append({"path": url.path, "query": query})

            if raw_body is not None:
                body = raw_body
            elif url.path == "/api/words":
                words = LIVE_ENTRIES.get(query.get("word", ""), [])
                body = json.dumps({"total": len(words), "offset": 0, "limit": 10, "words": words})
            elif url.path == "/api/words/random":
                level = query.get("level")
                pool = [word for word in POOL if not level or str(word["level"]) == level] or POOL
                body = json.dumps(pool[len(served) % len(pool)])
                served.append(body)
            else:
                body = json.dumps({"error": "not found"})

            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:  # keep test output clean
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield f"http://127.0.0.1:{server.server_port}/", received
    finally:
        server.shutdown()
        server.server_close()


def closed_port_url() -> str:
    """A local URL nothing is listening on - "the API is down"."""
    with socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    return f"http://127.0.0.1:{port}/"
