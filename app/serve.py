"""Day 30: start the backend so other machines on the network can reach it.

    python -m app.serve                      # 0.0.0.0:8000, one worker
    SERVICE_HOST=127.0.0.1 python -m app.serve
    SERVICE_PORT=8080 python -m app.serve

The same app as ``uvicorn app.main:app --host 0.0.0.0 --port 8000`` - this
only adds the defaults a network service needs and prints where it can be
reached. One worker on purpose: the rate limit and the concurrency cap of
``/local-chat`` live in the process, and two workers would each keep their
own count.

Ollama itself is not touched: it stays on the loopback address it binds to
by default, and only this backend talks to it.
"""

import os
import socket
from urllib.parse import urlparse

import uvicorn

from app.services import ollama_service


def lan_address() -> str | None:
    """The address this machine uses on its network.

    A UDP socket "connected" to a documentation address (RFC 5737) - no
    packet is sent; the system only picks the interface it would use.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        probe.connect(("192.0.2.1", 80))
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()


def main() -> None:
    host = os.getenv("SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("SERVICE_PORT", "8000"))
    ollama_host = urlparse(ollama_service.BASE_URL).hostname or ""
    lan = lan_address()

    print("Day 30 · local LLM service")
    print(f"  listening on   http://{host}:{port}")

    if host in {"0.0.0.0", "::"} and lan:
        print(f"  on the network http://{lan}:{port}  (health: /local-llm/health?strict=true)")
    elif host in {"127.0.0.1", "localhost"}:
        print("  loopback only - other machines cannot reach this; set SERVICE_HOST=0.0.0.0")

    if ollama_host in {"localhost", "127.0.0.1", "::1"}:
        print(f"  Ollama         {ollama_service.BASE_URL}  (loopback - not exposed to the network)")
    else:
        print(f"  Ollama         {ollama_service.BASE_URL}  (NOT loopback - make sure 11434 is firewalled)")

    print("  No authentication: keep this on a trusted network, behind the host firewall.\n")

    uvicorn.run("app.main:app", host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
