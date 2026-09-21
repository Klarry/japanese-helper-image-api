"""Day 16: a minimal MCP client.

The whole job is to prove the four steps of an MCP connection work end to
end, against a real server in a real subprocess:

    start the server (stdio)  ->  initialize the session  ->  list_tools()
                              ->  log what came back

Nothing here calls a tool, and nothing is wired into the agent yet: the
client connects, reads what the server offers, logs it and disconnects. The
result is also returned as plain data, so a test (or, later, the agent) can
check it without scraping logs.

It uses the official MCP Python SDK directly - ``stdio_client`` for the
transport and ``ClientSession`` for the protocol - rather than the SDK's
higher-level ``Client`` wrapper, because the point of the exercise is to see
each step happen: connection, initialize, list_tools.
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp import types

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SERVER_SCRIPT = _PROJECT_ROOT / "mcp_servers" / "japanese_learning.py"

# How the client introduces itself to the server during initialize.
CLIENT_INFO = types.Implementation(name="japanese-helper-image-api", version="1.0.0")

# A server that starts but never answers must not hang the caller forever.
_READ_TIMEOUT_SECONDS = 15.0


class McpConnectionError(RuntimeError):
    """The server could not be started, or never completed the handshake.

    One type for every way a connection can fail, so a caller does not need
    to know that the SDK reports them as exception groups from its task
    group; the original is kept as ``__cause__``.
    """


@dataclass(frozen=True)
class McpToolInfo:
    """One tool as the server described it."""

    name: str
    title: str
    description: str
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class McpServerReport:
    """What one connection learned: who the server is and what it offers."""

    server_name: str
    server_version: str
    protocol_version: str
    tools: tuple[McpToolInfo, ...]


def local_server_parameters() -> StdioServerParameters:
    """How to start the bundled Japanese-learning server: the same Python
    that runs this code, so the server sees the same installed SDK."""
    return StdioServerParameters(
        command=sys.executable,
        args=[str(LOCAL_SERVER_SCRIPT)],
        cwd=str(_PROJECT_ROOT),
    )


async def list_server_tools(server: StdioServerParameters | None = None) -> McpServerReport:
    """Connect to an MCP server over stdio, initialize, list its tools, log
    them, disconnect.

    Raises McpConnectionError, after logging it, when the server cannot be
    started or does not complete the handshake - a failed connection is
    reported, never mistaken for a server with no tools.
    """
    parameters = server or local_server_parameters()
    logger.info("MCP: starting server over stdio: %s %s", parameters.command, " ".join(parameters.args))

    try:
        async with stdio_client(parameters) as (read_stream, write_stream):
            logger.info("MCP: server process started, stdio pipes open")

            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=_READ_TIMEOUT_SECONDS,
                client_info=CLIENT_INFO,
            ) as session:
                initialized = await session.initialize()
                logger.info(
                    "MCP: session initialized - server %r version %s, protocol %s",
                    initialized.server_info.name,
                    initialized.server_info.version or "?",
                    initialized.protocol_version,
                )

                tools = await _list_all_tools(session)
    except Exception as error:
        reason = _root_cause(error)
        logger.error("MCP: connection failed: %s", reason)
        raise McpConnectionError(f"Could not connect to the MCP server: {reason}") from error

    report = McpServerReport(
        server_name=initialized.server_info.name,
        server_version=initialized.server_info.version or "",
        protocol_version=initialized.protocol_version,
        tools=tuple(_describe(tool) for tool in tools),
    )
    _log_tools(report)

    return report


async def _list_all_tools(session: ClientSession) -> list[types.Tool]:
    """list_tools(), following the cursor: a server may return its tools a
    page at a time, and a client that reads only the first page would report
    a partial list as the whole one."""
    tools: list[types.Tool] = []
    cursor: str | None = None

    while True:
        params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
        page = await session.list_tools(params=params)
        tools.extend(page.tools)
        cursor = page.next_cursor

        if not cursor:
            return tools


def _root_cause(error: BaseException) -> BaseException:
    """The first real error inside the exception groups the SDK's task
    groups wrap failures in ("Connection closed", not "unhandled errors in
    a TaskGroup"). Looked up by attribute rather than by type: on Python
    3.10 exception groups come from a backport, not from the builtins."""
    inner = getattr(error, "exceptions", None)

    while inner:
        error = inner[0]
        inner = getattr(error, "exceptions", None)

    return error


def _describe(tool: types.Tool) -> McpToolInfo:
    properties = (tool.input_schema or {}).get("properties", {})

    return McpToolInfo(
        name=tool.name,
        title=tool.title or "",
        description=" ".join((tool.description or "").split()),
        parameters=tuple(properties),
    )


def _log_tools(report: McpServerReport) -> None:
    logger.info("MCP: list_tools() returned %d tool(s) from %r", len(report.tools), report.server_name)

    for tool in report.tools:
        logger.info("MCP:   - %s(%s): %s", tool.name, ", ".join(tool.parameters), tool.description)


def main() -> None:
    """``python -m app.services.mcp_client`` - connect to the local server
    and print the log to the terminal."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(list_server_tools())


if __name__ == "__main__":
    main()
