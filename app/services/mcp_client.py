"""The MCP client: connect to a server over stdio, list its tools, call one.

Day 16 proved the connection - start the server, initialize, list_tools().
Day 17 adds the step after it: call_tool(), so the agent can use what a
server offers. Every connection goes through the same four steps, and each
is logged as it happens:

    start the server (stdio)  ->  initialize the session
                              ->  list_tools()  or  call_tool()
                              ->  log what came back

Built on the official MCP Python SDK directly - ``stdio_client`` for the
transport and ``ClientSession`` for the protocol - rather than the SDK's
higher-level ``Client`` wrapper, so each of those steps stays visible.

Two servers ship with the project, both run as subprocesses:
``mcp_servers/japanese_learning.py`` (Day 16, dictionaries built in) and
``mcp_servers/jlpt_vocab.py`` (Day 17, the real JLPT vocabulary API).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp import types

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SERVER_SCRIPT = _PROJECT_ROOT / "mcp_servers" / "japanese_learning.py"

# How the client introduces itself to the server during initialize.
CLIENT_INFO = types.Implementation(name="japanese-helper-image-api", version="1.0.0")

# A server that starts but never answers must not hang the caller forever.
_READ_TIMEOUT_SECONDS = 15.0

# What the JLPT server needs from this process's environment: where the API
# is, and how to get out to the internet where a proxy is required. The SDK
# starts servers with a minimal environment of its own, so anything else -
# GEMINI_API_KEY above all - never reaches them.
_PASSED_THROUGH = (
    "JLPT_VOCAB_API_URL",
    "DIGEST_TASKS_FILE_PATH",
    "DIGEST_STORE_FILE_PATH",
    "PIPELINE_DIR_PATH",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "https_proxy",
    "http_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
)


class McpConnectionError(RuntimeError):
    """The server could not be started, never completed the handshake, or
    the connection broke during a request.

    One type for every way a connection can fail, so a caller does not need
    to know that the SDK reports them as exception groups from its task
    group; the original is kept as ``__cause__``. A tool that runs and
    reports a failure is not this - see McpToolCallResult.
    """


@dataclass(frozen=True)
class McpToolInfo:
    """One tool as the server described it."""

    name: str
    title: str
    description: str
    parameters: tuple[str, ...]
    # The JSON schema of the arguments, exactly as the server sent it - what
    # a model is shown so it can call the tool correctly.
    input_schema: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True)
class McpServerReport:
    """What one connection learned: who the server is and what it offers."""

    server_name: str
    server_version: str
    protocol_version: str
    tools: tuple[McpToolInfo, ...]


@dataclass(frozen=True)
class McpToolCallResult:
    """What one call_tool() came back with.

    ``ok`` is False when the tool ran and reported a failure (is_error in
    MCP) - the connection worked, the tool did not. ``data`` is the tool's
    structured output when it has one, otherwise its text.
    """

    name: str
    arguments: dict[str, Any]
    ok: bool
    data: dict[str, Any] | str | None = None
    error: str = ""


def local_server_parameters() -> StdioServerParameters:
    """The Day 16 test server: the same Python that runs this code, so the
    server sees the same installed SDK."""
    return StdioServerParameters(
        command=sys.executable,
        args=[str(LOCAL_SERVER_SCRIPT)],
        cwd=str(_PROJECT_ROOT),
    )


def module_server_parameters(module: str) -> StdioServerParameters:
    """One of this project's servers, run as a module from the project root
    so it can import the backend's own services, with the part of the
    environment it needs passed through."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", module],
        cwd=str(_PROJECT_ROOT),
        env={name: os.environ[name] for name in _PASSED_THROUGH if name in os.environ},
    )


def jlpt_vocab_server_parameters() -> StdioServerParameters:
    """The JLPT vocabulary server: single-word lookups and the periodic digest."""
    return module_server_parameters("mcp_servers.jlpt_vocab")


def japanese_data_server_parameters() -> StdioServerParameters:
    """Server #1 of the chain: search, over the JLPT vocabulary API."""
    return module_server_parameters("mcp_servers.japanese_data")


def processing_server_parameters() -> StdioServerParameters:
    """Server #2 of the chain: summarize, with no client of its own."""
    return module_server_parameters("mcp_servers.processing")


def storage_server_parameters() -> StdioServerParameters:
    """Server #3 of the chain: save_to_file, the only one that writes."""
    return module_server_parameters("mcp_servers.storage")


# Every server this project can start, by the name it answers to. The agent
# reaches them through app/services/mcp_registry.py; this map is also what
# the CLI below takes for --server.
SERVERS = {
    "japanese-learning": local_server_parameters,
    "jlpt-vocab": jlpt_vocab_server_parameters,
    "japanese-data": japanese_data_server_parameters,
    "processing": processing_server_parameters,
    "storage": storage_server_parameters,
}


@asynccontextmanager
async def _connected(parameters: StdioServerParameters) -> AsyncIterator[tuple[ClientSession, types.InitializeResult]]:
    """Start the server, run the handshake, and yield the session.

    Anything that goes wrong - before the handshake or during the request
    that follows - is logged and raised as McpConnectionError with the real
    reason.
    """
    logger.info("MCP: starting server over stdio: %s %s", parameters.command, " ".join(parameters.args))
    step = "connection"

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
                step = "request"

                yield session, initialized
    except Exception as error:
        reason = _root_cause(error)
        logger.error("MCP: %s failed: %s", step, reason)
        raise McpConnectionError(f"Could not use the MCP server: {reason}") from error


async def list_server_tools(server: StdioServerParameters | None = None) -> McpServerReport:
    """Connect to an MCP server over stdio, initialize, list its tools, log
    them, disconnect.

    Raises McpConnectionError, after logging it, when the server cannot be
    started or does not complete the handshake - a failed connection is
    reported, never mistaken for a server with no tools.
    """
    async with _connected(server or local_server_parameters()) as (session, initialized):
        tools = await _list_all_tools(session)

    report = McpServerReport(
        server_name=initialized.server_info.name,
        server_version=initialized.server_info.version or "",
        protocol_version=initialized.protocol_version,
        tools=tuple(_describe(tool) for tool in tools),
    )
    _log_tools(report)

    return report


async def call_server_tool(
    name: str,
    arguments: dict[str, Any],
    server: StdioServerParameters | None = None,
) -> McpToolCallResult:
    """Connect, initialize, call one tool, disconnect.

    A tool that runs and fails comes back as ``ok=False`` with its message;
    a server that cannot be reached raises McpConnectionError. The two are
    kept apart because they mean different things to the caller: the first
    is an answer about the data, the second is that there was no answer.
    """
    async with _connected(server or jlpt_vocab_server_parameters()) as (session, _):
        logger.info("MCP: call_tool %s(%s)", name, json.dumps(arguments, ensure_ascii=False))
        result = await session.call_tool(name, arguments)

    if result.is_error:
        message = _text_of(result) or "the tool reported an error"
        logger.warning("MCP: %s failed: %s", name, message)
        return McpToolCallResult(name=name, arguments=arguments, ok=False, error=message)

    data: dict[str, Any] | str | None = result.structured_content or _text_of(result)
    logger.info("MCP: %s returned %s", name, json.dumps(data, ensure_ascii=False)[:300])

    return McpToolCallResult(name=name, arguments=arguments, ok=True, data=data)


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


def _text_of(result: types.CallToolResult) -> str:
    return " ".join(block.text for block in result.content if isinstance(block, types.TextContent)).strip()


def _describe(tool: types.Tool) -> McpToolInfo:
    schema = tool.input_schema or {}

    return McpToolInfo(
        name=tool.name,
        title=tool.title or "",
        description=" ".join((tool.description or "").split()),
        parameters=tuple(schema.get("properties", {})),
        input_schema=schema,
    )


def _log_tools(report: McpServerReport) -> None:
    logger.info("MCP: list_tools() returned %d tool(s) from %r", len(report.tools), report.server_name)

    for tool in report.tools:
        logger.info("MCP:   - %s(%s): %s", tool.name, ", ".join(tool.parameters), tool.description)


def main() -> None:
    """From the project root:

        python -m app.services.mcp_client                      # Day 16 server, list its tools
        python -m app.services.mcp_client --server jlpt-vocab  # the JLPT server
        python -m app.services.mcp_client --server jlpt-vocab \\
            --call get_japanese_word_info --args '{"word": "学習"}'
    """
    parser = argparse.ArgumentParser(prog="python -m app.services.mcp_client")
    parser.add_argument("--server", choices=sorted(SERVERS), default="japanese-learning")
    parser.add_argument("--call", metavar="TOOL", help="call this tool instead of listing the tools")
    parser.add_argument("--args", default="{}", help="the tool's arguments, as a JSON object")
    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parameters = SERVERS[options.server]()

    if options.call:
        result = asyncio.run(call_server_tool(options.call, json.loads(options.args), parameters))
        print(json.dumps(result.data if result.ok else {"error": result.error}, ensure_ascii=False, indent=2))
    else:
        asyncio.run(list_server_tools(parameters))


if __name__ == "__main__":
    main()
