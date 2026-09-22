"""Day 17: the agent's toolbox - one MCP server, reached through the MCP client.

The agent does not know how any tool works. It knows a server exists, asks
it what it offers (list_tools, kept once it has succeeded - a server's tools
do not change between two messages), and asks it to run one (call_tool, a
fresh connection each time, so a server that crashed on one call cannot take
the next one down with it).

What comes back is turned into a TOOL RESULTS block for the prompt: the call,
and either its data or its failure. A failure is said out loud - the model is
told a lookup failed rather than left to fill the gap from memory as if it
had succeeded.
"""

import json
import logging
from collections.abc import Callable, Sequence

from mcp import StdioServerParameters

from app.services.mcp_client import (
    McpConnectionError,
    McpToolCallResult,
    McpToolInfo,
    call_server_tool,
    jlpt_vocab_server_parameters,
    list_server_tools,
)

logger = logging.getLogger(__name__)

_RESULTS_CAPTION = (
    "TOOL RESULTS (looked up just now for this request, through MCP, in the JLPT "
    "vocabulary dictionary the app uses). This is real dictionary data: base the "
    "readings, meanings and JLPT levels in your answer on it rather than on memory, and "
    "say that they come from the dictionary. If a lookup found nothing or failed, say so "
    "plainly instead of presenting a guess as dictionary data."
)

_UNAVAILABLE_SECTION = (
    "TOOL RESULTS: the dictionary tools could not be reached for this request. If the "
    "answer needs dictionary data (a reading, a meaning, a JLPT level), say that it "
    "could not be checked in the dictionary right now."
)


class McpToolbox:
    """One MCP server, as the agent sees it."""

    def __init__(self, server: Callable[[], StdioServerParameters] = jlpt_vocab_server_parameters) -> None:
        # A factory rather than fixed parameters: the environment the server
        # is started with is read at connection time.
        self._server = server
        self._tools: tuple[McpToolInfo, ...] | None = None

    async def tools(self) -> tuple[McpToolInfo, ...]:
        """What the server offers. Raises McpConnectionError when it cannot be
        reached; a failure is not remembered, so the next message tries again."""
        if self._tools is None:
            report = await list_server_tools(self._server())
            self._tools = report.tools

        return self._tools

    async def call(self, tool: str, arguments: dict) -> McpToolCallResult:
        """Run one tool. Never raises: a server that cannot be reached comes
        back as a failed result, the same shape as a tool that failed."""
        try:
            return await call_server_tool(tool, arguments, self._server())
        except McpConnectionError as error:
            return McpToolCallResult(name=tool, arguments=arguments, ok=False, error=str(error))


def results_section(results: Sequence[McpToolCallResult]) -> str:
    """The calls and what they returned, as the model reads them."""
    if not results:
        return ""

    lines = []

    for result in results:
        call = f"{result.name}({json.dumps(result.arguments, ensure_ascii=False)})"
        outcome = (
            json.dumps(result.data, ensure_ascii=False)
            if result.ok
            else f"FAILED: {result.error}"
        )
        lines.append(f"- {call} -> {outcome}")

    return f"{_RESULTS_CAPTION}\n" + "\n".join(lines)


def unavailable_section() -> str:
    return _UNAVAILABLE_SECTION
