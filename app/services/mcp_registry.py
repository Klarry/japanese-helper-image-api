"""Day 20: which MCP servers exist, what each one offers, and where a call goes.

Until now the agent knew one server. Now there are four, each with its own
process and its own job, and a tool name alone no longer says where to send
a call. This module is the answer to that: the list of servers, the
discovery that asks each one what it has, and the routing table built from
their answers.

Discovery, not configuration. The tools are not written down here - each
server is asked, over MCP, with the same ``list_tools()`` Day 16 used, and
what it reports is what the agent is told it can use. A tool that a server
stops offering disappears from the agent's list by itself; a server that
cannot be reached takes only its own tools with it, and the rest keep
working.

Two failures, told apart on purpose: a tool nobody offers (the plan named
something that does not exist) and a server that cannot be reached (it
exists but is down). Both come back as a failed call rather than an
exception, because a chain that stops has to say which of the two happened.
"""

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from mcp import StdioServerParameters

from app.services.mcp_client import (
    McpConnectionError,
    McpToolCallResult,
    McpToolInfo,
    call_server_tool,
    japanese_data_server_parameters,
    jlpt_vocab_server_parameters,
    processing_server_parameters,
    storage_server_parameters,
)

logger = logging.getLogger(__name__)

# How long one tool call may take before the orchestrator gives up on it.
# The MCP client has read timeouts of its own; this is the outer bound on a
# server that accepted the call and then went quiet.
CALL_TIMEOUT_SECONDS = 45.0


@dataclass(frozen=True)
class McpServerSpec:
    """One registered server: the name it answers to, what it is for, and
    how to start it."""

    name: str
    role: str
    parameters: Callable[[], StdioServerParameters]


# The registered servers, in the order the chain uses them. japanese-data,
# processing and storage are one stage each; jlpt-vocab is the Day 17-18
# server, still here because its tools are still the right answer to a
# question about a single word.
REGISTERED_SERVERS: tuple[McpServerSpec, ...] = (
    McpServerSpec(
        "japanese-data",
        "looks words up in the JLPT vocabulary API the app uses",
        japanese_data_server_parameters,
    ),
    McpServerSpec(
        "processing",
        "turns findings into a summary, without looking anything up",
        processing_server_parameters,
    ),
    McpServerSpec(
        "storage",
        "writes a result to a timestamped JSON file",
        storage_server_parameters,
    ),
    McpServerSpec(
        "jlpt-vocab",
        "single-word lookups and the periodic digest",
        jlpt_vocab_server_parameters,
    ),
)


class ToolNotFound(LookupError):
    """No registered server offers a tool by that name."""


class McpServerRegistry:
    """The servers the agent can reach, and what each of them offers."""

    def __init__(self, servers: Sequence[McpServerSpec] = REGISTERED_SERVERS) -> None:
        self._servers = tuple(servers)
        # server name -> its tools, filled by discovery. A server that failed
        # to answer is simply absent, so the next discovery asks it again.
        self._discovered: dict[str, tuple[McpToolInfo, ...]] = {}

    @property
    def servers(self) -> tuple[McpServerSpec, ...]:
        return self._servers

    def spec(self, name: str) -> McpServerSpec | None:
        return next((server for server in self._servers if server.name == name), None)

    async def discover(self) -> dict[str, tuple[McpToolInfo, ...]]:
        """Ask every server that has not answered yet what it offers.

        Never raises: a server that cannot be reached is logged and left out
        of the result, so the tools of the others are still usable.
        """
        for server in self._servers:
            if server.name in self._discovered:
                continue

            try:
                report = await call_with_timeout(_list(server))
            except (McpConnectionError, TimeoutError) as error:
                logger.warning("[Orchestrator] Server %s could not be listed: %s", server.name, error)
                continue

            self._discovered[server.name] = report
            logger.info(
                "[Orchestrator] Server %s offers %s",
                server.name,
                ", ".join(tool.name for tool in report) or "nothing",
            )

        return dict(self._discovered)

    async def tools(self) -> tuple[McpToolInfo, ...]:
        """Every tool on every server that answered, in registration order."""
        discovered = await self.discover()

        return tuple(tool for server in self._servers for tool in discovered.get(server.name, ()))

    async def server_of(self, tool: str) -> str:
        """Which server offers ``tool``. Raises ToolNotFound when none does."""
        discovered = await self.discover()

        for server in self._servers:
            if any(offered.name == tool for offered in discovered.get(server.name, ())):
                return server.name

        raise ToolNotFound(tool)

    async def routing_table(self) -> dict[str, str]:
        """tool name -> server name, for everything discovered so far."""
        discovered = await self.discover()

        return {
            tool.name: server.name
            for server in self._servers
            for tool in discovered.get(server.name, ())
        }

    async def call(self, tool: str, arguments: dict) -> McpToolCallResult:
        """Send one call to the server that offers it.

        Never raises. A tool nobody offers, a server that cannot be reached
        and a server that went quiet all come back as a failed result, with
        the reason in ``error``.
        """
        try:
            name = await self.server_of(tool)
        except ToolNotFound:
            logger.warning("[Orchestrator] Tool not found on any server: %s", tool)
            return McpToolCallResult(
                name=tool,
                arguments=arguments,
                ok=False,
                error=f"tool '{tool}' is not offered by any registered MCP server",
            )

        spec = self.spec(name)
        assert spec is not None  # noqa: S101 - server_of only returns registered names

        try:
            return await call_with_timeout(call_server_tool(tool, arguments, spec.parameters()))
        except McpConnectionError as error:
            # The server was reachable at discovery and is not now: forget it,
            # so the next request tries to list it again.
            self._discovered.pop(name, None)
            return McpToolCallResult(
                name=tool,
                arguments=arguments,
                ok=False,
                error=f"MCP server '{name}' is unavailable: {error}",
            )
        except TimeoutError:
            self._discovered.pop(name, None)
            return McpToolCallResult(
                name=tool,
                arguments=arguments,
                ok=False,
                error=f"MCP server '{name}' did not answer within {CALL_TIMEOUT_SECONDS:.0f}s",
            )


async def _list(server: McpServerSpec) -> tuple[McpToolInfo, ...]:
    from app.services.mcp_client import list_server_tools

    report = await list_server_tools(server.parameters())

    return report.tools


async def call_with_timeout(awaitable, seconds: float = CALL_TIMEOUT_SECONDS):
    """asyncio.wait_for, with the 3.10-friendly spelling of its error."""
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except asyncio.TimeoutError as error:  # noqa: UP041 - 3.10 has no TimeoutError alias
        raise TimeoutError(str(error) or "timed out") from error
