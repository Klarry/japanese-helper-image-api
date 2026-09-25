"""Day 17: the agent's toolbox - the MCP servers, reached through the MCP client.

The agent does not know how any tool works, and since Day 20 it does not
know where one lives either. It asks the registry what exists (discovery:
every registered server is asked with list_tools, and what it reports is
what the agent may use), and asks it to run one - the registry sends the
call to the server that offers it.

What comes back is turned into a TOOL RESULTS block for the prompt: the call,
and either its data or its failure. A failure is said out loud - the model is
told a lookup failed rather than left to fill the gap from memory as if it
had succeeded.
"""

import json
import logging
from collections.abc import Sequence

from app.services.mcp_client import McpToolCallResult
from app.services.mcp_registry import REGISTERED_SERVERS, McpServerRegistry, McpServerSpec

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


class McpToolbox(McpServerRegistry):
    """The registered MCP servers, as the agent sees them.

    A name of its own because that is what the agent calls it, and a
    subclass rather than a wrapper because there is nothing to add: what the
    agent needs from the servers - what exists, and running one thing - is
    exactly what the registry does.
    """

    def __init__(self, servers: Sequence[McpServerSpec] = REGISTERED_SERVERS) -> None:
        super().__init__(servers)


def call_line(result: McpToolCallResult) -> str:
    """One call and its outcome, the way every tool block writes it."""
    call = f"{result.name}({json.dumps(result.arguments, ensure_ascii=False)})"
    outcome = (
        json.dumps(result.data, ensure_ascii=False)
        if result.ok
        else f"FAILED: {result.error}"
    )

    return f"- {call} -> {outcome}"


def results_section(results: Sequence[McpToolCallResult]) -> str:
    """The calls and what they returned, as the model reads them."""
    if not results:
        return ""

    return f"{_RESULTS_CAPTION}\n" + "\n".join(call_line(result) for result in results)


def unavailable_section() -> str:
    return _UNAVAILABLE_SECTION
