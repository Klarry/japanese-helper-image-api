"""MCP server #2: processing.

Stage two of the chain. It takes what the japanese-data server returned and
turns it into a short summary - and it is the only one of the three servers
with no client of any kind in it: no HTTP, no disk. Whatever it says about a
word therefore comes from the findings it was handed, which is what makes
the chain's data flow checkable rather than a matter of trust.

Run it from the project root as a module:

    python -m mcp_servers.processing
"""

from typing import Annotated

from pydantic import Field

from mcp.server import MCPServer

from app.services.pipeline_tools import summarize as summarize_findings
from mcp_servers.pipeline_models import PipelineSummary, SearchFindings

SERVER_NAME = "processing"
SERVER_VERSION = "1.0.0"

server = MCPServer(
    SERVER_NAME,
    version=SERVER_VERSION,
    instructions="Turn search findings into a short summary. Looks nothing up.",
)


@server.tool(title="Summarize")
def summarize(
    findings: Annotated[
        SearchFindings,
        Field(description="Exactly what the search tool returned, passed through unchanged."),
    ],
) -> PipelineSummary:
    """Stage 2 of the chain: turn what search returned into a short summary. This tool does not
    search - it only reads the findings it is given, so it can never disagree with them. Give it
    the result of a search that has already run."""
    return PipelineSummary(**summarize_findings(findings.model_dump()))


if __name__ == "__main__":
    server.run("stdio")
