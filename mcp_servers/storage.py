"""MCP server #3: storage.

Stage three of the chain, and the only server that writes anything. It takes
the summary and the findings it was made from and puts both in one
timestamped JSON file, through the same atomic write the rest of the project
uses, then reports where the file went.

Run it from the project root as a module:

    python -m mcp_servers.storage
"""

from typing import Annotated

from pydantic import Field

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.services.pipeline_tools import save_to_file as save_result
from mcp_servers.pipeline_models import PipelineSummary, SavedResult, SearchFindings

SERVER_NAME = "storage"
SERVER_VERSION = "1.0.0"

server = MCPServer(
    SERVER_NAME,
    version=SERVER_VERSION,
    instructions="Save a processed result to a timestamped JSON file.",
)


@server.tool(title="Save to file")
def save_to_file(
    summary: Annotated[
        PipelineSummary,
        Field(description="Exactly what the summarize tool returned."),
    ],
    findings: Annotated[
        SearchFindings,
        Field(description="The search findings the summary was made from, so the file keeps both."),
    ],
) -> SavedResult:
    """Stage 3 of the chain: write the summary, together with the findings it was made from, to a
    timestamped JSON file, and report the file name and whether it was saved. Nothing is
    recomputed here - what is saved is what the previous stages returned."""
    try:
        return SavedResult(**save_result(summary.model_dump(), findings.model_dump()))
    except OSError as error:
        raise ToolError(f"could not save the result: {error}") from error


if __name__ == "__main__":
    server.run("stdio")
