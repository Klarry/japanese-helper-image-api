"""MCP server #1: Japanese data.

Stage one of the chain, and the only server of the three that talks to the
outside world: it asks the JLPT vocabulary API the Android app already uses,
through the backend's own client for it, and hands back structured findings.

Run it from the project root as a module, so ``app`` is importable:

    python -m mcp_servers.japanese_data

It then waits for a client on stdin. Nothing may be printed to stdout - in
the stdio transport stdout is the protocol.
"""

from typing import Annotated

from pydantic import Field

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.services.jlpt_vocab_api import JlptVocabApiError
from app.services.pipeline_tools import search as run_search
from mcp_servers.pipeline_models import SearchFindings

SERVER_NAME = "japanese-data"
SERVER_VERSION = "1.0.0"

server = MCPServer(
    SERVER_NAME,
    version=SERVER_VERSION,
    instructions="Look Japanese words up in the JLPT vocabulary list (N5 to N1).",
)


@server.tool(title="Search")
async def search(
    query: Annotated[
        str,
        Field(
            description=(
                "A Japanese word or kanji to look up, written in Japanese exactly as it "
                "appears - for example 学習 or 勉強. Not romaji, not a translation."
            ),
            min_length=1,
            max_length=40,
        ),
    ],
) -> SearchFindings:
    """Stage 1 of the search -> summarize -> save_to_file chain: look the query up in the JLPT
    vocabulary API the app uses and return the findings as structured data. Use this when the
    learner wants the result summarised or saved; a plain question about one word is answered
    with get_japanese_word_info instead."""
    try:
        return SearchFindings(**await run_search(query))
    except JlptVocabApiError as error:
        # ToolError: an anticipated failure, so its text reaches the caller as
        # is_error=True instead of being withheld like a crash.
        raise ToolError(str(error)) from error


if __name__ == "__main__":
    server.run("stdio")
