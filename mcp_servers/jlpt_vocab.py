"""An MCP server around the JLPT vocabulary API the app already uses.

Day 16's server had its dictionaries built in - it existed to be connected
to. This one is the first with a tool that does real work: it asks the same
JLPT vocabulary API the Android app reads its words from, through the
backend's own client for it (app.services.jlpt_vocab_api), and hands the
answer back as structured data.

Run it from the project root as a module, so ``app`` is importable:

    python -m mcp_servers.jlpt_vocab

It then waits for a client on stdin. Nothing may be printed to stdout - in
the stdio transport stdout is the protocol.
"""

from typing import Annotated

from pydantic import BaseModel, Field

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.services.jlpt_vocab_api import JlptVocabApiError, search_words

SERVER_NAME = "jlpt-vocab"
SERVER_VERSION = "1.0.0"
SOURCE = "jlpt-vocab-api.vercel.app"

server = MCPServer(
    SERVER_NAME,
    version=SERVER_VERSION,
    instructions="Look up Japanese words and kanji in the JLPT vocabulary list (N5 to N1).",
)


class WordMatch(BaseModel):
    word: str = Field(description="The word as written in Japanese")
    reading: str | None = Field(description="Reading in kana (furigana)")
    romaji: str | None = Field(description="Reading in Latin letters")
    meaning: str | None = Field(description="English meaning")
    jlpt_level: str | None = Field(description="JLPT level, N5 (easiest) to N1")


class JapaneseWordInfo(BaseModel):
    query: str = Field(description="What was looked up")
    found: bool = Field(description="Whether the JLPT list has an entry spelled exactly like this")
    matches: list[WordMatch] = Field(description="Every entry spelled exactly like the query")
    source: str = Field(default=SOURCE, description="Where the data came from")


def _jlpt(level: int | None) -> str | None:
    return f"N{level}" if level in (1, 2, 3, 4, 5) else None


@server.tool(title="Get Japanese word info")
async def get_japanese_word_info(
    word: Annotated[
        str,
        Field(
            description=(
                "A Japanese word or a single kanji, written in Japanese (kanji and/or kana) "
                "exactly as it appears - for example 学習, 学 or ありがとう. Not romaji and "
                "not a translation."
            ),
            min_length=1,
            max_length=40,
        ),
    ],
) -> JapaneseWordInfo:
    """Look up a Japanese word or kanji in the JLPT vocabulary list (N5 to N1) - the same
    dictionary API the Japanese Helper app uses. Returns the kana reading, romaji, English
    meaning and JLPT level of every entry written exactly like the query. found=false means
    the word is not in the JLPT list, which is an answer, not an error."""
    query = word.strip()

    try:
        entries = await search_words(query)
    except JlptVocabApiError as error:
        # ToolError: an anticipated failure, so its text reaches the caller as
        # is_error=True instead of being withheld like a crash.
        raise ToolError(str(error)) from error

    matches = [
        WordMatch(
            word=entry.word,
            reading=entry.furigana,
            romaji=entry.romaji,
            meaning=entry.meaning,
            jlpt_level=_jlpt(entry.level),
        )
        for entry in entries
    ]

    return JapaneseWordInfo(query=query, found=bool(matches), matches=matches)


if __name__ == "__main__":
    server.run("stdio")
