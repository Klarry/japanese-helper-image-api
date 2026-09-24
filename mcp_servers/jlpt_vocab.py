"""An MCP server around the JLPT vocabulary API the app already uses.

Day 16's server had its dictionaries built in - it existed to be connected
to. This one does real work: it asks the same JLPT vocabulary API the Android
app reads its words from, through the backend's own client for it
(app.services.jlpt_vocab_api), and hands the answer back as structured data.

Six tools, three kinds. ``get_japanese_word_info`` answers now (Day 17).
``create_periodic_digest`` and ``get_latest_digest`` (Day 18) work on a task
that outlives the call: creating one writes it to disk, the backend's
scheduler runs it on its own from then on, and reading the digest is reading
what those runs have piled up. This process does not run anything itself -
it is a subprocess that ends when the call does, which is exactly why the
runs belong to the backend.

``search``, ``summarize`` and ``save_to_file`` (Day 19) are one chain: each
takes what the one before it returned. They are three separate tools rather
than one because the agent should be able to stop after any of them, and
because a step that fails must stop the chain where it failed - which is
only visible if the steps are visible.

Run it from the project root as a module, so ``app`` is importable:

    python -m mcp_servers.jlpt_vocab

It then waits for a client on stdin. Nothing may be printed to stdout - in
the stdio transport stdout is the protocol.
"""

from typing import Annotated

from pydantic import BaseModel, Field

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.services.digest import (
    MAX_INTERVAL_SECONDS,
    MIN_INTERVAL_SECONDS,
    WORDS_PER_RUN,
    DigestStore,
    DigestTaskStorage,
    build_digest,
)
from app.services.jlpt_vocab_api import JlptVocabApiError, search_words
from app.services.pipeline_tools import save_to_file as save_result
from app.services.pipeline_tools import search as run_search
from app.services.pipeline_tools import summarize as summarize_findings

SERVER_NAME = "jlpt-vocab"
SERVER_VERSION = "1.0.0"
SOURCE = "jlpt-vocab-api.vercel.app"

server = MCPServer(
    SERVER_NAME,
    version=SERVER_VERSION,
    instructions=(
        "Look up Japanese words and kanji in the JLPT vocabulary list (N5 to N1), keep "
        "a periodic digest of words collected from it, and run the search -> summarize -> "
        "save_to_file chain over it."
    ),
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


# --- the periodic digest (Day 18) ------------------------------------------


class PeriodicDigestTask(BaseModel):
    task_id: str = Field(description="The id of the task that was created")
    query: str = Field(description="What it collects")
    interval_seconds: int = Field(description="How often it runs")
    level: str | None = Field(description="The JLPT level it collects, if the query named one")
    created_at: str = Field(description="When it was created, UTC")
    words_per_run: int = Field(description="How many words each run collects")
    note: str = Field(description="What happens next, in words")


@server.tool(title="Create periodic digest")
def create_periodic_digest(
    interval_seconds: Annotated[
        int,
        Field(
            description=(
                "How often the task runs, in seconds. Use something short (10-60) for a "
                f"demonstration; {MIN_INTERVAL_SECONDS} is the minimum and "
                f"{MAX_INTERVAL_SECONDS} (a day) the maximum."
            ),
            ge=MIN_INTERVAL_SECONDS,
            le=MAX_INTERVAL_SECONDS,
        ),
    ],
    query: Annotated[
        str,
        Field(
            description=(
                "What to collect, in words - for example 'Japanese words' or 'N3 words'. "
                "Naming a JLPT level (N1-N5) restricts the collection to that level."
            ),
            min_length=1,
            max_length=120,
        ),
    ],
) -> PeriodicDigestTask:
    """Start collecting Japanese words in the background, every interval_seconds, and keep a
    running digest of them. Each run asks the JLPT vocabulary API for a few words and saves them
    with a timestamp; the digest is then read with get_latest_digest. Call this only when the
    learner asks to start collecting or to set something up regularly - reading what has already
    been collected does not need a new task."""
    task = DigestTaskStorage().create(query, interval_seconds)

    return PeriodicDigestTask(
        task_id=task.id,
        query=task.query,
        interval_seconds=task.interval_seconds,
        level=f"N{task.level}" if task.level else None,
        created_at=task.created_at,
        words_per_run=WORDS_PER_RUN,
        note=(
            f"The first run happens within seconds, then every {task.interval_seconds}s "
            "for as long as the backend runs. Read the result with get_latest_digest."
        ),
    )


class LatestDigest(BaseModel):
    """The aggregate of everything the periodic task has collected so far."""

    found: bool = Field(description="Whether a periodic task exists at all")
    summary: str = Field(description="One line describing what has been collected")
    task_id: str = Field(default="", description="The task the digest belongs to")
    query: str = Field(default="", description="What the task collects")
    interval_seconds: int = Field(default=0, description="How often it runs")
    active: bool = Field(default=False, description="Whether it is still running")
    runs: int = Field(default=0, description="How many runs have happened")
    failed_runs: int = Field(default=0, description="How many of them failed")
    last_run: str = Field(default="", description="When the last run happened, UTC")
    next_run: str = Field(default="", description="When the next one is due, UTC")
    items_collected: int = Field(default=0, description="How many words were collected in total")
    unique_words: int = Field(default=0, description="How many different words among the newest ones")
    levels: dict[str, int] = Field(default_factory=dict, description="JLPT levels among the newest ones")
    latest_items: list[dict[str, str]] = Field(default_factory=list, description="The newest words, newest first")
    last_error: str = Field(default="", description="Why the last run failed, if it did")


@server.tool(title="Get latest digest")
def get_latest_digest() -> LatestDigest:
    """Read the digest the periodic task has been building: how many runs have happened, when the
    last one was, how many words were collected, which JLPT levels they were, the newest ones, and
    a one-line summary. Returns found=false when no periodic task has been created yet."""
    task = DigestTaskStorage().latest()

    if task is None:
        return LatestDigest(
            found=False,
            summary="No periodic digest has been created yet; create_periodic_digest starts one.",
        )

    return LatestDigest(found=True, **build_digest(task, DigestStore().state_of(task.id)))


# --- the pipeline: search -> summarize -> save_to_file (Day 19) -------------
#
# The three stages are ordinary tools, and each one's input is the previous
# one's output - that is the whole contract. The agent runs them in order
# (app/services/agent_pipeline.py); nothing here knows about the others, so
# any of them can also be called on its own.


class SearchFindings(BaseModel):
    """What ``search`` found - and what ``summarize`` and ``save_to_file``
    expect to be given back, unchanged."""

    query: str = Field(description="What was searched for")
    found: bool = Field(description="Whether the JLPT list has anything spelled exactly like it")
    count: int = Field(default=0, description="How many entries were found")
    matches: list[WordMatch] = Field(default_factory=list, description="The entries themselves")
    source: str = Field(default=SOURCE, description="Where the data came from")
    searched_at: str = Field(default="", description="When the search ran, UTC")


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
        raise ToolError(str(error)) from error


class PipelineSummary(BaseModel):
    """What ``summarize`` made of the findings."""

    query: str = Field(description="What was searched for")
    headline: str = Field(description="One line naming the query and what was found")
    summary: str = Field(description="The summary itself, in a few sentences")
    based_on: int = Field(default=0, description="How many entries it was made from")
    levels: dict[str, int] = Field(default_factory=dict, description="JLPT levels among them")
    words: list[str] = Field(default_factory=list, description="The words it covers")
    source: str = Field(default=SOURCE, description="Where the findings came from")
    searched_at: str = Field(default="", description="When the search behind it ran, UTC")
    summarized_at: str = Field(default="", description="When this summary was made, UTC")


@server.tool(title="Summarize")
def summarize(
    findings: Annotated[
        SearchFindings,
        Field(description="Exactly what search returned, passed through unchanged."),
    ],
) -> PipelineSummary:
    """Stage 2 of the chain: turn what search returned into a short summary. This tool does not
    search - it only reads the findings it is given, so it can never disagree with them. Give it
    the result of a search that has already run."""
    return PipelineSummary(**summarize_findings(findings.model_dump()))


class SavedResult(BaseModel):
    """Where the chain's result ended up."""

    status: str = Field(description="'saved' when the file is on disk")
    file_name: str = Field(description="The name it was saved under, with a timestamp in it")
    path: str = Field(description="Its path, relative to the backend's working directory")
    bytes_written: int = Field(default=0, description="How large the file is")
    query: str = Field(default="", description="What the saved result is about")
    saved_at: str = Field(default="", description="When it was written, UTC")


@server.tool(title="Save to file")
def save_to_file(
    summary: Annotated[
        PipelineSummary,
        Field(description="Exactly what summarize returned."),
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
