"""The data the pipeline servers hand to each other.

Day 19 had the three stages on one server, so their models lived with it.
Day 20 put each stage on a server of its own - and the whole point of the
chain is that what one server returns is what the next one accepts, so the
shapes belong in one place that all three import. This is the contract
between the servers, written once.
"""

from pydantic import BaseModel, Field

SOURCE = "jlpt-vocab-api.vercel.app"


class WordMatch(BaseModel):
    word: str = Field(description="The word as written in Japanese")
    reading: str | None = Field(description="Reading in kana (furigana)")
    romaji: str | None = Field(description="Reading in Latin letters")
    meaning: str | None = Field(description="English meaning")
    jlpt_level: str | None = Field(description="JLPT level, N5 (easiest) to N1")


class SearchFindings(BaseModel):
    """What ``search`` found on the japanese-data server - and what
    ``summarize`` and ``save_to_file`` expect to be given back, unchanged."""

    query: str = Field(description="What was searched for")
    found: bool = Field(description="Whether the JLPT list has anything spelled exactly like it")
    count: int = Field(default=0, description="How many entries were found")
    matches: list[WordMatch] = Field(default_factory=list, description="The entries themselves")
    source: str = Field(default=SOURCE, description="Where the data came from")
    searched_at: str = Field(default="", description="When the search ran, UTC")


class PipelineSummary(BaseModel):
    """What ``summarize`` made of the findings, on the processing server."""

    query: str = Field(description="What was searched for")
    headline: str = Field(description="One line naming the query and what was found")
    summary: str = Field(description="The summary itself, in a few sentences")
    based_on: int = Field(default=0, description="How many entries it was made from")
    levels: dict[str, int] = Field(default_factory=dict, description="JLPT levels among them")
    words: list[str] = Field(default_factory=list, description="The words it covers")
    source: str = Field(default=SOURCE, description="Where the findings came from")
    searched_at: str = Field(default="", description="When the search behind it ran, UTC")
    summarized_at: str = Field(default="", description="When this summary was made, UTC")


class SavedResult(BaseModel):
    """Where the storage server put the chain's result."""

    status: str = Field(description="'saved' when the file is on disk")
    file_name: str = Field(description="The name it was saved under, with a timestamp in it")
    path: str = Field(description="Its path, relative to the backend's working directory")
    bytes_written: int = Field(default=0, description="How large the file is")
    query: str = Field(default="", description="What the saved result is about")
    saved_at: str = Field(default="", description="When it was written, UTC")
