"""Day 19: the automatic chain - search -> summarize -> save_to_file.

The six things the assignment asks to prove, in order: that search returns
data, that summarize is given exactly what search returned, that
save_to_file is given the summary and writes a file, that one message runs
the whole chain without anyone calling the steps by hand, that no data is
lost on the way between stages, and that a failing stage stops the chain and
reaches the learner.

The toolbox here is a stand-in for the MCP client - what the stages do to
the data is covered against the real server, over stdio, in
test_mcp_jlpt_server.py. What is tested here is the orchestration: order,
arguments, and stopping.
"""

import asyncio

import pytest

from app.services.agent_pipeline import PipelineRunner, pipeline_request, pipeline_section
from app.services.agent_tool_planner import PlannedCall
from app.services.mcp_client import McpToolCallResult

FINDINGS = {
    "query": "学習",
    "found": True,
    "count": 1,
    "matches": [
        {
            "word": "学習",
            "reading": "がくしゅう",
            "romaji": "gakushū",
            "meaning": "study, learning",
            "jlpt_level": "N3",
        }
    ],
    "source": "jlpt-vocab-api.vercel.app",
    "searched_at": "2026-09-24T10:00:00+00:00",
}

SUMMARY = {
    "query": "学習",
    "headline": "'学習': 1 JLPT entry, N3 x1.",
    "summary": "'学習': 1 JLPT entry, N3 x1. 学習 - がくしゅう (gakushū) - study, learning - N3.",
    "based_on": 1,
    "levels": {"N3": 1},
    "words": ["学習"],
    "source": "jlpt-vocab-api.vercel.app",
    "searched_at": "2026-09-24T10:00:00+00:00",
    "summarized_at": "2026-09-24T10:00:01+00:00",
}

SAVED = {
    "status": "saved",
    "file_name": "20260924T100001-学習.json",
    "path": "data/pipeline/20260924T100001-学習.json",
    "bytes_written": 939,
    "query": "学習",
    "saved_at": "2026-09-24T10:00:01+00:00",
}


class FakeToolbox:
    """Answers like the MCP client: one result per call, nothing raised."""

    def __init__(self, fails_at: str | None = None, error: str = "the API did not answer") -> None:
        self.calls: list[tuple[str, dict]] = []
        self._fails_at = fails_at
        self._error = error
        self._data = {"search": FINDINGS, "summarize": SUMMARY, "save_to_file": SAVED}

    async def call(self, tool: str, arguments: dict) -> McpToolCallResult:
        self.calls.append((tool, arguments))

        if tool == self._fails_at:
            return McpToolCallResult(name=tool, arguments=arguments, ok=False, error=self._error)

        return McpToolCallResult(name=tool, arguments=arguments, ok=True, data=self._data[tool])


def run_chain(toolbox, calls):
    """The agent's half of one message: read the plan, run what it asks for."""
    request = pipeline_request(calls)
    assert request is not None, "the plan was not read as a pipeline"

    return asyncio.run(PipelineRunner(toolbox).run(request))


def asked_for_everything():
    """What the planner produces for "look 学習 up, summarise it and save it":
    the stages, with only the first one's arguments knowable in advance."""
    return [
        PlannedCall("search", {"query": "学習"}),
        PlannedCall("summarize", {}),
        PlannedCall("save_to_file", {}),
    ]


# --- 1. search returns data -------------------------------------------------


def test_search_runs_first_with_the_word_and_returns_data():
    toolbox = FakeToolbox()

    run = run_chain(toolbox, asked_for_everything())

    assert toolbox.calls[0] == ("search", {"query": "学習"})
    assert run.results[0].name == "search"
    assert run.results[0].ok
    assert run.results[0].data["matches"][0]["reading"] == "がくしゅう"


# --- 2. summarize is given what search returned -----------------------------


def test_summarize_is_handed_the_search_result_itself():
    toolbox = FakeToolbox()

    run_chain(toolbox, asked_for_everything())

    tool, arguments = toolbox.calls[1]
    assert tool == "summarize"
    assert arguments == {"findings": FINDINGS}


def test_summarize_is_never_asked_to_search_again():
    """It gets the findings, not the query: it cannot look anything up."""
    toolbox = FakeToolbox()

    run_chain(toolbox, asked_for_everything())

    assert "query" not in toolbox.calls[1][1]


# --- 3. save_to_file is given the summary and saves -------------------------


def test_save_to_file_gets_the_summary_and_the_findings_behind_it():
    toolbox = FakeToolbox()

    run = run_chain(toolbox, asked_for_everything())

    tool, arguments = toolbox.calls[2]
    assert tool == "save_to_file"
    assert arguments == {"summary": SUMMARY, "findings": FINDINGS}
    assert run.results[2].data["status"] == "saved"
    assert run.saved_as == "20260924T100001-学習.json"


# --- 4. the whole chain runs from one message -------------------------------


def test_one_message_runs_all_three_stages_in_order():
    toolbox = FakeToolbox()

    run = run_chain(toolbox, asked_for_everything())

    assert [tool for tool, _ in toolbox.calls] == ["search", "summarize", "save_to_file"]
    assert run.completed
    assert run.failed_at is None


def test_a_plan_that_only_asks_to_summarise_stops_before_saving():
    """The furthest stage named is how far the chain goes - asking for a
    summary does not quietly write a file."""
    toolbox = FakeToolbox()

    run = run_chain(toolbox, [PlannedCall("search", {"query": "学習"}), PlannedCall("summarize", {})])

    assert [tool for tool, _ in toolbox.calls] == ["search", "summarize"]
    assert run.completed
    assert run.saved_as == ""


def test_a_plain_lookup_is_not_a_pipeline():
    """search on its own is the ordinary single call, not a chain."""
    assert pipeline_request([PlannedCall("search", {"query": "学習"})]) is None
    assert pipeline_request([PlannedCall("get_japanese_word_info", {"word": "学習"})]) is None


def test_a_pipeline_without_anything_to_search_for_is_not_run():
    assert pipeline_request([PlannedCall("save_to_file", {})]) is None


# --- 5. nothing is lost between the stages ----------------------------------


def test_every_stage_receives_the_previous_stage_result_unchanged():
    toolbox = FakeToolbox()

    run = run_chain(toolbox, asked_for_everything())

    search_result, summarize_result, _ = (result.data for result in run.results)
    assert toolbox.calls[1][1]["findings"] is not None
    assert toolbox.calls[1][1]["findings"] == search_result
    assert toolbox.calls[2][1]["summary"] == summarize_result
    assert toolbox.calls[2][1]["findings"] == search_result
    # The word itself survives all three stages, not just the shape.
    assert toolbox.calls[2][1]["summary"]["words"] == ["学習"]
    assert toolbox.calls[2][1]["findings"]["matches"][0]["meaning"] == "study, learning"


def test_the_prompt_block_carries_every_stage_and_the_file_name():
    toolbox = FakeToolbox()

    section = pipeline_section(run_chain(toolbox, asked_for_everything()))

    assert "search -> summarize -> save_to_file" in section
    assert "がくしゅう" in section
    assert "Saved as 20260924T100001-学習.json." in section


# --- 6. a failing stage stops the chain and is reported ---------------------


@pytest.mark.parametrize(
    ("fails_at", "expected"),
    [("search", ["search"]), ("summarize", ["search", "summarize"])],
)
def test_a_failing_stage_stops_everything_after_it(fails_at, expected):
    toolbox = FakeToolbox(fails_at=fails_at)

    run = run_chain(toolbox, asked_for_everything())

    assert [tool for tool, _ in toolbox.calls] == expected
    assert not run.completed
    assert run.failed_at.name == fails_at
    assert run.saved_as == ""


def test_the_failure_is_named_in_the_prompt_block_and_nothing_is_claimed_saved():
    toolbox = FakeToolbox(fails_at="summarize", error="the summary step crashed")

    section = pipeline_section(run_chain(toolbox, asked_for_everything()))

    assert "FAILED: the summary step crashed" in section
    assert "The chain stopped at summarize" in section
    assert "do not claim anything was saved" in section
    assert "Saved as" not in section
