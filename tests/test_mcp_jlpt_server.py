"""Day 17: the JLPT vocabulary MCP server, end to end over stdio.

Each test starts ``python -m mcp_servers.jlpt_vocab`` as a real subprocess
and talks MCP to it. The server calls the backend's real client for the
API; only the API itself is the local stand-in, pointed at through
JLPT_VOCAB_API_URL - which is exactly how the MCP client hands it over.
"""

import asyncio
import json

import pytest

from app.services.agent_pipeline import PipelineRequest, PipelineRunner
from app.services.agent_tools import McpToolbox
from app.services.digest import DigestStore, DigestTaskStorage, collect_once
from app.services.mcp_client import (
    call_server_tool,
    jlpt_vocab_server_parameters,
    list_server_tools,
)

from tests.jlpt_api_standin import closed_port_url, jlpt_api

TOOL = "get_japanese_word_info"


@pytest.fixture(autouse=True)
def _local_traffic_skips_any_proxy(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


def _call(word, monkeypatch, api_url):
    monkeypatch.setenv("JLPT_VOCAB_API_URL", api_url)
    return asyncio.run(call_server_tool(TOOL, {"word": word}, jlpt_vocab_server_parameters()))


def test_the_server_starts_and_lists_the_tool():
    """1-2. The server starts, and the tool is in list_tools() with a
    description and a described parameter."""
    report = asyncio.run(list_server_tools(jlpt_vocab_server_parameters()))

    assert report.server_name == "jlpt-vocab"
    tool = {tool.name: tool for tool in report.tools}[TOOL]
    assert {tool.name for tool in report.tools} == {
        TOOL,
        "create_periodic_digest",
        "get_latest_digest",
        "search",
        "summarize",
        "save_to_file",
    }
    assert "JLPT" in tool.description
    assert tool.parameters == ("word",)
    assert "Japanese" in tool.input_schema["properties"]["word"]["description"]
    assert tool.input_schema["required"] == ["word"]


def test_the_tool_gets_the_word_asks_the_api_and_returns_structured_data(monkeypatch):
    """4-6. The parameter reaches the tool, the tool reaches the API with it,
    and the answer comes back through MCP as structured data."""
    with jlpt_api() as (url, received):
        result = _call("学習", monkeypatch, url)

    assert received == [{"path": "/api/words", "query": {"word": "学習"}}]
    assert result.ok
    assert result.data["found"] is True
    assert result.data["matches"] == [
        {"word": "学習", "reading": "がくしゅう", "romaji": "gakushū", "meaning": "study, learning", "jlpt_level": "N3"}
    ]
    assert result.data["source"] == "jlpt-vocab-api.vercel.app"


def test_a_word_with_two_entries_returns_both(monkeypatch):
    with jlpt_api() as (url, _):
        result = _call("勉強", monkeypatch, url)

    assert [match["jlpt_level"] for match in result.data["matches"]] == ["N5", "N3"]


def test_a_word_the_list_does_not_have_is_found_false_not_an_error(monkeypatch):
    with jlpt_api() as (url, _):
        result = _call("ぷりんたー", monkeypatch, url)

    assert result.ok
    assert result.data["found"] is False
    assert result.data["matches"] == []


def test_an_api_that_fails_is_a_failed_tool_call_with_the_reason(monkeypatch):
    """8. The connection works, the tool does not: ok=False, and the reason
    travels back through MCP instead of being swallowed."""
    with jlpt_api(status=503) as (url, _):
        result = _call("学習", monkeypatch, url)

    assert not result.ok
    assert "status 503" in result.error


def test_an_api_that_is_down_is_a_failed_tool_call(monkeypatch):
    result = _call("学習", monkeypatch, closed_port_url())

    assert not result.ok
    assert "unreachable" in result.error


def test_an_argument_outside_the_schema_is_refused_by_the_server(monkeypatch):
    with jlpt_api() as (url, received):
        result = _call("", monkeypatch, url)

    assert not result.ok
    assert received == []


# --- the periodic digest (Day 18) ------------------------------------------


@pytest.fixture
def digest_files(tmp_path, monkeypatch):
    """The two JSON files, in a throwaway directory. The MCP client passes
    these very variables to the server subprocess."""
    monkeypatch.setenv("DIGEST_TASKS_FILE_PATH", str(tmp_path / "digest_tasks.json"))
    monkeypatch.setenv("DIGEST_STORE_FILE_PATH", str(tmp_path / "digest_store.json"))
    return tmp_path


def _call_tool(name, arguments):
    return asyncio.run(call_server_tool(name, arguments, jlpt_vocab_server_parameters()))


def test_both_digest_tools_are_listed_with_their_parameters(digest_files):
    tools = {tool.name: tool for tool in asyncio.run(list_server_tools(jlpt_vocab_server_parameters())).tools}

    create = tools["create_periodic_digest"]
    assert sorted(create.parameters) == ["interval_seconds", "query"]
    assert create.input_schema["properties"]["interval_seconds"]["type"] == "integer"
    assert "seconds" in create.input_schema["properties"]["interval_seconds"]["description"]
    assert "JLPT level" in create.input_schema["properties"]["query"]["description"]
    assert tools["get_latest_digest"].parameters == ()


def test_creating_a_task_through_mcp_writes_it_to_disk(digest_files):
    result = _call_tool("create_periodic_digest", {"interval_seconds": 15, "query": "N5 words"})

    assert result.ok
    assert result.data["task_id"] == "digest-1"
    assert result.data["interval_seconds"] == 15
    assert result.data["level"] == "N5"
    stored = json.loads((digest_files / "digest_tasks.json").read_text(encoding="utf-8"))
    assert stored["tasks"][0]["query"] == "N5 words"


def test_the_digest_is_empty_until_a_task_exists(digest_files):
    result = _call_tool("get_latest_digest", {})

    assert result.ok
    assert result.data["found"] is False
    assert "No periodic digest" in result.data["summary"]


def test_the_digest_read_through_mcp_is_what_the_runs_wrote(digest_files, monkeypatch):
    """The run happens in this process (as it does in the backend); the
    digest is read in the server's subprocess. Two processes, one file."""
    _call_tool("create_periodic_digest", {"interval_seconds": 20, "query": "N5 words"})
    task = DigestTaskStorage().latest()

    with jlpt_api() as (url, _):
        monkeypatch.setenv("JLPT_VOCAB_API_URL", url)
        asyncio.run(collect_once(task, DigestStore()))
        asyncio.run(collect_once(task, DigestStore()))

    result = _call_tool("get_latest_digest", {})

    assert result.data["found"] is True
    assert result.data["runs"] == 2
    assert result.data["items_collected"] == 6
    assert result.data["last_run"]
    assert result.data["levels"] == {"N5": 6}
    assert "2 run(s)" in result.data["summary"]
    assert len(result.data["latest_items"]) == 5


def test_an_interval_the_schema_forbids_never_creates_a_task(digest_files):
    result = _call_tool("create_periodic_digest", {"interval_seconds": 1, "query": "N5 words"})

    assert not result.ok
    assert not (digest_files / "digest_tasks.json").exists()


# --- the pipeline: search -> summarize -> save_to_file (Day 19) -------------


@pytest.fixture
def pipeline_dir(tmp_path, monkeypatch):
    """Where save_to_file writes, in a throwaway directory - through the same
    variable the MCP client passes to the subprocess."""
    directory = tmp_path / "pipeline"
    monkeypatch.setenv("PIPELINE_DIR_PATH", str(directory))
    return directory


def test_the_three_pipeline_tools_are_listed_with_their_parameters():
    tools = {tool.name: tool for tool in asyncio.run(list_server_tools(jlpt_vocab_server_parameters())).tools}

    assert tools["search"].parameters == ("query",)
    assert "Japanese" in tools["search"].input_schema["properties"]["query"]["description"]
    assert tools["summarize"].parameters == ("findings",)
    assert sorted(tools["save_to_file"].parameters) == ["findings", "summary"]


def test_search_asks_the_api_and_returns_structured_findings(monkeypatch):
    """1. The first stage gets real data from the API the app uses."""
    with jlpt_api() as (url, received):
        monkeypatch.setenv("JLPT_VOCAB_API_URL", url)
        result = _call_tool("search", {"query": "学習"})

    assert received == [{"path": "/api/words", "query": {"word": "学習"}}]
    assert result.ok
    assert result.data["found"] is True
    assert result.data["count"] == 1
    assert result.data["matches"][0]["reading"] == "がくしゅう"
    assert result.data["matches"][0]["jlpt_level"] == "N3"
    assert result.data["searched_at"]


def test_summarize_works_on_what_search_returned_and_asks_no_one(monkeypatch):
    """2. The second stage is given the first stage's own result, and does
    not reach the API at all - the stand-in records no request for it."""
    with jlpt_api() as (url, received):
        monkeypatch.setenv("JLPT_VOCAB_API_URL", url)
        findings = _call_tool("search", {"query": "勉強"}).data
        requests_after_search = len(received)

        result = _call_tool("summarize", {"findings": findings})

        assert len(received) == requests_after_search

    assert result.ok
    assert result.data["based_on"] == 2
    assert result.data["levels"] == {"N3": 1, "N5": 1}
    assert result.data["words"] == ["勉強", "勉強"]
    assert "べんきょう" in result.data["summary"]
    assert result.data["searched_at"] == findings["searched_at"]


def test_save_to_file_writes_a_timestamped_json_with_both_halves(monkeypatch, pipeline_dir):
    """3. The third stage saves what it was given, under a name with a
    timestamp in it, and says where."""
    with jlpt_api() as (url, _):
        monkeypatch.setenv("JLPT_VOCAB_API_URL", url)
        findings = _call_tool("search", {"query": "学習"}).data
        summary = _call_tool("summarize", {"findings": findings}).data

    result = _call_tool("save_to_file", {"summary": summary, "findings": findings})

    assert result.ok
    assert result.data["status"] == "saved"
    assert result.data["file_name"].endswith(".json")
    saved = json.loads((pipeline_dir / result.data["file_name"]).read_text(encoding="utf-8"))
    assert saved["query"] == "学習"
    assert saved["pipeline"] == ["search", "summarize", "save_to_file"]
    assert saved["summary"]["headline"] == summary["headline"]
    assert saved["findings"]["matches"] == findings["matches"]
    # The name carries the moment it was written: 20260924T101530-学習.json
    stamp, _, _ = result.data["file_name"].partition("-")
    assert len(stamp) == len("20260924T101530")
    assert saved["saved_at"].startswith(stamp[:4])


def test_the_whole_chain_runs_over_the_real_server(monkeypatch, pipeline_dir):
    """4-5. The runner drives the real tools, over stdio, in order - and what
    lands on disk is what the first stage found."""
    with jlpt_api() as (url, received):
        monkeypatch.setenv("JLPT_VOCAB_API_URL", url)
        run = asyncio.run(
            PipelineRunner(McpToolbox()).run(PipelineRequest("学習", ("search", "summarize", "save_to_file")))
        )

    assert [result.name for result in run.results] == ["search", "summarize", "save_to_file"]
    assert run.completed
    # One request to the API for the whole chain: only the first stage asks.
    assert received == [{"path": "/api/words", "query": {"word": "学習"}}]
    saved = json.loads((pipeline_dir / run.saved_as).read_text(encoding="utf-8"))
    assert saved["findings"]["matches"][0]["meaning"] == "study, learning"
    assert "がくしゅう" in saved["summary"]["summary"]


def test_a_failing_search_stops_the_chain_before_anything_is_saved(monkeypatch, pipeline_dir):
    """6. The API is unreachable, so stage one fails - and stages two and
    three do not run."""
    monkeypatch.setenv("JLPT_VOCAB_API_URL", closed_port_url())

    run = asyncio.run(
        PipelineRunner(McpToolbox()).run(PipelineRequest("学習", ("search", "summarize", "save_to_file")))
    )

    assert [result.name for result in run.results] == ["search"]
    assert not run.completed
    assert run.failed_at.name == "search"
    assert run.saved_as == ""
    assert not pipeline_dir.exists() or list(pipeline_dir.iterdir()) == []
