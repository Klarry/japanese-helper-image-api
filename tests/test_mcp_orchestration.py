"""Day 20: several MCP servers, and the agent routing between them.

The eleven things the assignment asks to prove, in order: that several
servers are registered, that each one's tools are discovered, that search,
summarize and save_to_file are each routed to the right server, that they
run in the right order, that data crosses from one server to the next, that
the whole long flow completes, and that a failing tool, an unreachable
server and an unusable intermediate result each stop it.

Most of it runs against the real servers, each in its own subprocess, over
stdio; only the far end of the HTTP request is the local stand-in for the
Japanese API. The two tests that need a server to misbehave use a registry
of stand-in servers instead - there is no other way to make a healthy one
fail on purpose.
"""

import asyncio
import json
import sys

import pytest
from mcp import StdioServerParameters

from app.services.agent_pipeline import PipelineRequest, PipelineRunner, pipeline_request, usable
from app.services.agent_tool_planner import PlannedCall
from app.services.agent_tools import McpToolbox
from app.services.mcp_client import McpToolCallResult
from app.services.mcp_registry import (
    REGISTERED_SERVERS,
    McpServerRegistry,
    McpServerSpec,
    ToolNotFound,
    japanese_data_server_parameters,
    processing_server_parameters,
    storage_server_parameters,
)

from tests.jlpt_api_standin import closed_port_url, jlpt_api

CHAIN = ("search", "summarize", "save_to_file")


@pytest.fixture(autouse=True)
def _local_traffic_skips_any_proxy(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@pytest.fixture
def pipeline_dir(tmp_path, monkeypatch):
    """Where the storage server writes, in a throwaway directory - through
    the same variable the MCP client passes to the subprocess."""
    directory = tmp_path / "pipeline"
    monkeypatch.setenv("PIPELINE_DIR_PATH", str(directory))
    return directory


def run(coroutine):
    return asyncio.run(coroutine)


def broken_server(module: str = "mcp_servers.does_not_exist"):
    """Parameters for a server that cannot start at all."""
    return lambda: StdioServerParameters(command=sys.executable, args=["-m", module])


# --- 1. several servers are registered --------------------------------------


def test_the_chain_has_a_server_per_stage_and_they_are_all_registered():
    registered = {server.name: server for server in REGISTERED_SERVERS}

    assert {"japanese-data", "processing", "storage"} <= set(registered)
    assert len(registered) == len(REGISTERED_SERVERS), "server names must be unique"
    # each one is startable on its own, as its own process
    for name in ("japanese-data", "processing", "storage"):
        assert registered[name].parameters().args[-1].startswith("mcp_servers.")


# --- 2. every server's tools are discovered ---------------------------------


def test_discovery_asks_every_server_what_it_offers():
    """Not configuration: each server is asked over MCP, and what it reports
    is what the agent is told it can use."""
    discovered = run(McpServerRegistry().discover())

    assert {tool.name for tool in discovered["japanese-data"]} == {"search"}
    assert {tool.name for tool in discovered["processing"]} == {"summarize"}
    assert {tool.name for tool in discovered["storage"]} == {"save_to_file"}
    assert {tool.name for tool in discovered["jlpt-vocab"]} == {
        "get_japanese_word_info",
        "create_periodic_digest",
        "get_latest_digest",
    }


def test_the_agent_is_shown_every_tool_from_every_server():
    tools = run(McpToolbox().tools())

    assert [tool.name for tool in tools][:3] == list(CHAIN)
    assert len(tools) == 6
    assert all(tool.description for tool in tools)


# --- 3-5. each tool is routed to the right server ---------------------------


@pytest.mark.parametrize(
    ("tool", "server"),
    [("search", "japanese-data"), ("summarize", "processing"), ("save_to_file", "storage")],
)
def test_each_stage_is_routed_to_its_own_server(tool, server):
    assert run(McpServerRegistry().server_of(tool)) == server


def test_the_routing_table_covers_the_other_server_too():
    table = run(McpServerRegistry().routing_table())

    assert table == {
        "search": "japanese-data",
        "summarize": "processing",
        "save_to_file": "storage",
        "get_japanese_word_info": "jlpt-vocab",
        "create_periodic_digest": "jlpt-vocab",
        "get_latest_digest": "jlpt-vocab",
    }


def test_a_tool_no_server_offers_is_reported_as_such():
    registry = McpServerRegistry()

    with pytest.raises(ToolNotFound):
        run(registry.server_of("translate_everything"))

    result = run(registry.call("translate_everything", {}))
    assert not result.ok
    assert "not offered by any registered MCP server" in result.error


# --- 6-7. order, and data crossing from one server to the next --------------


def test_the_stages_run_in_order_and_each_one_gets_the_last_one_s_result(monkeypatch, pipeline_dir):
    with jlpt_api() as (url, received):
        monkeypatch.setenv("JLPT_VOCAB_API_URL", url)
        run_result = run(PipelineRunner(McpToolbox()).run(PipelineRequest("学習", CHAIN)))

    assert [result.name for result in run_result.results] == list(CHAIN)
    assert run_result.servers == ("japanese-data", "processing", "storage")
    # only the first server went to the API
    assert received == [{"path": "/api/words", "query": {"word": "学習"}}]

    search, summarize, save = run_result.results
    assert summarize.arguments["findings"] == search.data
    assert save.arguments["summary"] == summarize.data
    assert save.arguments["findings"] == search.data
    # the word itself survived two process boundaries
    assert save.arguments["findings"]["matches"][0]["reading"] == "がくしゅう"


def test_the_plan_decides_how_far_the_chain_goes():
    assert pipeline_request([PlannedCall("search", {"query": "学習"})]) is None
    assert pipeline_request(
        [PlannedCall("search", {"query": "学習"}), PlannedCall("summarize", {})]
    ) == PipelineRequest("学習", ("search", "summarize"))
    assert pipeline_request(
        [PlannedCall("search", {"query": "学習"}), PlannedCall("save_to_file", {})]
    ) == PipelineRequest("学習", CHAIN)


# --- 8. the whole long flow -------------------------------------------------


def test_the_long_flow_completes_across_three_servers(monkeypatch, pipeline_dir):
    with jlpt_api() as (url, _):
        monkeypatch.setenv("JLPT_VOCAB_API_URL", url)
        run_result = run(PipelineRunner(McpToolbox()).run(PipelineRequest("勉強", CHAIN)))

    assert run_result.completed
    assert run_result.failed_at is None
    saved = json.loads((pipeline_dir / run_result.saved_as).read_text(encoding="utf-8"))
    assert saved["pipeline"] == ["search", "summarize", "save_to_file"]
    # 勉強 has two entries in the JLPT list; both of them made it to the end
    assert len(saved["findings"]["matches"]) == 2
    assert saved["summary"]["levels"] == {"N3": 1, "N5": 1}


# --- 9. a tool on one server fails ------------------------------------------


def test_a_failing_tool_stops_the_chain_where_it_failed(monkeypatch, pipeline_dir):
    """The API is unreachable, so the tool on japanese-data runs and fails;
    processing and storage are never asked."""
    monkeypatch.setenv("JLPT_VOCAB_API_URL", closed_port_url())

    run_result = run(PipelineRunner(McpToolbox()).run(PipelineRequest("学習", CHAIN)))

    assert [result.name for result in run_result.results] == ["search"]
    assert not run_result.completed
    assert run_result.failed_at.name == "search"
    assert run_result.saved_as == ""
    assert not pipeline_dir.exists() or list(pipeline_dir.iterdir()) == []


# --- 10. a server that cannot be reached at all -----------------------------


def test_an_unreachable_server_takes_only_its_own_tools_with_it(monkeypatch, pipeline_dir):
    """processing cannot start. Discovery still finds the other two, the
    chain still starts, and it stops at the stage nobody can run."""
    servers = [
        McpServerSpec("japanese-data", "search", japanese_data_server_parameters),
        McpServerSpec("processing", "summarize", broken_server()),
        McpServerSpec("storage", "save", storage_server_parameters),
    ]
    toolbox = McpToolbox(servers)

    with jlpt_api() as (url, _):
        monkeypatch.setenv("JLPT_VOCAB_API_URL", url)
        discovered = run(toolbox.discover())
        run_result = run(PipelineRunner(toolbox).run(PipelineRequest("学習", CHAIN)))

    assert set(discovered) == {"japanese-data", "storage"}
    assert [result.name for result in run_result.results] == ["search", "summarize"]
    assert run_result.results[0].ok
    assert not run_result.results[1].ok
    assert "not offered by any registered MCP server" in run_result.results[1].error
    assert run_result.saved_as == ""
    assert not pipeline_dir.exists() or list(pipeline_dir.iterdir()) == []


def test_a_server_that_disappears_between_discovery_and_the_call_is_reported():
    """Discovered, then gone: the registry says which server is unavailable
    rather than pretending the tool does not exist."""
    registry = McpServerRegistry([McpServerSpec("processing", "summarize", processing_server_parameters)])
    assert run(registry.routing_table()) == {"summarize": "processing"}

    registry._servers = (McpServerSpec("processing", "summarize", broken_server()),)
    result = run(registry.call("summarize", {"findings": {}}))

    assert not result.ok
    assert "MCP server 'processing' is unavailable" in result.error


# --- 11. an intermediate result the next stage cannot use -------------------


def test_an_unusable_result_is_caught_before_it_reaches_the_next_stage():
    """The one failure MCP cannot report: the call worked, the data did not."""
    assert usable("search", {"query": "学習", "matches": []}) == ""
    assert "no 'matches' list" in usable("search", {"query": "学習"})
    assert "no 'query'" in usable("search", {"matches": []})
    assert "not an object" in usable("search", "just some text")
    assert "empty 'summary'" in usable("summarize", {"query": "学習", "summary": ""})


def test_a_stage_that_answers_with_nonsense_stops_the_chain():
    class SaysNothingUseful(McpToolbox):
        def __init__(self):
            super().__init__(REGISTERED_SERVERS)
            self.called: list[str] = []

        async def routing_table(self):
            return {"search": "japanese-data", "summarize": "processing", "save_to_file": "storage"}

        async def call(self, tool, arguments):
            self.called.append(tool)
            data = {"query": "学習", "matches": "not a list"} if tool == "search" else {}

            return McpToolCallResult(name=tool, arguments=arguments, ok=True, data=data)

    toolbox = SaysNothingUseful()
    run_result = run(PipelineRunner(toolbox).run(PipelineRequest("学習", CHAIN)))

    assert toolbox.called == ["search"], "summarize must not be given data it cannot use"
    assert not run_result.completed
    assert run_result.failed_at.name == "search"
    assert "invalid output from search" in run_result.failed_at.error
