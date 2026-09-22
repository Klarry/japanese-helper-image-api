"""Day 17: the JLPT vocabulary MCP server, end to end over stdio.

Each test starts ``python -m mcp_servers.jlpt_vocab`` as a real subprocess
and talks MCP to it. The server calls the backend's real client for the
API; only the API itself is the local stand-in, pointed at through
JLPT_VOCAB_API_URL - which is exactly how the MCP client hands it over.
"""

import asyncio

import pytest

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
