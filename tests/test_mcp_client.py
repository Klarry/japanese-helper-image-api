"""Day 16: the MCP client against the real local server, in a real subprocess.

Nothing is faked here. Each test starts mcp_servers/japanese_learning.py over
stdio, runs the handshake and list_tools(), and shuts it down - the same path
`python -m app.services.mcp_client` takes.
"""

import asyncio
import logging
import sys

import pytest

from mcp import StdioServerParameters

from app.services.mcp_client import McpConnectionError, list_server_tools

EXPECTED_TOOLS = {"search_japanese_word", "get_kanji_info", "create_example_sentence"}


def _connect(server: StdioServerParameters | None = None):
    return asyncio.run(list_server_tools(server))


def test_the_client_connects_initializes_and_gets_the_tools():
    report = _connect()

    assert report.server_name == "japanese-learning"
    assert report.server_version == "1.0.0"
    assert report.protocol_version
    assert {tool.name for tool in report.tools} == EXPECTED_TOOLS


def test_every_tool_comes_back_with_a_description_and_its_parameters():
    """The description is what a model will later choose a tool by, and the
    input schema is what it will call it with - both have to survive the
    trip, not just the names."""
    tools = {tool.name: tool for tool in _connect().tools}

    assert all(tool.description for tool in tools.values())
    assert tools["search_japanese_word"].parameters == ("query",)
    assert tools["get_kanji_info"].parameters == ("kanji",)
    assert tools["create_example_sentence"].parameters == ("word",)
    assert "kanji" in tools["get_kanji_info"].description


def test_the_connection_and_the_tools_are_logged(caplog):
    with caplog.at_level(logging.INFO, logger="app.services.mcp_client"):
        _connect()

    log = caplog.text
    assert "session initialized - server 'japanese-learning'" in log
    assert "list_tools() returned 3 tool(s)" in log
    for name in EXPECTED_TOOLS:
        assert name in log


def test_a_server_that_cannot_start_is_a_failed_connection_not_an_empty_one(caplog):
    missing = StdioServerParameters(command=sys.executable, args=["mcp_servers/does_not_exist.py"])

    with caplog.at_level(logging.INFO, logger="app.services.mcp_client"):
        with pytest.raises(McpConnectionError):
            _connect(missing)

    assert "connection failed" in caplog.text
    assert "list_tools() returned" not in caplog.text
