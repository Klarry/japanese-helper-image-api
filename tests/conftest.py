"""Test-wide setup.

app/core/config.py reads GEMINI_API_KEY from the environment at import
time, so it must be set before any test module imports app.main /
app.services.* - conftest.py is loaded by pytest before test collection,
so this runs early enough.
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-key")


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_local_chat_guard():
    """Day 30: the chat endpoint counts requests per address, in memory, for
    the life of the process. Every test starts with fresh counters so one
    test's requests never use up another's rate limit."""
    from app.api.routes import local_llm

    local_llm.reset_guard()
    yield
