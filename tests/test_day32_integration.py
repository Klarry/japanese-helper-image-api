"""Day 32: the review against the real local model and the real index.

Opt-in, because it needs a running Ollama with the model, the project index
and network access for the index's embedding model:

    RUN_DAY32_INTEGRATION=1 OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M \\
        .venv-mac/bin/python -m pytest tests/test_day32_integration.py -v

It checks the contract, not the reviewer's judgement: a real review comes
back from Ollama, every finding survived validation against the diff, and
Gemini wrote nothing. What the findings say is recorded by app.evaluate_day32.
"""

import asyncio
import os
from pathlib import Path

import pytest

from app.evaluate_day31 import trap_gemini_generation
from app.services.code_review import CodeReviewer, PullRequest
from app.services.code_review_diff import parse_diff
from app.services.llm_provider import OLLAMA

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DAY32_INTEGRATION") != "1", reason="set RUN_DAY32_INTEGRATION=1 to use the real Ollama and index"
)

FIXTURE = Path(__file__).parent / "fixtures" / "day32" / "docs_architecture_git_shell.diff"


def test_a_real_local_review_of_a_real_diff(monkeypatch):
    trap_gemini_generation()
    diff = FIXTURE.read_text()
    files = {item.path: item for item in parse_diff(diff)}
    pr = PullRequest("local/japanese-helper-image-api", 1, "main", "demo/git-commands", diff,
                     title="Let the developer assistant run git commands")

    result = asyncio.run(CodeReviewer(OLLAMA).review(pr)).as_dict()

    assert result["provider"] == "ollama" and result["model"]
    assert result["validation"]["parse"] in ("ok", "repaired")
    assert result["summary"]

    for finding in result["findings"]:
        assert finding["file"] in files
        assert finding["line"] is None or finding["line"] in files[finding["file"]].new_numbers
