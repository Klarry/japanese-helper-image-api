"""Day 35: the release assistant on this repository, with the real local model.

Opt-in - it needs Ollama running with the model, the project index, and network
access for the index's embedding model (local mode sends only titles, paths and
identifier names there):

    RUN_DAY35_INTEGRATION=1 OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M \\
        .venv-mac/bin/python -m pytest tests/test_day35_integration.py -v

Everything is real: the project-git and project-files MCP servers over stdio, this
repository, the RAG index, Ollama. Gemini text generation is a trap. What is checked
is the contract, not the wording: the repository was not changed, every changelog
entry quotes a line that is really in the diff (or a real commit subject), every
finding names a file that really changed, the artifacts are where they should be and
a check is reported with the exit status it really had. Run folders go to output/
(git-ignored); this test removes none of them.
"""

import asyncio
import json
import os
import subprocess

import pytest

from app.evaluate_day31 import trap_gemini_generation
from app.services.llm_provider import OLLAMA
from app.services.project_files import PROJECT_ROOT
from app.services.release_assistant import COMPLETED, PARTIAL, ReleaseAssistant, ReleaseRequest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DAY35_INTEGRATION") != "1", reason="set RUN_DAY35_INTEGRATION=1 to use the real Ollama, index and MCP servers"
)

BASE = os.getenv("RELEASE_TEST_BASE", "HEAD~1")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], capture_output=True, text=True, check=True).stdout


def normal(text: str) -> str:
    return " ".join(str(text).replace("`", "").split()).lower()


@pytest.fixture(autouse=True)
def local_only(monkeypatch):
    trap_gemini_generation()
    monkeypatch.delenv("PROJECT_FILES_ROOT", raising=False)
    monkeypatch.delenv("PROJECT_GIT_ROOT", raising=False)


def test_a_real_release_preparation_on_this_repository():
    before = (git("rev-parse", "HEAD"), git("status", "--porcelain"), git("stash", "list"))
    run = asyncio.run(
        ReleaseAssistant(OLLAMA).prepare(ReleaseRequest(base_ref=BASE, provider=OLLAMA, repository="backend", max_review_batches=1))
    )
    after = (git("rev-parse", "HEAD"), git("status", "--porcelain"), git("stash", "list"))
    document = run.as_dict()

    assert run.status in (COMPLETED, PARTIAL), run.errors
    assert before == after  # nothing reset, cleaned, stashed, committed or edited
    assert document["llm"]["provider"] == OLLAMA and document["llm"]["local"] and document["llm"]["models"]
    assert [item["status"] for item in document["checks"]] == ["not_run"] * len(document["checks"])  # checks are opt-in
    assert document["readiness"]["verdict"] in ("not_verified", "blocked", "ready_with_warnings", "ready")

    names = {item["name"] for item in document["artifacts"]}
    assert names == {"release_report.md", "CHANGELOG.md", "release_notes.md", "review_findings.json", "run.json"}
    assert all((PROJECT_ROOT / item["path"]).is_file() for item in document["artifacts"])
    changed = {item["path"] for item in document["files"]}
    assert changed == set(git("diff", "--name-only", f"{BASE}...HEAD").split())

    for finding in document["findings"]:
        if finding["source"] == "code_review":
            assert finding["file"] in changed and finding["evidence"] and finding["validation_status"] == "evidence_verified"

    subjects = [normal(item["subject"]) for item in document["commits"]["items"]]

    for entry in document["draft"].get("changelog", []):
        assert set(entry["files"]) <= changed and entry["evidence"]

        if entry["evidence_kind"] == "commit_subject":
            assert normal(entry["evidence"]) in subjects
        else:
            diff = git("diff", "-U0", f"{BASE}...HEAD", "--", *entry["files"])
            lines = [normal(line[1:]) for line in diff.splitlines() if line[:1] in "+-" and not line.startswith(("+++", "---"))]
            assert any(normal(entry["evidence"]) in line for line in lines), entry

    record = json.loads((PROJECT_ROOT / document["run_dir"] / "run.json").read_text(encoding="utf-8"))
    assert record["run_id"] == run.run_id and "diff --git" not in json.dumps(record)


def test_a_cheap_check_is_reported_with_its_real_exit_status():
    run = asyncio.run(
        ReleaseAssistant(OLLAMA).prepare(
            ReleaseRequest(base_ref=BASE, provider=OLLAMA, repository="backend", run_checks=True, checks=("backend-compile",), max_review_batches=1)
        )
    )
    checks = {item["id"]: item for item in run.checks}

    assert checks["backend-compile"]["status"] == "passed" and checks["backend-compile"]["exit_code"] == 0
    assert checks["backend-tests"]["status"] == "not_run"  # configured, applicable, not selected
