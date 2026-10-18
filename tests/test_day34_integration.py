"""Day 34: the file assistant on this repository, with the real local model.

Opt-in - it needs Ollama running with the model:

    RUN_DAY34_INTEGRATION=1 .venv-mac/bin/python -m pytest tests/test_day34_integration.py -v

Everything is real: the project-files MCP server over stdio, this
repository's files, Ollama. Every Gemini function is a trap. What is checked
is the contract, not the wording: the model only read, every flow step and
every finding that is reported as verified quotes a line that is really
there, every diff applies to the file as it is and the file itself was not
changed. Reports, proposals and run records go to output/ (git-ignored).
"""

import asyncio
import os

import pytest

from app.services import gemini_service
from app.services.file_assistant import COMPLETED, DOCS, PARTIAL, READ_TOOLS, USAGE, FileAssistant
from app.services.llm_provider import OLLAMA
from app.services.project_files import ProjectFiles, apply_unified_diff, sha256_bytes

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DAY34_INTEGRATION") != "1", reason="set RUN_DAY34_INTEGRATION=1 to use the real Ollama and MCP server"
)

USAGE_TASK = ("Find every place where OllamaService is used. Read the relevant files and explain how requests flow "
              "from the API to the local model.")
DOCS_TASK = ("Compare the Ollama implementation with the documentation, identify verified discrepancies, and generate "
             "an updated documentation proposal and unified diff.")


@pytest.fixture(autouse=True)
def no_cloud(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Gemini was called during a local file-assistant run")

    for name in ("_generate", "_post_to_gemini", "embed_texts", "generate_text_with_usage"):
        monkeypatch.setattr(gemini_service, name, forbidden, raising=False)

    monkeypatch.delenv("PROJECT_FILES_ROOT", raising=False)


def normal(text: str) -> str:
    return " ".join(str(text).replace("`", "").split())


def quoted(files: ProjectFiles, path: str, line: int, quote: str) -> bool:
    """The quote is on that line of the file as it is (or starts there and runs on)."""
    lines = (files.root / path).read_text(encoding="utf-8").splitlines()
    return normal(quote) in normal(" ".join(lines[line - 1 : line + 2]))


def common(result: dict, workflow: str) -> ProjectFiles:
    files = ProjectFiles()

    assert result["provider"] == OLLAMA and result["model"]
    assert result["workflow"] == workflow and result["status"] in (COMPLETED, PARTIAL), result["errors"]
    assert result["files_inspected"] and result["artifacts"]
    assert all(call["tool"] in READ_TOOLS for call in result["tool_calls"] if call["planned_by"] == "model")

    for artifact in result["artifacts"]:
        assert (files.root / artifact["path"]).is_file() and artifact["path"].startswith(f"{files.output_rel}/")

    assert (files.root / result["run_record"]).is_file()

    return files


def test_workflow_a_usages_of_ollama_service():
    result = asyncio.run(FileAssistant(OLLAMA).run(USAGE_TASK)).as_dict()
    files = common(result, USAGE)

    assert any(row["path"] == "app/services/llm_provider.py" for row in result["findings"]["usages"])

    for step in result["findings"]["flow"]:
        assert quoted(files, step["path"], step["line"], step["quote"]), step


def test_workflow_b_ollama_documentation_against_the_code():
    result = asyncio.run(FileAssistant(OLLAMA).run(DOCS_TASK)).as_dict()
    files = common(result, DOCS)

    for finding in result["findings"]["findings"]:
        if finding["state"] == "rejected":
            continue

        assert quoted(files, finding["code_path"], finding["code_line"], finding["code_quote"]), finding
        assert normal(finding["doc_quote"]) in normal((files.root / finding["doc_path"]).read_text(encoding="utf-8")), finding

    for diff in result["diffs"]:
        current = (files.root / diff["path"]).read_bytes()
        proposal = (files.root / diff["proposal"]).read_text(encoding="utf-8")

        assert sha256_bytes(current) == diff["base_sha256"]  # the original was not touched
        assert apply_unified_diff(current.decode("utf-8").replace("\r\n", "\n"), diff["diff"]) == proposal
