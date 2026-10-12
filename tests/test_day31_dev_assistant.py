"""Day 31: the developer assistant - /help, the docs in the index, and the
live Git branch through MCP.

Most tests stand in for the model and for Git. Two do not: the ones marked
"real" below start the actual project-git MCP server and read the actual
repository's branch, and build a real (offline-embedded) index from the
actual README and docs/ - they prove the pieces work, not the prose of an
answer from a real model, which evaluate_day31 checks on a running setup.
"""

import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.routes import dev_assistant as route
from app.index_documents import index_strategy
from app.main import app
from app.services import gemini_service, mcp_client, mcp_registry, ollama_service
from app.services.chunking import STRUCTURAL
from app.services.dev_assistant import (
    CAPABILITIES,
    DevAssistant,
    NotAHelpCommand,
    ToolUse,
    describe_branch,
    parse,
    read_git_branch,
)
from app.services.document_loader import CORPUS, PROJECT_ROOT, CorpusRoot, load_documents
from app.services.embedding_service import EmbeddingService
from app.services.llm_provider import OLLAMA, Generated, OllamaProvider
from app.services.rag_agent import RagAgent
from app.services.rag_prompt import DONT_KNOW
from app.services.rag_response import ANSWERED, DISABLED, INSUFFICIENT
from app.services.rag_retriever import RAGRetriever, Retrieval, RetrievedChunk
from app.services.rag_settings import DEFAULT_SETTINGS
from mcp_servers.project_git import GitError, read_branch

client = TestClient(app)
OPEN_GATE = DEFAULT_SETTINGS.with_overrides(answer_threshold=0.2, query_rewrite=False)

QUOTE = "Every chunk keeps its file, section and `chunk_id`."
CHUNK_TEXT = f"The corpus is README.md, docs/*.md, app/**/*.py, mcp_servers/*.py and data/documents/. {QUOTE}"


def chunk(score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="docs-architecture_7",
        text=CHUNK_TEXT,
        source="docs",
        file="docs/architecture.md",
        title="Architecture",
        section="Architecture > RAG",
        strategy="structural",
        position=7,
        score=score,
    )


class Retriever:
    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.queries: list[str] = []

    async def retrieve(self, query, top_k=10):
        self.queries.append(query)
        return Retrieval(
            query=query, chunks=(chunk(self.score),), seconds=0.01, top_k=top_k,
            strategy="structural", embedding_model="local-hashing",
        )


class Model:
    """A provider that records its prompts and gives one fixed reply."""

    def __init__(self, reply: str, name: str = OLLAMA) -> None:
        self.name = name
        self.model = "qwen3:4b"
        self.prompts: list[str] = []
        self.reply = reply

    async def generate(self, prompt, temperature=None):
        self.prompts.append(prompt)
        return Generated(text=self.reply, model=self.model, input_tokens=3, output_tokens=4)


def cited(*quotes: str, answer: str = "Chunks keep their file and section.") -> str:
    return json.dumps({"answer": answer, "citations": [{"source": 1, "quote": quote} for quote in quotes]})


def assistant(reply: str = "", branch: ToolUse | None = None, score: float = 0.9):
    retriever = Retriever(score)
    model = Model(reply or cited(QUOTE))
    agent = RagAgent(retriever=retriever, provider=model, settings=OPEN_GATE, check_support=False)
    calls: list[str] = []

    async def git():
        calls.append("get_git_branch")
        return branch or ToolUse("get_git_branch", "project-git", True, 5, result={"branch": "feature/dev-assistant", "detached": False})

    return DevAssistant(OLLAMA, agent=agent, git_reader=git, settings=OPEN_GATE), retriever, model, calls


def no_cloud(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the developer assistant called Gemini")

    for name in ("_generate", "_post_to_gemini", "generate_text", "generate_text_with_usage"):
        monkeypatch.setattr(gemini_service, name, forbidden)


# --- parsing /help -------------------------------------------------------------------


def test_help_alone_is_the_overview():
    request = parse("/help")

    assert request.question == ""
    assert not request.wants_git and not request.wants_docs


@pytest.mark.parametrize(
    "message, git, docs",
    [
        ("/help How does RAG work in this project?", False, True),
        ("/help Explain the project architecture.", False, True),
        ("/help Which Git branch is currently active?", True, False),
        ("/help какая ветка сейчас?", True, False),
        ("/help Which branch is active, and how does RAG work?", True, True),
        # "git" alone is a documentation question - no tool call for it.
        ("/help How does the project use git?", False, True),
    ],
)
def test_the_question_decides_what_is_needed(message, git, docs):
    request = parse(message)

    assert (request.wants_git, request.wants_docs) == (git, docs)


def test_a_combined_question_sends_rag_only_the_documentation_part():
    request = parse("/help Which branch is active, and how does RAG work in this project?")

    assert "branch" not in request.docs_question.lower()
    assert "rag" in request.docs_question.lower()


@pytest.mark.parametrize("message", ["hello", "/helpme", "/status", ""])
def test_anything_but_help_is_refused(message):
    with pytest.raises(NotAHelpCommand):
        parse(message)


# --- the assistant --------------------------------------------------------------------


def test_help_without_a_question_calls_nothing():
    helper, retriever, model, calls = assistant()
    answer = asyncio.run(helper.ask("/help"))

    assert answer.answer == CAPABILITIES
    assert answer.rag_status == DISABLED
    assert answer.model is None
    assert (retriever.queries, model.prompts, calls) == ([], [], [])


def test_a_documentation_question_is_answered_from_the_index_with_sources():
    helper, retriever, model, calls = assistant()
    answer = asyncio.run(helper.ask("/help How does RAG work in this project?"))

    assert answer.rag_status == ANSWERED
    assert answer.sources[0]["file"] == "docs/architecture.md"
    assert answer.citations[0]["quote"] == QUOTE
    assert answer.provider == OLLAMA and answer.model == "qwen3:4b"
    assert answer.tools_used == []
    assert calls == []
    assert "retrieval_ms" in answer.timing and "generation_ms" in answer.timing


def test_the_branch_comes_from_the_tool_not_the_index():
    helper, retriever, model, calls = assistant()
    answer = asyncio.run(helper.ask("/help Which Git branch is currently active?"))

    assert "`feature/dev-assistant`" in answer.answer
    assert answer.tools_used[0]["name"] == "get_git_branch"
    assert answer.tools_used[0]["ok"] is True
    assert answer.rag_status == DISABLED
    assert (retriever.queries, model.prompts) == ([], [])


def test_a_combined_question_uses_both():
    helper, retriever, model, calls = assistant()
    answer = asyncio.run(helper.ask("/help Which branch is active, and how does RAG work in this project?"))

    assert calls == ["get_git_branch"]
    assert retriever.queries and "branch" not in retriever.queries[0].lower()
    assert answer.answer.startswith("The current Git branch is `feature/dev-assistant`")
    assert answer.sources


def test_a_git_failure_is_said_not_guessed():
    failed = ToolUse("get_git_branch", "project-git", False, 3, error="not_a_git_repository: …")
    helper, *_ = assistant(branch=failed)
    answer = asyncio.run(helper.ask("/help Which Git branch is currently active?"))

    assert answer.answer.startswith("I could not determine the current Git branch")
    assert answer.tools_used[0]["ok"] is False


def test_a_detached_head_is_reported_as_detached():
    sentence = describe_branch(
        ToolUse("get_git_branch", "project-git", True, 3, result={"branch": None, "detached": True, "commit": "a1b2c3d"})
    )

    assert "detached HEAD" in sentence and "a1b2c3d" in sentence and "`None`" not in sentence


def test_too_little_context_is_refused_not_invented():
    helper, retriever, model, calls = assistant(score=0.2)
    helper._settings = DEFAULT_SETTINGS.with_overrides(answer_threshold=0.99, query_rewrite=False)
    answer = asyncio.run(helper.ask("/help Which Kubernetes namespace is the backend deployed to?"))

    assert answer.rag_status == INSUFFICIENT
    assert answer.sources == [] and answer.citations == []
    assert model.prompts == []


def test_an_invented_quote_does_not_become_a_citation():
    helper, *_ = assistant(reply=cited(QUOTE, "The index is rebuilt every night by a cron job."))
    answer = asyncio.run(helper.ask("/help How does RAG work in this project?"))

    assert [citation["quote"] for citation in answer.citations] == [QUOTE]


# --- the git tool itself (a real git, in temporary repositories) ---------------------------


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True, capture_output=True,
    )


def _repo(path: Path, branch: str = "feature/dev-assistant") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", branch)
    _git(path, "commit", "-q", "--allow-empty", "-m", "first")
    return path


def test_the_branch_is_read_from_git(tmp_path):
    state = read_branch(_repo(tmp_path / "repo"))

    assert state.branch == "feature/dev-assistant"
    assert state.detached is False
    assert state.repository == "repo"


def test_a_detached_head_has_no_branch_but_a_commit(tmp_path):
    repo = _repo(tmp_path / "repo")
    _git(repo, "checkout", "-q", "--detach")
    state = read_branch(repo)

    assert state.branch is None
    assert state.detached is True
    assert state.commit


def test_a_directory_that_is_not_a_repository_is_an_error(tmp_path):
    with pytest.raises(GitError) as caught:
        read_branch(tmp_path)

    assert caught.value.kind == "not_a_git_repository"


def test_a_folder_inside_another_repository_is_refused(tmp_path):
    repo = _repo(tmp_path / "repo")
    inner = repo / "sub"
    inner.mkdir()

    with pytest.raises(GitError) as caught:
        read_branch(inner)

    assert caught.value.kind == "not_project_repository"


def test_no_git_installed_is_an_error_not_a_crash(monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing)

    with pytest.raises(GitError) as caught:
        read_branch(tmp_path)

    assert caught.value.kind == "git_not_found"


def test_the_tool_takes_no_arguments_and_runs_no_shell():
    source = (PROJECT_ROOT / "mcp_servers" / "project_git.py").read_text(encoding="utf-8")

    assert "shell=True" not in source
    assert "def get_git_branch() ->" in source


# --- real: the actual MCP server, on the actual repository ------------------------------------


def test_real_mcp_server_returns_this_repositorys_branch():
    """Starts mcp_servers.project_git over stdio through the existing MCP
    client, and compares with what git itself says right now."""
    expected = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"], capture_output=True, text=True, check=True
    ).stdout.strip()
    tool = asyncio.run(read_git_branch())

    assert tool.ok, tool.error
    if expected:
        assert tool.result["branch"] == expected
    else:
        assert tool.result["detached"] is True


def test_the_git_server_is_registered_with_the_client_not_the_agent():
    assert "project-git" in mcp_client.SERVERS
    names = [spec.name for spec in mcp_registry.REGISTERED_SERVERS]
    assert names == ["japanese-data", "processing", "storage", "jlpt-vocab"]


# --- real: README and docs/ in the index -----------------------------------------------------


def test_docs_are_part_of_the_corpus_with_their_own_label():
    assert CorpusRoot("docs", "docs", (".md",)) in CORPUS

    documents = load_documents((CorpusRoot("project", "README.md"), CorpusRoot("docs", "docs", (".md",))))
    files = {document.file: document.source for document in documents}

    assert files["README.md"] == "project"
    assert files["docs/architecture.md"] == "docs"
    assert files["docs/api.md"] == "docs"


def test_an_env_file_is_never_read(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# Guide\n\nHello.", encoding="utf-8")
    (tmp_path / "docs" / ".env.md").write_text("GEMINI_API_KEY=secret", encoding="utf-8")
    documents = load_documents((CorpusRoot("docs", "docs", (".md",)),), project_root=tmp_path)

    assert [document.file for document in documents] == ["docs/guide.md"]


@pytest.fixture(scope="module")
def docs_index(tmp_path_factory):
    """README + docs/, chunked and embedded exactly as the real pipeline does,
    with the offline embedding backend so no key is needed."""
    directory = tmp_path_factory.mktemp("index")
    embeddings = EmbeddingService(backend="local")
    documents = load_documents((CorpusRoot("project", "README.md"), CorpusRoot("docs", "docs", (".md",))))
    asyncio.run(index_strategy(documents, STRUCTURAL, embeddings, directory, 1200, 150))

    return RAGRetriever(strategy=STRUCTURAL, index_dir=directory, embeddings=embeddings)


def test_readme_sections_are_retrieved(docs_index):
    found = asyncio.run(docs_index.retrieve("Local RAG: the same pipeline, either model provider gemini ollama", top_k=3))

    assert any(item.file == "README.md" for item in found.chunks)


def test_docs_sections_are_retrieved_and_labelled(docs_index):
    found = asyncio.run(docs_index.retrieve("project-git get_git_branch developer assistant only registered", top_k=3))
    top = found.chunks[0]

    assert top.file.startswith("docs/")
    assert top.source == "docs"
    assert top.section


def test_the_api_reference_answers_where_an_endpoint_is(docs_index):
    found = asyncio.run(docs_index.retrieve("POST /dev/chat DevChatRequest message provider", top_k=3))

    assert any(item.file == "docs/api.md" for item in found.chunks)


# --- the endpoint ---------------------------------------------------------------------------


def test_the_endpoint_answers_help(monkeypatch):
    helper, *_ = assistant()
    monkeypatch.setattr(route, "assistant_for", lambda name: helper)
    body = client.post("/dev/chat", json={"message": "/help Which Git branch is currently active?"}).json()

    assert body["tools_used"][0]["result"]["branch"] == "feature/dev-assistant"
    assert body["rag_status"] == DISABLED
    assert body["command"] == "/help"


def test_the_endpoint_refuses_other_commands_and_providers():
    assert client.post("/dev/chat", json={"message": "hello"}).status_code == 400
    assert client.post("/dev/chat", json={"message": "/help", "provider": "openai"}).status_code == 422


def test_local_mode_with_ollama_down_is_503_and_never_gemini(monkeypatch):
    no_cloud(monkeypatch)

    def refuse(request):
        import httpx
        raise httpx.ConnectError("refused")

    import httpx

    monkeypatch.setattr(ollama_service, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(refuse)))
    agent = RagAgent(retriever=Retriever(), provider=OllamaProvider(), settings=OPEN_GATE)

    async def git():
        return ToolUse("get_git_branch", "project-git", True, 1, result={"branch": "main", "detached": False})

    monkeypatch.setattr(
        route, "assistant_for", lambda name: DevAssistant(OLLAMA, agent=agent, git_reader=git, settings=OPEN_GATE)
    )

    docs = client.post("/dev/chat", json={"message": "/help How does RAG work?", "provider": "ollama"})
    branch = client.post("/dev/chat", json={"message": "/help Which Git branch is currently active?", "provider": "ollama"})

    assert docs.status_code == 503
    assert docs.json()["provider"] == "ollama"
    assert "answer" not in docs.json()
    # The branch needs no model, so it is answered even with the model down.
    assert branch.status_code == 200
    assert "`main`" in branch.json()["answer"]


def test_the_older_endpoints_are_still_there():
    paths = set(app.openapi()["paths"])

    assert {"/rag/chat", "/agent/chat", "/local-chat", "/dev/chat"} <= paths
