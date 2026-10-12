"""Day 31: the developer assistant behind ``/help``.

Nothing here is new machinery. Documentation questions go through the RAG
agent of Days 22-28 - the same index, the same filter and reranker, the
same relevance gate, the same citation validation - with whichever provider
the request names (Day 28). The current Git branch comes from the
``project-git`` MCP server, called through the existing MCP client. What
this module adds is the command and the decision of which of the two a
question needs:

    /help                       what the assistant can do - no model, no tool
    /help <docs question>       RAG only
    /help <branch question>     MCP only - the index cannot know the branch
    /help <both>                MCP for the branch, RAG for the rest

The branch is never asked of the index and never invented by a model: the
sentence that states it is built from the tool's result. And the tool is
not called for a question that does not ask about the branch.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field

from app.services import llm_provider, mcp_client
from app.services.rag_agent import ENHANCED, RagAgent
from app.services.rag_response import DISABLED
from app.services.rag_settings import DEFAULT_SETTINGS, RagSettings

logger = logging.getLogger(__name__)

HELP = "/help"
GIT_TOOL = "get_git_branch"
GIT_SERVER = "project-git"

#: The relevance a retrieved chunk must reach before a documentation
#: question is answered from it. Lower than the API's 0.70 default: 0.35 is
#: the value the Day 28 and Day 29 evaluations used, where it answered the
#: in-domain questions and still refused the out-of-domain ones (those fail
#: the similarity filter long before this gate).
ANSWER_THRESHOLD = float(os.getenv("DEV_ASSISTANT_ANSWER_THRESHOLD", "0.35"))

CAPABILITIES = (
    "Developer assistant for this project. Ask with /help followed by a question.\n\n"
    "It answers from:\n"
    "- the project's documentation - README.md, docs/ (architecture, API reference, guides) "
    "and the code's own docstrings - through the RAG index, with sources;\n"
    "- the live Git state, through the MCP tool get_git_branch.\n\n"
    "Examples:\n"
    "/help Explain the project architecture.\n"
    "/help How does RAG work in this project?\n"
    "/help How is the Ollama provider integrated?\n"
    "/help Which Git branch is currently active?"
)

#: A question is about the branch when it says "branch" (or "ветка"). "git"
#: alone is not enough: "how does the project use git?" is a documentation
#: question, and calling the tool for it would be calling it for nothing.
_BRANCH = re.compile(r"\b(branch(es)?|ветк\w*|бранч\w*)\b", re.IGNORECASE)

#: Words that only frame a branch question. If nothing else is left once they
#: are removed, the question is about the branch alone and needs no RAG.
_FRAME = frozenset(
    "which what whats is are the a an current currently active checked out now right on we i am "
    "our this project repo repository git branch branches name called of in do does use using "
    "please tell me "
    "какая какой какую сейчас текущая текущий текущей активная активный активна активен ветка "
    "ветку ветке ветки у нас мы на в проект проекта репозиторий репозитория называется как "
    "название git используется открыта скажи подскажи".split()
)

#: Where a combined question splits into its parts.
_CLAUSES = re.compile(r"\?|;|\.\s|,?\s+and\s+|,?\s+also\s+|,?\s+и\s+|,?\s+а\s+также\s+", re.IGNORECASE)


class NotAHelpCommand(ValueError):
    """The message is not a /help command."""


@dataclass(frozen=True)
class HelpRequest:
    question: str
    wants_git: bool
    wants_docs: bool

    @property
    def docs_question(self) -> str:
        """The question RAG is asked: without the branch clause when the
        question is a combined one, so the index is not asked something it
        cannot know and the model is not tempted to refuse the whole thing."""
        if not self.wants_git:
            return self.question

        parts = [part.strip(" ,.") for part in _CLAUSES.split(self.question) if part and part.strip(" ,.")]
        kept = [part for part in parts if not _BRANCH.search(part)]

        return (" ".join(kept).strip() + "?") if kept else self.question


def parse(message: str) -> HelpRequest:
    """Read a /help command. Anything else is refused, not guessed at."""
    text = (message or "").strip()

    if not text.lower().startswith(HELP) or (len(text) > len(HELP) and not text[len(HELP)].isspace()):
        raise NotAHelpCommand("the developer assistant understands one command: /help [question]")

    question = text[len(HELP) :].strip()

    if not question:
        return HelpRequest("", wants_git=False, wants_docs=False)

    wants_git = bool(_BRANCH.search(question))
    rest = [word for word in re.findall(r"[\w-]+", question.lower()) if word not in _FRAME]

    return HelpRequest(question, wants_git=wants_git, wants_docs=not wants_git or bool(rest))


@dataclass
class ToolUse:
    """One MCP call, as it happened."""

    name: str
    server: str
    ok: bool
    ms: int
    result: dict | None = None
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "server": self.server,
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
            "ms": self.ms,
        }


async def read_git_branch(call=None) -> ToolUse:
    """Ask the project-git MCP server for the current branch.

    A tool that ran and failed, and a server that could not be started, both
    come back as ``ok=False`` with the reason - the assistant says it could
    not tell, it does not guess.
    """
    call = call or mcp_client.call_server_tool
    started = time.perf_counter()

    try:
        result = await call(GIT_TOOL, {}, mcp_client.project_git_server_parameters())
    except mcp_client.McpConnectionError as error:
        return ToolUse(GIT_TOOL, GIT_SERVER, False, _ms(started), error=str(error))

    if not result.ok:
        return ToolUse(GIT_TOOL, GIT_SERVER, False, _ms(started), error=result.error)

    data = result.data if isinstance(result.data, dict) else {"raw": result.data}

    return ToolUse(GIT_TOOL, GIT_SERVER, True, _ms(started), result=data)


def describe_branch(tool: ToolUse) -> str:
    """The sentence about the branch - built from the tool's result, never
    written by a model."""
    if not tool.ok:
        return f"I could not determine the current Git branch: {tool.error}"

    data = tool.result or {}

    if data.get("detached"):
        return (
            f"The repository is in detached HEAD state at commit {data.get('commit')} - "
            "no branch is checked out (read live from Git through the MCP tool get_git_branch)."
        )

    return f"The current Git branch is `{data.get('branch')}` (read live from Git through the MCP tool get_git_branch)."


def _ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))


@dataclass
class DevAnswer:
    question: str
    answer: str
    provider: str
    model: str | None = None
    rag_status: str = DISABLED
    confidence: str = ""
    sources: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    tools_used: list[dict] = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    best_relevance: float = 0.0
    #: How the documentation was searched - None when it was not.
    retrieval: dict | None = None

    def as_dict(self) -> dict:
        return {
            "command": HELP,
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "citations": self.citations,
            "rag_status": self.rag_status,
            "confidence": self.confidence,
            "provider": self.provider,
            "model": self.model,
            "tools_used": self.tools_used,
            "timing": self.timing,
            "best_relevance": round(self.best_relevance, 4),
            "retrieval": self.retrieval,
        }


def retrieval_report(answer) -> dict | None:
    """What the search did for one documentation question - enough to tell
    "the index has nothing" from "the search query missed" without a log.
    Numbers and the search query only; no chunk text."""
    found, cited = answer.enhanced, answer.cited

    if found is None:
        return None

    return {
        "rewrite_used": found.query.used,
        "search_query": found.query.query,
        "retrieved": len(found.retrieval.chunks),
        "kept": len(found.filtered.kept),
        "best_similarity": round(found.best_similarity, 4),
        "similarity_threshold": found.settings.similarity_threshold,
        "retried_with_question": found.retried_with_question,
        "rewrite_best_similarity": (
            None if found.rewrite_best_similarity is None else round(found.rewrite_best_similarity, 4)
        ),
        "answer_threshold": found.settings.answer_threshold,
        "rejected_citations": len(cited.rejected_citations) if cited else 0,
    }


class DevAssistant:
    """Answers /help, from the documentation, the live Git state, or both."""

    def __init__(
        self,
        provider_name: str = llm_provider.GEMINI,
        agent: RagAgent | None = None,
        git_reader=read_git_branch,
        settings: RagSettings | None = None,
    ) -> None:
        # Refused here, not defaulted: the same rule as /rag/chat (Day 28).
        llm_provider.provider_for(provider_name)
        self._provider_name = provider_name
        self._agent = agent
        self._git_reader = git_reader
        # Day 31, after the first real run: with Ollama writing the search
        # query, the two broad questions kept nothing at the 0.70 filter while
        # Gemini's rewrite of the same questions found docs/architecture.md.
        # Only the rewrite depends on the provider, so a rewrite that keeps
        # nothing is retried with the question itself - at the same 0.70.
        self._settings = settings or DEFAULT_SETTINGS.with_overrides(
            answer_threshold=ANSWER_THRESHOLD, retry_with_question=True
        )

    def _rag_agent(self) -> RagAgent:
        if self._agent is None:
            self._agent = RagAgent(
                provider=llm_provider.provider_for(self._provider_name), settings=self._settings
            )

        return self._agent

    async def ask(self, message: str) -> DevAnswer:
        """Raises NotAHelpCommand for anything but /help, and
        ProviderUnavailable when the named model cannot answer - never an
        answer from a different model."""
        started = time.perf_counter()
        request = parse(message)

        if not request.question:
            return DevAnswer(
                question="",
                answer=CAPABILITIES,
                provider=self._provider_name,
                timing={"total_ms": _ms(started)},
            )

        parts: list[str] = []
        tools: list[ToolUse] = []
        result = DevAnswer(question=request.question, answer="", provider=self._provider_name)

        if request.wants_git:
            tool = await self._git_reader()
            tools.append(tool)
            parts.append(describe_branch(tool))

        if request.wants_docs:
            agent = self._rag_agent()
            answer = await agent.ask(request.docs_question, mode=ENHANCED, settings=self._settings)
            cited = answer.cited
            parts.append(answer.answer)
            result.model = agent.model_name
            result.rag_status = cited.rag_status if cited else ""
            result.confidence = cited.confidence if cited else ""
            result.best_relevance = cited.best_relevance if cited else 0.0
            result.sources = [source.as_dict() for source in (cited.sources if cited else ())]
            result.citations = [citation.as_dict() for citation in (cited.citations if cited else ())]
            result.retrieval = retrieval_report(answer)
            result.timing.update(
                retrieval_ms=int(round(answer.retrieval_seconds * 1000)),
                generation_ms=int(round(answer.llm_seconds * 1000)),
            )

        if tools:
            result.timing["tools_ms"] = sum(tool.ms for tool in tools)

        result.answer = "\n\n".join(part for part in parts if part)
        result.tools_used = [tool.as_dict() for tool in tools]
        result.timing["total_ms"] = _ms(started)
        logger.info(
            "[DevAssistant] provider=%s git=%s docs=%s tools=%s rag_status=%s sources=%d total=%dms",
            self._provider_name,
            request.wants_git,
            request.wants_docs,
            ",".join(f"{tool.name}:{'ok' if tool.ok else 'failed'}" for tool in tools) or "-",
            result.rag_status,
            len(result.sources),
            result.timing["total_ms"],
        )

        return result
