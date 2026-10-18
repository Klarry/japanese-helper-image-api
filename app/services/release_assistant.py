"""Day 35: the AI Release Assistant - prepare a release *draft*, publish nothing.

    "Prepare a release draft. Analyze the current Git changes, review potential
    issues, compare the changes with the project documentation, and generate a
    changelog, release notes and a release report."

This module is the pipeline for that request. It is a conductor, not a second
platform: every capability it uses already existed.

    Git facts ................ the project-git MCP server (extended today with status,
                               changes, diff, commits - read-only)
    Files read / written ..... the project-files MCP server (Day 34), pointed at the
                               repository being released (read) and at the assistant's
                               own release folder (write)
    The review ............... CodeReviewer (Day 32) - its prompt, its RAG context from
                               the Day 23 retriever, its evidence validation
    The model ................ the existing Ollama / Gemini providers; a provider that is
                               unavailable is reported, never replaced by the other one
    Checks ................... only those a person listed in data/release/release_config.json,
                               only when asked, without a shell (app/services/release_checks.py)

What it never does: commit, tag, stash, reset, clean, push, merge, build a release
artifact or publish - there is no code path for any of them. A dirty working tree
is reported, not touched, and uncommitted changes are not assumed to belong to the
release. Repository text (diffs, commit subjects, documentation) is untrusted data
in the prompts. Nothing is claimed that was not observed: a check is "passed" only
with a real exit status 0, a finding only with evidence from the diff, a changelog
entry only with a changed line or a commit subject behind it.

The steps, as the run record numbers them:

    1 inspect   2 collect   3 review   4 validate (checks)   5 draft + artifacts
    (and the safe writing of the artifacts, the run record, the readiness verdict)
"""

import asyncio
import json
import logging
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.services import mcp_client, release_artifacts, release_checks, release_draft, release_safety
from app.services.code_review import (
    CATEGORIES as REVIEW_CATEGORIES,
    MAX_FILES,
    CodeReviewer,
    InvalidReviewOutput,
    PullRequest,
    ReviewRequestError,
    ReviewTooLarge,
    review_provider,
)
from app.services.llm_provider import GEMINI, OLLAMA, PROVIDERS, ProviderUnavailable
from app.services.mcp_client import McpConnectionError, McpToolCallResult
from app.services.project_files import PROJECT_ROOT
from app.services.release_config import (
    CheckSpec,
    ReleaseConfig,
    ReleaseConfigError,
    RepositorySpec,
    load_config,
    output_dir_for,
)

logger = logging.getLogger(__name__)

COMPLETED = "completed"
PARTIAL = "partial"
FAILED = "failed"
NO_CHANGES = "no_changes"

READY = "ready"
READY_WITH_WARNINGS = "ready_with_warnings"
NOT_VERIFIED = "not_verified"
BLOCKED = "blocked"
NOTHING_TO_RELEASE = "nothing_to_release"

GIT = "git"
TARGET = "target"
OUTPUT = "output"
SERVER_NAMES = {GIT: "project-git", TARGET: "project-files", OUTPUT: "project-files"}

#: Review and draft budgets. Each batch is one model call on the Day 32 reviewer.
BATCH_CHARS = 12_000
MAX_BATCHES = 3
DIFF_CHARS = 150_000
COMMIT_LIMIT = 30
DOC_QUERIES = 16
DOC_FLAGS = 8
CONCURRENCY = 4
MAX_ID_TRIES = 5

_PRIORITY_CODE = {".py", ".kt", ".java", ".ts", ".tsx", ".js", ".swift", ".go", ".rs"}
_PRIORITY_CONFIG = {".gradle", ".kts", ".json", ".xml", ".yaml", ".yml", ".toml", ".properties", ".sh", ".ini", ".cfg"}
_PRIORITY_DOCS = {".md", ".txt", ".rst"}
_STEM_STOPLIST = frozenset({"main", "index", "utils", "config", "models", "init", "__init__", "common", "types", "constants", "app", "test", "tests"})
_ERROR_KIND = re.compile(r"\b([a-z]+(?:_[a-z]+)+):\s")
_SDK_PREFIX = re.compile(r"^Error executing tool \S+:\s*")


class ReleaseError(Exception):
    """The release cannot be prepared as asked: the request or the environment
    is wrong (unknown repository or base reference, a server that will not
    start). Nothing is written for these."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message


class ToolRefused(Exception):
    """An MCP tool ran and refused or failed; ``kind`` is the first word of its message."""

    def __init__(self, tool: str, message: str) -> None:
        message = _SDK_PREFIX.sub("", message or "")  # "Error executing tool get_git_status: " comes from the MCP SDK
        match = _ERROR_KIND.search(message)
        self.tool = tool
        self.kind = match.group(1) if match else "tool_error"
        self.message = message[match.end():].strip() if match else message
        super().__init__(f"{tool}: {self.kind}: {self.message}")


@dataclass
class ReleaseRequest:
    """What was asked. ``base_ref`` is the start of the release range; without it nothing is guessed."""

    base_ref: str = ""
    provider: str = OLLAMA
    repository: str = "backend"
    repository_path: str = ""  # CLI only: a folder instead of a configured name
    include_uncommitted: bool = False
    run_checks: bool = False
    checks: tuple[str, ...] = ()
    output_dir: str = ""
    max_review_batches: int | None = None

    def as_dict(self) -> dict:
        return {
            "base_ref": self.base_ref.strip(),
            "provider": self.provider,
            "repository": self.repository_path and Path(self.repository_path).name or self.repository,
            "include_uncommitted": self.include_uncommitted,
            "run_checks": self.run_checks,
            "checks": list(self.checks),
            "output_dir": self.output_dir,
            "max_review_batches": self.max_review_batches,
        }


def _utc(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _short(arguments: dict) -> dict:
    """Arguments as the run record keeps them: a long text (an artifact) becomes its size."""
    return {key: f"<{len(value)} characters>" if isinstance(value, str) and len(value) > 120 else value for key, value in arguments.items()}


# --- the MCP calls ---------------------------------------------------------------------------------


class Tools:
    """Every MCP call the run makes, in one place: which server, how long it took,
    whether it worked. ``call`` is mcp_client.call_server_tool unless a test passes another."""

    def __init__(self, repository: Path, project: Path, output_dir: str, call=None, list_tools=None) -> None:
        self._call = call or mcp_client.call_server_tool
        self._list = list_tools or mcp_client.list_server_tools
        self._parameters = {
            GIT: lambda: mcp_client.project_git_server_parameters(repository),
            TARGET: lambda: mcp_client.project_files_server_parameters(repository),
            OUTPUT: lambda: mcp_client.project_files_server_parameters(project, output_dir),
        }
        self._gate = asyncio.Semaphore(CONCURRENCY)
        self.log: list[dict] = []

    async def call(self, server: str, tool: str, arguments: dict | None = None) -> dict:
        """The tool's structured answer. Raises ToolRefused when the tool refused,
        ReleaseError("mcp_unavailable") when the server could not be reached."""
        arguments = arguments or {}
        entry = {"server": SERVER_NAMES[server], "role": server, "tool": tool, "arguments": _short(arguments)}

        async with self._gate:
            started = time.perf_counter()

            try:
                result: McpToolCallResult = await self._call(tool, arguments, self._parameters[server]())
            except McpConnectionError as error:
                self.log.append({**entry, "ok": False, "ms": int((time.perf_counter() - started) * 1000), "error": str(error)[:200]})
                raise ReleaseError("mcp_unavailable", f"the {SERVER_NAMES[server]} MCP server could not be used: {error}") from error

        entry["ms"] = int((time.perf_counter() - started) * 1000)

        if not result.ok:
            self.log.append({**entry, "ok": False, "error": (result.error or "")[:200]})
            raise ToolRefused(tool, result.error or "the tool reported an error")

        self.log.append({**entry, "ok": True})

        return result.data if isinstance(result.data, dict) else {}

    async def tool_names(self, server: str) -> list[str]:
        try:
            report = await self._list(self._parameters[server]())
        except McpConnectionError as error:
            raise ReleaseError("mcp_unavailable", f"the {SERVER_NAMES[server]} MCP server could not be used: {error}") from error

        return sorted(tool.name for tool in report.tools)


# --- small pure helpers ----------------------------------------------------------------------------


def is_test_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    lowered = path.lower()

    return (
        lowered.startswith(("tests/", "test/"))
        or "/tests/" in lowered
        or "/test/" in lowered
        or "/androidtest/" in lowered
        or name.startswith("test_")
        or name.endswith(("_test.py", "Test.kt", "Tests.kt", "Test.java"))
    )


def _extension(path: str) -> str:
    name = path.rsplit("/", 1)[-1]

    return name[name.rfind("."):].lower() if "." in name else ""


def review_priority(path: str) -> int:
    """Source first, then build and configuration, tests, documentation, the rest."""
    if is_test_path(path):
        return 2

    extension = _extension(path)

    if extension in _PRIORITY_CODE:
        return 0

    if extension in _PRIORITY_CONFIG:
        return 1

    return 3 if extension in _PRIORITY_DOCS else 4


def _glob_regex(pattern: str) -> re.Pattern:
    out, index = [], 0

    while index < len(pattern):
        if pattern.startswith("**", index):
            out.append(".*")
            index += 2
        elif pattern[index] == "*":
            out.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            out.append("[^/]")
            index += 1
        else:
            out.append(re.escape(pattern[index]))
            index += 1

    return re.compile("^" + "".join(out) + "$")


def matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(_glob_regex(pattern).match(path) for pattern in patterns)


def is_user_facing(path: str, repository: RepositorySpec) -> bool:
    return not is_test_path(path) and matches_any(path, repository.user_facing)


def make_batches(files: list[dict], batch_chars: int = BATCH_CHARS, max_batches: int = MAX_BATCHES) -> tuple[list[list[dict]], list[dict]]:
    """Group files with a diff into review batches of about ``batch_chars`` each, most
    important first. A file bigger than a batch gets one to itself. Files that do not
    fit into ``max_batches`` batches (or the reviewer's file limit) are returned apart."""
    ordered = sorted(files, key=lambda item: (review_priority(item["path"]), item["path"]))
    batches: list[list[dict]] = []
    used = 0

    for item in ordered:
        size = len(item["diff"])

        if not batches or used + size > batch_chars or len(batches[-1]) >= MAX_FILES:
            batches.append([])
            used = 0

        batches[-1].append(item)
        used += size

    return batches[:max_batches], [item for batch in batches[max_batches:] for item in batch]


def stems(path: str) -> list[str]:
    """Names to look for in documentation: the path, and the file's own name when it is distinctive."""
    name = path.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    terms = [path]

    if len(stem) >= 6 and stem.lower() not in _STEM_STOPLIST:
        terms.append(stem)

    return terms


def android_facts(text: str) -> dict:
    """Package and version settings from a Gradle build script (read, never changed)."""
    found = {}
    patterns = {
        "namespace": r'namespace\s*=\s*"([^"]+)"',
        "application_id": r'applicationId\s*=\s*"([^"]+)"',
        "version_code": r"versionCode\s*=\s*(\d+)",
        "version_name": r'versionName\s*=\s*"([^"]+)"',
        "compile_sdk": r"compileSdk\s*(?:=\s*|\{[^}]*?)(\d+)",
        "min_sdk": r"minSdk\s*=\s*(\d+)",
        "target_sdk": r"targetSdk\s*=\s*(\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)

        if match:
            found[key] = match.group(1)

    return found


# --- the run ---------------------------------------------------------------------------------------


@dataclass
class Run:
    """Everything a release preparation observed and produced. JSON-ready; holds no diffs and no file contents."""

    run_id: str
    request: dict
    started_at: str
    status: str = FAILED
    finished_at: str = ""
    seconds: float = 0.0
    repository: dict = field(default_factory=dict)
    git_status_at_start: dict = field(default_factory=dict)
    scope: dict = field(default_factory=dict)
    files: list[dict] = field(default_factory=list)
    other_changes: dict = field(default_factory=dict)
    commits: dict = field(default_factory=dict)
    inspection: dict = field(default_factory=dict)
    documentation: dict = field(default_factory=dict)
    review: dict = field(default_factory=dict)
    findings: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    tree_after_checks: dict = field(default_factory=dict)
    draft: dict = field(default_factory=dict)
    readiness: dict = field(default_factory=dict)
    artifacts: list[dict] = field(default_factory=list)
    output_dir: str = ""
    run_dir: str = ""
    limitations: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    llm: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def error(self, stage: str, kind: str, message: str) -> None:
        self.errors.append({"stage": stage, "kind": kind, "message": message[:400]})


def _tree_signature(status: dict) -> dict:
    return {
        "staged": sorted((item["path"], item["status"]) for item in status.get("staged", [])),
        "unstaged": sorted((item["path"], item["status"]) for item in status.get("unstaged", [])),
        "untracked": sorted(status.get("untracked", [])),
    }


def _new_id(now: datetime) -> str:
    return now.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


class ReleaseAssistant:
    """Prepares a release draft for one repository. See the module docstring.

    Everything that talks to something is replaceable, so the pipeline can be
    tested without a model, a Git server or a build: ``call`` / ``list_tools``
    (the MCP client), ``reviewer`` (the Day 32 CodeReviewer), ``draft_provider``,
    ``check_runner``, ``clock`` and ``id_factory``.
    """

    def __init__(
        self,
        provider_name: str = OLLAMA,
        *,
        config: ReleaseConfig | None = None,
        project_root: Path = PROJECT_ROOT,
        call=None,
        list_tools=None,
        reviewer=None,
        draft_provider=None,
        check_runner: Callable[[CheckSpec, Path], release_checks.CheckResult] | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[datetime], str] | None = None,
    ) -> None:
        self.provider_name = (provider_name or OLLAMA).strip().lower()
        self._config = config
        self._project_root = project_root
        self._call = call
        self._list_tools = list_tools
        self._reviewer = reviewer
        self._draft_provider = draft_provider
        self._check_runner = check_runner or release_checks.run_check
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._id_factory = id_factory or _new_id
        self._diffs: dict[str, str] = {}

    # --- entry point ---------------------------------------------------------------------------------

    async def prepare(self, request: ReleaseRequest) -> Run:
        """Raises ReleaseError for a request that cannot be served (nothing is written then).
        Everything after the request is accepted ends in a Run, including a failed one."""
        if self.provider_name not in PROVIDERS:
            raise ReleaseError("invalid_provider", f"provider must be one of {', '.join(PROVIDERS)}, got {request.provider!r}")

        try:
            config = self._config or load_config()
            spec = self._repository(config, request)
            output_dir = output_dir_for(config, request.output_dir)
        except ReleaseConfigError as error:
            raise ReleaseError(error.kind, error.message) from error

        if not spec.path.is_dir():
            raise ReleaseError("repository_missing", f"the folder of repository {spec.key!r} does not exist on this machine")

        applicable = config.checks_for(spec.key)
        wanted = self._selected_checks(request, applicable)
        started = time.perf_counter()
        now = self._clock()
        tools = Tools(spec.path, self._project_root, output_dir, self._call, self._list_tools)
        run = Run(
            run_id=self._id_factory(now),
            request={**request.as_dict(), "provider": self.provider_name, "output_dir": output_dir},
            started_at=_utc(now),
            output_dir=output_dir,
        )
        self._diffs = {}
        review_failed = False
        draft_failed = False

        await self._inspect(run, tools, spec, applicable)
        await self._collect(run, tools, spec, request)

        if run.files and any(item["analysis"] == "analyzed" for item in run.files):
            await self._documentation(run, tools, spec)

            try:
                await self._review(run, spec)
            except ProviderUnavailable as error:
                run.error("review", "provider_unavailable", f"{error.provider}: {error.message} ({error.reason}) - no other provider was asked")
                review_failed = not run.review.get("files_reviewed")
                draft_failed = True
                run.draft = {"status": "skipped", "reason": f"{error.provider} became unavailable during the review; no other provider was asked", "changelog": [], "release_notes": [], "rejected": [], "adjusted": []}

                if review_failed:
                    await self._run_checks(run, tools, spec, applicable, set(), "the run stopped before the checks: the model was unavailable")

            if not review_failed:
                await self._run_checks(run, tools, spec, applicable, wanted, "checks are opt-in: they run only when asked for")

                if not draft_failed:
                    draft_failed = not await self._draft(run, spec)
        else:
            await self._run_checks(run, tools, spec, applicable, set(), "there are no changes to validate")
            run.draft = {"status": "skipped", "reason": "there were no analyzable text changes", "changelog": [], "release_notes": [], "rejected": [], "adjusted": []}

        self._conclude(run, spec, review_failed, draft_failed)
        await self._write(run, tools, started)
        run.tool_calls = tools.log
        run.finished_at = run.finished_at or _utc(self._clock())
        run.seconds = run.seconds or round(time.perf_counter() - started, 1)

        logger.info(
            "[ReleaseAssistant] run=%s repo=%s status=%s readiness=%s files=%d findings=%d checks=%d provider=%s",
            run.run_id, spec.key, run.status, run.readiness.get("verdict"), len(run.files), len(run.findings), len(run.checks), self.provider_name,
        )

        return run

    # --- the request ---------------------------------------------------------------------------------

    @staticmethod
    def _repository(config: ReleaseConfig, request: ReleaseRequest) -> RepositorySpec:
        if request.repository_path:
            path = Path(request.repository_path).expanduser().resolve()
            return RepositorySpec(key="custom", path=path, title=path.name, documentation=("README.md", "docs/*.md"))

        return config.repository(request.repository)

    @staticmethod
    def _selected_checks(request: ReleaseRequest, applicable: list[CheckSpec]) -> set[str]:
        known = {check.id for check in applicable}

        if request.checks and not request.run_checks:
            raise ReleaseError("invalid_request", "checks were named but run_checks is off; nothing runs unless it is asked for")

        unknown = [name for name in request.checks if name not in known]

        if unknown:
            raise ReleaseError("unknown_check", f"{', '.join(unknown)} is not a configured check for this repository (configured: {', '.join(sorted(known)) or 'none'})")

        if not request.run_checks:
            return set()

        return set(request.checks) if request.checks else known

    # --- step 1: inspect -----------------------------------------------------------------------------

    async def _inspect(self, run: Run, tools: Tools, spec: RepositorySpec, applicable: list[CheckSpec]) -> None:
        """Step 1: what is here. The Git state of the repository, the MCP tools that exist, the
        Gradle version settings, release documents already in the repository, the commands the
        project can be checked with. The Git status is the one answer a run cannot do without."""
        jobs = {
            "status": tools.call(GIT, "get_git_status"),
            "git_tools": tools.tool_names(GIT),
            "file_tools": tools.tool_names(OUTPUT),
            "changelog": tools.call(TARGET, "search_project_files", {"pattern": "CHANGELOG"}),
            "release": tools.call(TARGET, "search_project_files", {"pattern": "RELEASE"}),
        }

        for path in spec.version_files:
            jobs[f"version:{path}"] = tools.call(TARGET, "read_project_file", {"path": path})

        outcomes = dict(zip(jobs, await asyncio.gather(*jobs.values(), return_exceptions=True)))
        status = outcomes["status"]

        if isinstance(status, ToolRefused):
            raise ReleaseError(status.kind, status.message)

        if isinstance(status, BaseException):
            raise status

        run.repository = {
            "key": spec.key,
            "name": spec.path.name,
            "title": spec.title,
            "branch": status["branch"],
            "detached": status["detached"],
            "head": status["head"],
            "head_short": status["head_short"],
            "upstream": status.get("upstream"),
        }
        run.git_status_at_start = {
            "clean": status["clean"],
            "staged": status["staged"],
            "unstaged": status["unstaged"],
            "untracked": status["untracked"],
            "conflicted": status["conflicted"],
            "counts": status["counts"],
            "truncated": status["truncated"],
        }
        gaps = []
        versions = []

        for path in spec.version_files:
            outcome = outcomes[f"version:{path}"]

            if isinstance(outcome, dict):
                gradle = path.endswith((".kts", ".gradle"))
                versions.append({"path": path, "sha256": outcome.get("sha256", ""), "facts": android_facts(outcome.get("content", "")) if gradle else {}})
            else:
                gaps.append(f"{path}: {outcome}")

        existing = set()

        for key in ("changelog", "release"):
            outcome = outcomes[key]

            if isinstance(outcome, dict):
                existing |= {name for name in outcome.get("files", []) if re.search(r"(?i)changelog|release[-_ ]?notes|releases?\.md", name)}
            else:
                gaps.append(f"search for release documents: {outcome}")

        names = {}

        for key in ("git_tools", "file_tools"):
            if isinstance(outcomes[key], list):
                names[key] = outcomes[key]
            else:
                names[key] = []
                gaps.append(f"{key}: {outcomes[key]}")

        run.inspection = {
            "repository": {"key": spec.key, "name": spec.path.name, "branch": status["branch"], "head": status["head_short"], "clean": status["clean"]},
            "mcp_tools": {"project-git": names["git_tools"], "project-files": names["file_tools"]},
            "version_files": versions,
            "existing_release_documents": sorted(existing),
            "commands": [
                {
                    "id": check.id,
                    "command": release_checks.display_command(check),
                    "documented_in": check.documented_in,
                    "available": all((spec.path / marker).exists() for marker in check.requires),
                }
                for check in applicable
            ],
            "assistant_project": self._project_facts(),
            "gaps": gaps,
        }

    def _project_facts(self) -> dict:
        """What this project already has that the release uses: the RAG index, the review, the invariants (read-only)."""
        facts: dict = {"name": self._project_root.name}
        indexes = []

        for metadata in sorted((self._project_root / "data" / "index").glob("*/metadata.json")):
            try:
                header = json.loads(metadata.read_text(encoding="utf-8"))
                indexes.append({"strategy": header.get("strategy", metadata.parent.name), "chunks": header.get("count"), "embedding_model": header.get("embedding_model"), "created_at": header.get("created_at")})
            except (OSError, json.JSONDecodeError):
                indexes.append({"strategy": metadata.parent.name, "error": "metadata unreadable"})

        facts["rag_indexes"] = indexes
        facts["code_review"] = {"module": "app.services.code_review", "categories": list(REVIEW_CATEGORIES), "max_files_per_review": MAX_FILES}
        invariants = self._project_root / "data" / "agent_invariants.json"

        try:
            facts["invariants"] = [{"id": item.get("id"), "rule": item.get("rule")} for item in json.loads(invariants.read_text(encoding="utf-8")).get("invariants", [])]
        except (OSError, json.JSONDecodeError):
            facts["invariants"] = []

        return facts

    # --- step 2: collect -----------------------------------------------------------------------------

    async def _collect(self, run: Run, tools: Tools, spec: RepositorySpec, request: ReleaseRequest) -> None:
        """Step 2: which changes, kept apart by where they are. Committed changes are measured
        against the base the caller named; uncommitted and untracked files are listed, and analyzed
        only when asked (with a base) or when there is no base to measure against at all."""
        base = request.base_ref.strip()
        changes, commits = await asyncio.gather(
            tools.call(GIT, "get_git_changes", {"base_ref": base}),
            tools.call(GIT, "get_recent_commits", {"base_ref": base, "limit": COMMIT_LIMIT}),
            return_exceptions=True,
        )

        for outcome in (changes, commits):
            if isinstance(outcome, ToolRefused):
                kind = "invalid_base_ref" if outcome.kind in ("invalid_ref", "unknown_ref", "no_merge_base") else outcome.kind
                raise ReleaseError(kind, outcome.message)

            if isinstance(outcome, BaseException):
                raise outcome

        committed = {item["path"]: item for item in changes["committed"]}
        uncommitted = {item["path"]: item for item in changes["uncommitted"]}
        untracked = changes["untracked"]
        head = run.repository["head_short"]
        include = bool(base) and request.include_uncommitted

        if base:
            diff_scope = "combined" if include else "committed"
            mode = "committed_range_and_uncommitted" if include else "committed_range"
            relevant = {**committed, **uncommitted} if include else committed
            label = changes["base"]["short"]
            described = (
                f"Committed changes in {base}...HEAD (merge base {changes['base']['merge_base_short']}, HEAD {head}): "
                f"{len(committed)} files." + (f" Uncommitted changes ({len(uncommitted)} files) are included because they were asked for." if include else "")
            )
        else:
            diff_scope = "uncommitted"
            mode = "working_tree"
            relevant = uncommitted
            label = "HEAD"
            described = (
                f"No base reference was given, so no release range was assumed. Analyzed the uncommitted changes of the working "
                f"tree (index and working tree against HEAD {head}): {len(uncommitted)} files. Commits were not analyzed."
            )

        not_analyzed = [] if (include or not base) else sorted(uncommitted)
        run.scope = {
            "mode": mode,
            "base_ref": base,
            "base": changes["base"],
            "diff_scope": diff_scope,
            "head": run.repository["head"],
            "head_short": head,
            "base_label": label,
            "description": described,
            "counts": changes["counts"],
        }
        run.other_changes = {
            "uncommitted_not_analyzed": not_analyzed,
            "untracked": untracked,
            "truncated": changes["truncated"],
        }
        run.commits = {
            "range": commits["range"],
            "total": commits["total"],
            "truncated": commits["truncated"],
            "items": [{"short": item["short"], "date": item["date"], "subject": item["subject"]} for item in commits["commits"]],
            "used_as_evidence": bool(base),
        }

        if not relevant:
            run.scope["description"] += " There is nothing to analyze."
            return

        diff = await tools.call(GIT, "get_git_diff", {"base_ref": base, "scope": diff_scope, "max_chars": DIFF_CHARS})

        for item in sorted(diff["files"], key=lambda entry: entry["path"]):
            path = item["path"]
            reason = item["withheld"] or release_safety.classify_path(path)
            text = item.get("diff", "")

            if not reason and text and release_safety.contains_secret(text):
                reason = release_safety.POSSIBLE_SECRET

            if reason:
                text = ""

            analyzed_uncommitted = include or not base
            where = "uncommitted" if path not in committed else "committed+uncommitted" if path in uncommitted and analyzed_uncommitted else "committed"

            if text:
                self._diffs[path] = text

            run.files.append(
                {
                    "path": path,
                    "status": item["status"],
                    "additions": item["additions"],
                    "deletions": item["deletions"],
                    "origin": where,
                    "also_edited_uncommitted": path in committed and path in uncommitted and not analyzed_uncommitted,
                    "user_facing": is_user_facing(path, spec),
                    "analysis": "withheld" if reason else "analyzed",
                    "withheld": reason,
                    "review": "withheld" if reason else "not_reviewed",
                }
            )

    # --- the documentation check ---------------------------------------------------------------------

    async def _documentation(self, run: Run, tools: Tools, spec: RepositorySpec) -> None:
        """Compare the change set with the project documentation - by search, not by opinion: which
        documents mention a file that changed, while the document itself did not. That is a question
        for a person (is the text still true?), not a defect, and is reported as one."""
        if not spec.documentation:
            run.documentation = {"documents": [], "terms": 0, "flags": [], "total_flagged": 0, "note": "no documentation is configured for this repository"}
            return

        changed_docs = sorted(item["path"] for item in run.files if matches_any(item["path"], spec.documentation))
        analyzed = [item for item in run.files if item["analysis"] == "analyzed" and item["path"] not in changed_docs]
        queries: list[str] = []

        for item in sorted(analyzed, key=lambda entry: (review_priority(entry["path"]), entry["path"])):
            for term in stems(item["path"]):
                if term not in queries:
                    queries.append(term)

        queries = queries[:DOC_QUERIES]
        answers = await asyncio.gather(
            *(tools.call(TARGET, "search_in_files", {"query": term, "file_pattern": "*.md"}) for term in queries),
            return_exceptions=True,
        )
        paths = {item["path"] for item in analyzed}
        mentions: dict[str, dict] = {}
        failed = 0

        for term, answer in zip(queries, answers):
            if not isinstance(answer, dict):
                failed += 1
                continue

            for match in answer.get("matches", []):
                document = match["path"]

                if not matches_any(document, spec.documentation) or document in changed_docs:
                    continue

                entry = mentions.setdefault(document, {"document": document, "line": match["line"], "excerpt": match["excerpt"], "mentions": [], "changed_files": []})
                entry["mentions"].append(term)

                for path in paths:
                    if term == path or term == path.rsplit("/", 1)[-1].rsplit(".", 1)[0]:
                        if path not in entry["changed_files"]:
                            entry["changed_files"].append(path)

        flags = sorted(mentions.values(), key=lambda entry: (-len(entry["changed_files"]), entry["document"]))
        shown = [
            {**entry, "mentions": sorted(set(entry["mentions"])), "changed_files": sorted(entry["changed_files"]), "excerpt": release_safety.redact(entry["excerpt"])[:200]}
            for entry in flags[:DOC_FLAGS]
        ]
        run.documentation = {
            "documents": list(spec.documentation),
            "changed_documents": changed_docs,
            "terms": len(queries),
            "search_failures": failed,
            "flags": shown,
            "total_flagged": len(flags),
        }

    # --- step 3: review ------------------------------------------------------------------------------

    async def _review(self, run: Run, spec: RepositorySpec) -> None:
        """Step 3: the Day 32 reviewer, on batches of the analyzed diffs. Its prompt, its RAG
        context and its validation of findings against the diff are used as they are."""
        analyzed = [{**item, "diff": self._diffs[item["path"]]} for item in run.files if item["analysis"] == "analyzed" and item["path"] in self._diffs]
        limit = max(1, int(run.request.get("max_review_batches") or MAX_BATCHES))
        batches, skipped = make_batches(analyzed, BATCH_CHARS, limit)
        reviewer = self._reviewer or CodeReviewer(self.provider_name)
        by_path = {item["path"]: item for item in run.files}
        run.review = {
            "provider": self.provider_name,
            "model": "",
            "batches": [],
            "files_reviewed": 0,
            "not_reviewed": [],
            "summaries": [],
            "architecture_notes": [],
            "recommendations": [],
            "limitations": [],
            "context_sources": [],
        }
        review = run.review
        found: list[dict] = run.findings
        sources: dict[str, dict] = {}

        def finish() -> None:
            review["files_reviewed"] = sum(1 for item in run.files if item["review"] in ("full", "partial"))
            review["context_sources"] = sorted(sources.values(), key=lambda entry: (entry["file"], entry["section"]))

            for key in ("limitations", "architecture_notes", "recommendations"):
                review[key] = list(dict.fromkeys(review[key]))

        if not analyzed:
            review["limitations"].append("There were no text diffs to review.")

        for item in skipped:
            by_path[item["path"]]["review"] = "not_reviewed"
            review["not_reviewed"].append({"path": item["path"], "reason": f"over the review budget of {limit} batches"})

        for index, batch in enumerate(batches, start=1):
            record = {"batch": index, "files": [item["path"] for item in batch], "status": "", "findings": 0, "ms": 0, "error": ""}
            review["batches"].append(record)
            request_id = f"{run.run_id}-r{index}"
            base_branch = run.scope["base_label"]
            head_branch = run.repository["branch"] or run.repository["head_short"]
            pull = PullRequest(
                repository=f"local/{run.repository['name']}",
                number=0,
                base_branch=base_branch,
                head_branch=head_branch if run.scope["base_ref"] else "working tree",
                title=f"Release candidate: {run.repository['title']} ({run.scope['mode'].replace('_', ' ')})",
                diff="\n".join(item["diff"].rstrip("\n") + "\n" for item in batch),
                changed_files=[
                    {
                        "path": item["path"],
                        "status": {"deleted": "removed"}.get(item["status"], item["status"] if item["status"] in ("added", "modified") else "changed"),
                        "additions": item["additions"],
                        "deletions": item["deletions"],
                    }
                    for item in batch
                ],
            )
            started = time.perf_counter()

            try:
                result = await reviewer.review(pull, request_id=request_id)
            except (ReviewRequestError, ReviewTooLarge) as error:
                record.update(status="skipped", error=str(error)[:300], ms=int((time.perf_counter() - started) * 1000))
                review["not_reviewed"] += [{"path": item["path"], "reason": str(error)[:200]} for item in batch]
                continue
            except InvalidReviewOutput as error:
                record.update(status="failed", error=str(error)[:300], ms=int((time.perf_counter() - started) * 1000))
                review["not_reviewed"] += [{"path": item["path"], "reason": "the model returned no usable review"} for item in batch]
                run.error("review", "invalid_model_output", f"batch {index}: {error}")
                continue
            except ProviderUnavailable:
                record.update(status="provider_unavailable", ms=int((time.perf_counter() - started) * 1000))
                review["not_reviewed"] += [{"path": item["path"], "reason": "the model became unavailable"} for item in batch]
                finish()
                raise

            document = result.as_dict()
            review["model"] = document["model"]

            for entry in document["files"]:
                if entry["path"] in by_path:
                    by_path[entry["path"]]["review"] = entry["reviewed"]

            for chunk in document["context"].get("chunks", []):
                if chunk.get("in_prompt"):
                    sources.setdefault(chunk["id"] + ":" + chunk["file"], {"file": chunk["file"], "section": chunk.get("section", "")})

            for finding in document["findings"]:
                found.append({**finding, "source": "code_review", "validation_status": "evidence_verified", "batch": index})

            review["summaries"].append(document["summary"])
            review["architecture_notes"] += document["architecture_notes"]
            review["recommendations"] += document["recommendations"]
            review["limitations"] += document["limitations"]
            record.update(
                status="reviewed",
                request_id=request_id,
                findings=len(document["findings"]),
                proposed=document["validation"]["proposed"],
                rejected=len(document["validation"]["rejected"]),
                context_chunks=sum(1 for chunk in document["context"].get("chunks", []) if chunk.get("in_prompt")),
                context_problem=document["context"].get("problem", ""),
                prompt_tokens_estimate=document["generation"]["prompt_tokens_estimate"],
                input_tokens=document["generation"].get("input_tokens"),
                output_tokens=document["generation"].get("output_tokens"),
                ms=int((time.perf_counter() - started) * 1000),
            )

        finish()

    # --- step 4: validate readiness (checks) -----------------------------------------------------------

    async def _run_checks(self, run: Run, tools: Tools, spec: RepositorySpec, applicable: list[CheckSpec], selected: set[str], why_not: str) -> None:
        """Step 4: run the checks a person configured - only those asked for - and record what
        really happened. Afterwards Git is asked again, to see whether running them changed the tree."""
        results = []
        executed = False

        for check in applicable:
            if check.id in selected:
                executed = True
                logger.info("[ReleaseAssistant] run=%s check %s: %s", run.run_id, check.id, release_checks.display_command(check))
                results.append((await asyncio.to_thread(self._check_runner, check, spec.path)).as_dict())
            else:
                results.append(release_checks.not_run(check, "not selected for this run" if selected else why_not).as_dict())

        run.checks = results

        if not executed:
            return

        try:
            after = await tools.call(GIT, "get_git_status")
        except (ToolRefused, ReleaseError) as error:
            run.tree_after_checks = {"compared": False, "reason": str(error)[:200]}
            return

        before, now = _tree_signature(run.git_status_at_start), _tree_signature(after)
        changed = {kind: sorted(set(map(tuple, now[kind])) ^ set(map(tuple, before[kind])) if kind != "untracked" else set(now[kind]) ^ set(before[kind])) for kind in before}
        run.tree_after_checks = {"compared": True, "changed": any(changed.values()), "differences": {kind: [list(item) if isinstance(item, tuple) else item for item in items][:20] for kind, items in changed.items() if items}}

    # --- step 5: the draft -------------------------------------------------------------------------------

    async def _draft(self, run: Run, spec: RepositorySpec) -> bool:
        """Changelog and release-note entries from the model, each kept only if the change set proves
        it. False when the model could not deliver (the run is then partial)."""
        files = []

        for item in run.files:
            if item["analysis"] != "analyzed" or item["path"] not in self._diffs:
                continue

            added, removed = release_draft.changed_lines(self._diffs[item["path"]])
            files.append(
                release_draft.FileFacts(item["path"], item["status"], item["additions"], item["deletions"], item["user_facing"], added, removed)
            )

        empty = {"changelog": [], "release_notes": [], "rejected": [], "adjusted": []}

        if not files:
            run.draft = {"status": "skipped", "reason": "no analyzed file has text changes", **empty}
            return True

        subjects = [entry["subject"] for entry in run.commits["items"]] if run.commits.get("used_as_evidence") else []
        scope = f"committed changes {run.scope['base_ref']}...HEAD" if run.scope["base_ref"] else "uncommitted working-tree changes against HEAD"
        data = release_draft.DraftInput(repository=run.repository["title"], scope=scope, files=files, commits=subjects)

        try:
            provider = self._draft_provider or release_draft.release_provider(self.provider_name)
            result = await release_draft.draft(provider, data)
        except ProviderUnavailable as error:
            run.draft = {"status": "failed", "reason": f"{error.provider} unavailable ({error.reason}); no other provider was asked", **empty}
            run.error("draft", "provider_unavailable", f"{error.provider}: {error.message}")
            return False
        except release_draft.InvalidDraftOutput as error:
            run.draft = {"status": "failed", "reason": str(error), **empty}
            run.error("draft", "invalid_model_output", str(error))
            return False

        run.draft = {"status": "generated", **result.as_dict()}

        return True

    # --- conclusions -------------------------------------------------------------------------------------

    def _conclude(self, run: Run, spec: RepositorySpec, review_failed: bool, draft_failed: bool) -> None:
        """Findings from rules and from executed checks join the reviewed ones; then the status, the
        readiness verdict, the limitations and the open questions. All of it is computed from what was
        observed - no model decides whether a release is ready."""
        analyzed = [item for item in run.files if item["analysis"] == "analyzed"]
        run.findings += self._rule_findings(run, analyzed)
        rank = {name: number for number, name in enumerate(("critical", "high", "medium", "low", "info"))}
        run.findings.sort(key=lambda item: (rank.get(item["severity"], 9), item["source"], item.get("file") or "", item.get("line") or 0))

        for number, item in enumerate(run.findings, start=1):
            item["id"] = f"F{number}"

        batches = run.review.get("batches", [])
        broken = any(batch["status"] in ("failed", "provider_unavailable") for batch in batches)

        if review_failed:
            run.status = FAILED
        elif not analyzed:
            run.status = NO_CHANGES
        elif draft_failed or broken or (analyzed and not run.review.get("files_reviewed")):
            run.status = PARTIAL
        else:
            run.status = COMPLETED

        run.readiness = self._readiness(run, analyzed)
        run.limitations = self._limitations(run, spec, analyzed)
        run.unresolved = self._unresolved(run, analyzed)
        calls = [
            {"stage": f"review batch {batch['batch']}", "ms": batch["ms"], "input_tokens": batch.get("input_tokens"), "output_tokens": batch.get("output_tokens"), "status": batch["status"]}
            for batch in batches
        ]

        if run.draft.get("status") in ("generated",):
            calls.append({"stage": "draft", "ms": int(run.draft.get("seconds", 0) * 1000), "input_tokens": run.draft.get("input_tokens"), "output_tokens": run.draft.get("output_tokens"), "status": "generated"})

        run.llm = {
            "provider": self.provider_name,
            "models": sorted({model for model in (run.review.get("model"), run.draft.get("model")) if model}),
            "calls": calls,
            "local": self.provider_name == OLLAMA,
        }

    @staticmethod
    def _rule_findings(run: Run, analyzed: list[dict]) -> list[dict]:
        found = []

        for flag in run.documentation.get("flags", []):
            found.append(
                {
                    "category": "documentation",
                    "severity": "low",
                    "kind": "risk",
                    "file": flag["document"],
                    "line": flag["line"],
                    "title": f"{flag['document']} mentions files that changed, and it did not",
                    "description": (
                        f"{flag['document']} (line {flag['line']}) mentions {', '.join(flag['changed_files']) or ', '.join(flag['mentions'])}, "
                        "which changed in this release range, while the document itself did not change. A person should check that the text is still accurate."
                    ),
                    "recommendation": f"Read the passage in {flag['document']} next to the changes and update it if it is out of date.",
                    "evidence": flag["excerpt"],
                    "sources": [],
                    "source": "documentation_check",
                    "validation_status": "needs_human_check",
                }
            )

        sources = [item for item in analyzed if review_priority(item["path"]) == 0]
        tests = [item for item in run.files if is_test_path(item["path"])]

        if sources and not tests:
            found.append(
                {
                    "category": "testing",
                    "severity": "info",
                    "kind": "risk",
                    "file": None,
                    "line": None,
                    "title": "Source files changed and no test file did",
                    "description": f"{len(sources)} source files changed ({', '.join(item['path'] for item in sources[:4])}{'…' if len(sources) > 4 else ''}) and no test file is among the changed files of this range.",
                    "recommendation": "Check that the changed behaviour is covered by tests that already exist, or add tests.",
                    "evidence": f"changed files: {len(run.files)}, test files among them: 0",
                    "sources": [],
                    "source": "release_rules",
                    "validation_status": "deterministic",
                }
            )

        for check in run.checks:
            if check["status"] == release_checks.FAILED:
                found.append(
                    {
                        "category": "validation",
                        "severity": "high",
                        "kind": "defect",
                        "file": None,
                        "line": None,
                        "title": f"Check failed: {check['id']}",
                        "description": f"{check['command']} exited with status {check['exit_code']}.",
                        "recommendation": "Read the check's output in run.json, fix the cause, and run the release preparation again. The assistant does not change code to make a check pass.",
                        "evidence": (check["errors"] or [check["output_tail"][-200:]])[0] if (check["errors"] or check["output_tail"]) else "",
                        "sources": [],
                        "source": "check",
                        "validation_status": "executed",
                    }
                )

        return found

    @staticmethod
    def _readiness(run: Run, analyzed: list[dict]) -> dict:
        blockers, warnings, unverified = [], [], []
        passed = [check for check in run.checks if check["status"] == release_checks.PASSED]

        for check in run.checks:
            if check["status"] == release_checks.FAILED:
                blockers.append(f"check {check['id']} failed (exit status {check['exit_code']})")
            elif check["status"] != release_checks.PASSED:
                unverified.append(f"check {check['id']}: {check['status'].replace('_', ' ')}" + (f" - {check['reason']}" if check["reason"] else ""))

        if not run.checks:
            unverified.append("no check is configured for this repository")

        for item in run.findings:
            if item["source"] == "code_review" and item["kind"] == "defect" and item["severity"] in ("critical", "high"):
                blockers.append(f"{item['severity']} defect: {item['title']} ({item.get('file') or 'no file'})")
            elif item["source"] == "code_review" and item["severity"] in ("critical", "high", "medium"):
                warnings.append(f"{item['severity']} {item['kind']}: {item['title']} ({item.get('file') or 'no file'})")

        if run.git_status_at_start.get("conflicted"):
            blockers.append("the working tree has unresolved merge conflicts")

        if not run.scope.get("base_ref"):
            warnings.append("no base reference was given; the release range is not known")

        if run.other_changes.get("uncommitted_not_analyzed") or run.other_changes.get("untracked"):
            warnings.append("uncommitted or untracked files exist that were not analyzed")

        if run.scope.get("mode") in ("working_tree", "committed_range_and_uncommitted"):
            warnings.append("the analysis includes uncommitted changes, which may not belong to the release")

        if any(item["analysis"] == "withheld" for item in run.files):
            warnings.append("some changed files were withheld from analysis (secrets, generated or binary)")

        if run.review.get("not_reviewed"):
            warnings.append(f"{len(run.review['not_reviewed'])} changed files were not reviewed")

        if any(batch.get("context_problem") for batch in run.review.get("batches", [])):
            warnings.append("the RAG context was unavailable for part of the review")

        if run.documentation.get("flags"):
            warnings.append(f"{run.documentation.get('total_flagged', 0)} documents mention changed files and were not updated")

        if run.tree_after_checks.get("changed"):
            warnings.append("running the checks changed the working tree")

        if run.draft.get("status") == "failed":
            warnings.append("the changelog and release notes could not be generated")

        if run.status == FAILED:
            verdict, summary = NOT_VERIFIED, "Nothing was verified: the model was unavailable before the review could run."
        elif run.status == NO_CHANGES:
            verdict, summary = NOTHING_TO_RELEASE, "No changes were found in the selected scope."
        elif blockers:
            verdict, summary = BLOCKED, f"{len(blockers)} blocking problem(s) were found."
        elif not passed:
            verdict, summary = NOT_VERIFIED, "No check passed, so the changes are reviewed but not verified."
        elif warnings or unverified:
            verdict, summary = READY_WITH_WARNINGS, "The checks that ran passed; warnings and unrun checks remain."
        else:
            verdict, summary = READY, "Every configured check passed and nothing needs a person's attention."

        return {"verdict": verdict, "summary": summary, "blockers": blockers, "warnings": warnings, "unverified": unverified}

    @staticmethod
    def _limitations(run: Run, spec: RepositorySpec, analyzed: list[dict]) -> list[str]:
        notes = []

        if run.status != NO_CHANGES:
            notes.append("The review is done by a language model: findings are checked against the diff (the file exists, the evidence is a changed line), but the review can miss problems and its severity ratings are judgments.")

        if run.request.get("provider") == GEMINI:
            notes.append("Provider gemini was selected: the diffs and the review prompts were sent to Google's API. Use provider ollama to keep them on this machine.")
        elif not any("Local mode" in note for note in run.review.get("limitations", [])):
            notes.append("Provider ollama: the diffs and the drafts stayed on this machine. The review's context search sends only titles, paths and identifier names to the embedding model.")

        notes += [note for note in run.review.get("limitations", []) if note not in notes]

        if not any(check["status"] == release_checks.PASSED for check in run.checks):
            notes.append("No build, test or lint check passed in this run; nothing in this report states that the code builds or that tests pass.")

        notes.append("No release artifact was built and nothing was committed, tagged, pushed or published.")

        if any(item["analysis"] == "withheld" for item in run.files):
            notes.append("Withheld files were listed by path only; their content was not read by the model.")

        if not spec.user_facing:
            notes.append("This repository has no user-facing paths configured, so no release notes can be drawn from it.")

        return notes

    @staticmethod
    def _unresolved(run: Run, analyzed: list[dict]) -> list[str]:
        open_questions = []

        if not run.scope.get("base_ref"):
            open_questions.append("Which commit starts this release? Pass a base reference (a branch, tag or commit) to analyze a committed range instead of the working tree.")

        if run.other_changes.get("uncommitted_not_analyzed"):
            open_questions.append(f"{len(run.other_changes['uncommitted_not_analyzed'])} uncommitted files exist and were not analyzed. Do they belong to this release? (include them with include_uncommitted)")

        untracked = run.other_changes.get("untracked") or []

        if untracked:
            open_questions.append(f"{len(untracked)} untracked files or folders exist ({', '.join(item['path'] for item in untracked[:3])}{'…' if len(untracked) > 3 else ''}). Should they be part of the release?")

        upstream = run.repository.get("upstream")

        if upstream and upstream.get("ahead"):
            open_questions.append(f"The branch is {upstream['ahead']} commit(s) ahead of {upstream['name']}: they are not pushed yet.")

        if upstream and upstream.get("behind"):
            open_questions.append(f"The branch is {upstream['behind']} commit(s) behind {upstream['name']}.")

        for flag in run.documentation.get("flags", [])[:3]:
            open_questions.append(f"Is {flag['document']} (line {flag['line']}) still accurate after changes to {', '.join(flag['changed_files'][:2]) or 'the files it mentions'}?")

        existing = run.inspection.get("existing_release_documents", [])

        if existing:
            open_questions.append(f"{', '.join(existing[:3])} already exist in the repository and were not touched. Merge the generated entries by hand if they are wanted.")

        return open_questions

    # --- step 7 and 9: writing the artifacts, safely ----------------------------------------------------

    async def _write(self, run: Run, tools: Tools, started: float) -> None:
        """Save the artifacts through the project-files MCP server, which can only write inside the
        release output folder. They go into a run folder nobody has written to - a second run, or
        a person's own file with the same name, is never overwritten - and the credentials filter
        runs over every text first. The run record is written last and names the others."""
        names = (*release_artifacts.NAMES, release_artifacts.RECORD)
        directory = ""

        for _ in range(MAX_ID_TRIES):
            candidate = f"{run.output_dir}/{run.run_id}"

            try:
                probes = await asyncio.gather(*(tools.call(OUTPUT, "get_file_diff", {"path": f"{candidate}/{name}", "proposed_content": ""}) for name in names))
            except (ToolRefused, ReleaseError) as error:
                run.error("artifacts", "output_unavailable", f"{error}")
                run.status = FAILED if run.status == COMPLETED else run.status
                return

            if not any(probe.get("exists") for probe in probes):
                directory = candidate
                break

            run.run_id = self._id_factory(self._clock())

        if not directory:
            run.error("artifacts", "no_free_run_folder", "every run folder tried already held files; nothing was written")
            return

        run.run_dir = directory
        saved: list[dict] = []

        async def save(name: str, text: str) -> dict | None:
            clean = release_safety.redact(text)

            if clean != text:
                run.limitations.append(f"{name}: text that looked like a credential was redacted before it was written.")

            try:
                answer = await tools.call(OUTPUT, "write_project_file", {"path": f"{directory}/{name}", "content": clean})
            except (ToolRefused, ReleaseError) as error:
                run.error("artifacts", "write_failed", f"{name}: {error}")
                return None

            if not answer.get("created", True):
                run.error("artifacts", "replaced_existing_file", f"{name} already existed and was replaced")

            return {"name": name, "path": answer["path"], "sha256": answer["sha256"], "bytes": answer["bytes"], "created": answer.get("created", True)}

        for name, text in release_artifacts.render(run.as_dict()).items():
            entry = await save(name, text)

            if entry:
                saved.append(entry)

        run.artifacts = saved + [{"name": release_artifacts.RECORD, "path": f"{directory}/{release_artifacts.RECORD}", "sha256": "", "bytes": 0, "created": True}]

        if len(saved) < len(release_artifacts.NAMES) and run.status == COMPLETED:
            run.status = PARTIAL

        run.tool_calls = tools.log
        run.finished_at = _utc(self._clock())
        run.seconds = round(time.perf_counter() - started, 1)
        record = await save(release_artifacts.RECORD, release_artifacts.render_record(run.as_dict()))

        if record:
            run.artifacts[-1].update(record)
