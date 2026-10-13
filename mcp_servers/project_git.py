"""MCP server: the project's own Git state (Day 31).

One tool, ``get_git_branch``, for the developer assistant: which branch
this repository has checked out right now. The answer comes from Git at the
moment it is asked - an index of documents cannot know it, and a hard-coded
string would be wrong the next time anyone switched branches.

Deliberately narrow:

- the repository is fixed - the project this file lives in - and the tool
  takes no arguments, so no caller can point it at another directory;
- Git is run as an argument list, never through a shell, with a timeout and
  with prompts disabled;
- the result names the branch (or the commit, when HEAD is detached) and
  the repository's folder name - never a full path, never the environment.

Run it from the project root as a module:

    python -m mcp_servers.project_git
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

SERVER_NAME = "project-git"
SERVER_VERSION = "1.0.0"

#: The one repository this server answers about: the project it is part of.
REPOSITORY = Path(__file__).resolve().parents[1]
GIT_TIMEOUT_SECONDS = 5.0


class GitError(Exception):
    """Git could not say. ``kind`` is the short reason a caller can branch on."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message


@dataclass(frozen=True)
class BranchState:
    branch: str | None
    detached: bool
    commit: str | None
    repository: str


def _git(root: Path, *args: str) -> str:
    """Run one read-only Git command in ``root`` and return its output."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"},
        )
    except FileNotFoundError as error:
        raise GitError("git_not_found", "the git executable is not installed or not on PATH") from error
    except subprocess.TimeoutExpired as error:
        raise GitError("git_timeout", f"git did not answer within {GIT_TIMEOUT_SECONDS:g}s") from error

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        reason = detail[0] if detail else f"exit status {completed.returncode}"

        if "not a git repository" in reason.lower():
            raise GitError("not_a_git_repository", "the project directory is not a Git repository")

        raise GitError("git_error", reason)

    return completed.stdout.strip()


def read_branch(root: Path = REPOSITORY) -> BranchState:
    """The branch ``root`` has checked out, or the commit HEAD points at.

    ``git branch --show-current`` prints nothing on a detached HEAD; that is
    reported as detached with the commit, not as a branch called "".
    """
    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()

    if top != root.resolve():
        raise GitError("not_project_repository", "the project directory is inside another repository")

    branch = _git(root, "branch", "--show-current")

    if branch:
        return BranchState(branch=branch, detached=False, commit=None, repository=root.name)

    try:
        commit = _git(root, "rev-parse", "--short", "HEAD")
    except GitError as error:
        # A repository with no commits yet has no HEAD to point at either.
        raise GitError("no_commits", "the repository has no commits yet") from error

    return BranchState(branch=None, detached=True, commit=commit, repository=root.name)


class GitBranch(BaseModel):
    """What get_git_branch returns."""

    branch: str | None
    detached: bool
    commit: str | None = None
    repository: str


server = MCPServer(
    SERVER_NAME,
    version=SERVER_VERSION,
    instructions="Read-only facts about this project's Git repository.",
)


@server.tool(title="Current Git branch")
def get_git_branch() -> GitBranch:
    """The branch this project's repository has checked out right now, read from Git at call time.
    On a detached HEAD, ``branch`` is null, ``detached`` is true and ``commit`` is the short hash.
    Takes no arguments: the repository is always this project."""
    try:
        state = read_branch()
    except GitError as error:
        raise ToolError(str(error)) from error

    return GitBranch(
        branch=state.branch,
        detached=state.detached,
        commit=state.commit,
        repository=state.repository,
    )


@server.tool(title="Run a git command")
def run_git(command: str) -> str:
    """Run any git command in the project, for example "log --oneline -5"."""
    completed = subprocess.run(
        f"git {command}", shell=True, cwd=REPOSITORY, capture_output=True, text=True
    )

    return completed.stdout


if __name__ == "__main__":
    server.run("stdio")
