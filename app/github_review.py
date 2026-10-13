"""Day 32: the GitHub side of the AI code review - fetch, ask, publish.

    python3 -m app.github_review                                     # in GitHub Actions
    python3 -m app.github_review --repo owner/name --pr 12 --dry-run # by hand: print, do not post

Everything that talks to GitHub is here, and nothing else is: the review
itself is the backend's (``POST /code-review``), which never reaches GitHub.
This script reads the pull request and its diff through the GitHub API,
sends them to the backend, and publishes the result as one PR comment -
updated in place on every new push, so a PR collects one review, not one
per commit.

Standard library only, and Python 3.9-compatible: in Actions it runs with
the runner's system python3, from a checkout of the base branch - never of
the pull request, whose code is only ever read as a diff.

Environment:
    GITHUB_TOKEN          the workflow token (contents: read, pull-requests: write); never printed
    GITHUB_REPOSITORY     owner/name - set by Actions
    GITHUB_EVENT_PATH     the pull_request event - set by Actions
    GITHUB_API_URL        default https://api.github.com
    CODE_REVIEW_URL       the backend, default http://127.0.0.1:8000
    CODE_REVIEW_TOKEN     sent as "Authorization: Bearer ..." when set
    CODE_REVIEW_PROVIDER  ollama (default) or gemini

It never approves, requests changes or merges: the only write is an issue
comment on the pull request.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

MARKER = "<!-- ai-code-review -->"
BOT_LOGIN = "github-actions[bot]"
USER_AGENT = "japanese-helper-ai-code-review"
MAX_DOWNLOAD = 2_000_000  # bytes read from any one response
REVIEW_TIMEOUT = 900.0  # a local model on a laptop can take minutes on a large diff
GITHUB_TIMEOUT = 30.0
BUSY_WAIT = 900.0  # how long to wait for another PR's review to finish (the backend runs one at a time)
BUSY_POLL = 20.0

Transport = Callable[[str, str, dict, Optional[bytes], float], "tuple[int, dict, bytes]"]


class GitHubError(Exception):
    """GitHub answered with an error. Carries its status and message - never the token."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status
        self.message = message


class ReviewServiceError(Exception):
    """The review backend did not return a review."""

    def __init__(self, status: int, error: str, detail: str) -> None:
        super().__init__(f"review service {status} {error}: {detail}")
        self.status = status
        self.error = error
        self.detail = detail


def urllib_transport(method: str, url: str, headers: dict, body: Optional[bytes], timeout: float):
    request = urllib.request.Request(url, data=body, method=method, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read(MAX_DOWNLOAD)
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers or {}), error.read(MAX_DOWNLOAD) if error.fp else b""
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ConnectionError(str(getattr(error, "reason", error))) from None


def _message(raw: bytes) -> str:
    try:
        document = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return raw.decode("utf-8", "replace")[:200]

    if isinstance(document, dict):
        return str(document.get("message") or document.get("detail") or document)[:300]

    return str(document)[:300]


class GitHub:
    """The five GitHub calls the review needs, and no others."""

    def __init__(self, token: str, api: str = "https://api.github.com", transport: Transport = urllib_transport) -> None:
        if not token:
            raise ValueError("a GitHub token is required")

        self._token = token
        self._api = api.rstrip("/")
        self._transport = transport

    def __repr__(self) -> str:  # the token stays out of any traceback or log
        return f"GitHub(api={self._api!r})"

    def _call(self, method: str, path: str, accept: str = "application/vnd.github+json", payload=None):
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": USER_AGENT,
        }
        body = None

        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        status, _, raw = self._transport(method, f"{self._api}{path}", headers, body, GITHUB_TIMEOUT)

        if status >= 400:
            raise GitHubError(status, _message(raw))

        return raw

    def _json(self, method: str, path: str, payload=None):
        raw = self._call(method, path, payload=payload)
        return json.loads(raw.decode("utf-8")) if raw else None

    def pull(self, repo: str, number: int) -> dict:
        return self._json("GET", f"/repos/{repo}/pulls/{number}")

    def diff(self, repo: str, number: int) -> str:
        return self._call("GET", f"/repos/{repo}/pulls/{number}", accept="application/vnd.github.diff").decode(
            "utf-8", "replace"
        )

    def files(self, repo: str, number: int) -> list:
        found: list = []

        for page in range(1, 4):  # GitHub lists at most 3000; the backend reviews at most 100
            batch = self._json("GET", f"/repos/{repo}/pulls/{number}/files?per_page=100&page={page}") or []
            found.extend(batch)

            if len(batch) < 100:
                break

        return found

    def comments(self, repo: str, number: int) -> list:
        found: list = []

        for page in range(1, 11):
            batch = self._json("GET", f"/repos/{repo}/issues/{number}/comments?per_page=100&page={page}") or []
            found.extend(batch)

            if len(batch) < 100:
                break

        return found

    def login(self) -> str:
        return str((self._json("GET", "/user") or {}).get("login", ""))

    def create_comment(self, repo: str, number: int, body: str) -> dict:
        return self._json("POST", f"/repos/{repo}/issues/{number}/comments", {"body": body})

    def update_comment(self, repo: str, comment_id: int, body: str) -> dict:
        return self._json("PATCH", f"/repos/{repo}/issues/comments/{comment_id}", {"body": body})


# --- the backend ------------------------------------------------------------------------


def build_payload(repo: str, pull: dict, diff: str, files: list, provider: str) -> dict:
    return {
        "repository": repo,
        "pull_request_number": int(pull["number"]),
        "base_branch": pull["base"]["ref"],
        "head_branch": pull["head"]["ref"],
        "title": str(pull.get("title") or "")[:300],
        "diff": diff,
        "changed_files": [
            {
                "path": item["filename"],
                "status": item.get("status", "modified"),
                "additions": item.get("additions"),
                "deletions": item.get("deletions"),
            }
            for item in files[:300]
        ],
        "provider": provider,
    }


def request_review(url: str, payload: dict, token: str = "", transport: Transport = urllib_transport) -> dict:
    headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}

    if token:
        headers["Authorization"] = f"Bearer {token}"

    status, _, raw = transport(
        "POST", f"{url.rstrip('/')}/code-review", headers, json.dumps(payload).encode("utf-8"), REVIEW_TIMEOUT
    )

    if status != 200:
        try:
            document = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            document = {}

        if not isinstance(document, dict):
            document = {}

        detail = document.get("detail")
        detail = detail if isinstance(detail, str) else json.dumps(detail)[:300] if detail else raw.decode("utf-8", "replace")[:300]
        raise ReviewServiceError(status, str(document.get("error") or "error"), detail)

    return json.loads(raw.decode("utf-8"))


# --- the comment ------------------------------------------------------------------------

_MENTION = re.compile(r"@(?=[A-Za-z0-9])")
SEVERITY_ORDER = ("critical", "high", "medium", "low", "info")


def sanitize(text, limit: int = 2000) -> str:
    """Model output made safe to post: no HTML comments (the marker cannot be
    forged), no @mentions that would ping people, bounded length."""
    value = str(text or "").replace("<!--", "&lt;!--").replace("-->", "--&gt;")
    value = _MENTION.sub("@\u200b", value)

    return value if len(value) <= limit else value[: limit - 1] + "…"


def _cell(text, limit: int = 160) -> str:
    return sanitize(text, limit).replace("|", "\\|").replace("\n", " ")


def _fenced(code: str) -> str:
    fence = "~~~" if "```" in code else "```"
    return f"{fence}\n{sanitize(code, 600)}\n{fence}"


def render(review: dict, commit: str = "") -> str:
    findings = review.get("findings") or []
    context = review.get("context") or {}
    used = [chunk for chunk in context.get("chunks", []) if chunk.get("in_prompt")]
    seconds = (review.get("timing") or {}).get("total_ms", 0) / 1000
    lines = [
        MARKER,
        "### AI code review",
        "",
        f"**Model:** `{sanitize(review.get('model'), 80)}` ({sanitize(review.get('provider'), 20)}) · "
        f"**Findings:** {len(findings)} · **Project context:** {len(used)} chunk(s) · {seconds:.1f} s",
        "",
        sanitize(review.get("summary") or "", 1500),
        "",
    ]

    if findings:
        lines += ["| Severity | Category | Where | Finding |", "|---|---|---|---|"]

        for finding in findings:
            where = finding.get("file", "") + (f":{finding['line']}" if finding.get("line") else "")
            lines.append(
                f"| {_cell(finding.get('severity'), 12)} | {_cell(finding.get('category'), 20)} · {_cell(finding.get('kind'), 8)} "
                f"| `{_cell(where, 120)}` | {_cell(finding.get('title'))} |"
            )

        lines.append("")

        for number, finding in enumerate(findings, start=1):
            where = finding.get("file", "") + (f":{finding['line']}" if finding.get("line") else "")
            sources = ", ".join(
                f"`{sanitize(source.get('file'), 100)}` · {sanitize(source.get('section'), 80)}"
                for source in finding.get("sources") or []
            )
            lines += [
                f"<details><summary><b>{number}. {sanitize(finding.get('title'), 160)}</b> — {_cell(where, 120)}</summary>",
                "",
                sanitize(finding.get("description"), 1200),
                "",
                f"**Recommendation:** {sanitize(finding.get('recommendation'), 800) or '—'}",
                "",
                "**Evidence (from the diff):**",
                _fenced(str(finding.get("evidence") or "")),
            ]

            if sources:
                lines.append(f"**Project context:** {sources}")

            lines += ["", "</details>", ""]
    else:
        lines += ["No actionable findings.", ""]

    for title, key in (("Architecture notes", "architecture_notes"), ("Recommendations", "recommendations"), ("Limitations", "limitations")):
        items = review.get(key) or []

        if items:
            lines.append(f"**{title}**")
            lines += [f"- {sanitize(item, 500)}" for item in items]
            lines.append("")

    rejected = len((review.get("validation") or {}).get("rejected") or [])
    footer = f"commit `{commit[:7]}` · " if commit else ""
    lines.append(
        f"<sub>{footer}request `{sanitize(review.get('request_id'), 20)}` · {rejected} model finding(s) rejected by validation · "
        "automatic suggestions: this comment never approves, blocks or merges the PR.</sub>"
    )

    return "\n".join(lines)


def render_failure(reason: str, commit: str = "") -> str:
    footer = f"<sub>commit `{commit[:7]}`</sub>" if commit else ""
    return "\n".join(
        [MARKER, "### AI code review — not run", "", sanitize(reason, 600), "", "No other model was asked.", "", footer]
    ).strip()


def publish(github: GitHub, repo: str, number: int, body: str, author: str) -> "tuple[str, int]":
    """Update this tool's earlier comment, or create the first one. Only a
    comment that carries the marker *and* was written by ``author`` counts:
    anyone can paste the marker, nobody else can be the bot."""
    for comment in github.comments(repo, number):
        if MARKER in (comment.get("body") or "") and (comment.get("user") or {}).get("login") == author:
            github.update_comment(repo, comment["id"], body)
            return "updated", int(comment["id"])

    created = github.create_comment(repo, number, body)

    return "created", int(created["id"])


# --- the command ------------------------------------------------------------------------


def _event() -> dict:
    path = os.environ.get("GITHUB_EVENT_PATH", "")

    if not path:
        return {}

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def review_when_free(url: str, payload: dict, token: str, transport: Transport, out=print, sleep=time.sleep,
                     wait: float = BUSY_WAIT) -> dict:
    """The review, waiting while the backend is busy with another one. Only
    "busy" is retried: every other refusal is an answer."""
    waited = 0.0

    while True:
        try:
            return request_review(url, payload, token, transport)
        except ReviewServiceError as error:
            if error.error != "busy" or waited >= wait:
                raise

            out(f"backend busy with another review; retrying in {BUSY_POLL:.0f}s")
            sleep(BUSY_POLL)
            waited += BUSY_POLL


def run(options, environ=os.environ, transport: Transport = urllib_transport, out=print, sleep=time.sleep) -> int:
    token = environ.get("GITHUB_TOKEN", "")

    if not token:
        out("GITHUB_TOKEN is not set")
        return 2

    repo = options.repo or environ.get("GITHUB_REPOSITORY", "")
    number = options.pr or int(((_event() if not options.repo else {}).get("pull_request") or {}).get("number") or 0)

    if not repo or not number:
        out("no pull request: pass --repo and --pr, or run from a pull_request event")
        return 2

    github = GitHub(token, environ.get("GITHUB_API_URL", "https://api.github.com"), transport)
    provider = options.provider or environ.get("CODE_REVIEW_PROVIDER", "") or "ollama"
    url = options.review_url or environ.get("CODE_REVIEW_URL", "") or "http://127.0.0.1:8000"

    try:
        pull = github.pull(repo, number)

        if pull.get("state") != "open":
            out(f"{repo}#{number} is {pull.get('state')}; nothing to review")
            return 0

        commit = str((pull.get("head") or {}).get("sha") or "")
        diff = github.diff(repo, number)
        files = github.files(repo, number)
    except GitHubError as error:
        out(f"could not read {repo}#{number}: {error}")
        return 1
    except ConnectionError as error:
        out(f"could not reach GitHub: {error}")
        return 1

    out(f"{repo}#{number}: {len(files)} file(s), {len(diff)} diff characters -> {url} (provider {provider})")
    status = 0

    try:
        review = review_when_free(
            url, build_payload(repo, pull, diff, files, provider), environ.get("CODE_REVIEW_TOKEN", ""), transport, out, sleep
        )
        body = render(review, commit)
        out(f"review: {len(review.get('findings') or [])} finding(s), model {review.get('model')}, {review.get('timing', {}).get('total_ms')} ms")
    except ReviewServiceError as error:
        body = render_failure(f"The review service answered {error.status} `{error.error}`: {error.detail}", commit)
        out(f"review failed: {error.status} {error.error}")
        status = 1
    except ConnectionError as error:
        body = render_failure(f"The review service at {url} could not be reached ({error}).", commit)
        out(f"review service not reachable at {url}")
        status = 1

    if options.dry_run:
        out(body)
        return status

    author = BOT_LOGIN if environ.get("GITHUB_ACTIONS") == "true" else None

    try:
        author = author or github.login()
        action, comment_id = publish(github, repo, number, body, author)
    except GitHubError as error:
        out(f"could not publish the review: {error}")
        return 1
    except ConnectionError as error:
        out(f"could not reach GitHub to publish: {error}")
        return 1

    out(f"{action} comment {comment_id} on {repo}#{number}")

    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m app.github_review", description="Review a pull request and comment on it.")
    parser.add_argument("--repo", help="owner/name (default: GITHUB_REPOSITORY)")
    parser.add_argument("--pr", type=int, help="pull request number (default: from the Actions event)")
    parser.add_argument("--provider", choices=("ollama", "gemini"))
    parser.add_argument("--review-url", help="the backend (default: CODE_REVIEW_URL or http://127.0.0.1:8000)")
    parser.add_argument("--dry-run", action="store_true", help="print the comment instead of posting it")

    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
