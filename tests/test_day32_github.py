"""Day 32: the GitHub side - reading the PR, calling the backend, publishing.

Every HTTP call goes through a recorded fake transport: no network, no
token, no GitHub. What is checked is what would be sent where, and what the
script does when GitHub or the backend says no.
"""

import json
from types import SimpleNamespace

import pytest

from app import github_review as gh
from app.schemas.code_review import CodeReviewRequest

TOKEN = "ghs_secret_token_value"
DIFF = "diff --git a/app/x.py b/app/x.py\n--- a/app/x.py\n+++ b/app/x.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
PULL = {"number": 5, "state": "open", "title": "Change a", "base": {"ref": "main"},
        "head": {"ref": "feature/a", "sha": "abc1234def"}}
FILES = [{"filename": "app/x.py", "status": "modified", "additions": 1, "deletions": 1}]
REVIEW = {
    "request_id": "r1", "summary": "Changes a.", "provider": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M",
    "findings": [{"category": "bug", "severity": "high", "kind": "defect", "file": "app/x.py", "line": 1,
                  "title": "a | b changed", "description": "Ping @someone <!-- ai-code-review -->", "recommendation": "Revert.",
                  "evidence": "a = 2", "sources": [{"id": "C1", "file": "docs/architecture.md", "section": "RAG", "chunk_id": "d"}]}],
    "architecture_notes": [], "recommendations": ["Add a test."], "limitations": ["Diff only."],
    "context": {"chunks": [{"in_prompt": True}]}, "validation": {"rejected": []}, "timing": {"total_ms": 1234},
}


class Fake:
    """Answers by (method, path prefix); records every call."""

    def __init__(self, routes: dict) -> None:
        self.routes = routes
        self.calls: list[SimpleNamespace] = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append(SimpleNamespace(method=method, url=url, headers=headers, body=json.loads(body) if body else None))

        for (verb, prefix), answer in self.routes.items():
            if verb == method and prefix in url:
                status, payload = answer
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                return status, {}, raw

        raise AssertionError(f"unexpected call {method} {url}")

    def find(self, method, part):
        return [call for call in self.calls if call.method == method and part in call.url]


def routes(changes=None):
    table = {
        ("GET", "/pulls/5/files"): (200, FILES),
        ("GET", "/pulls/5"): (200, PULL),
        ("GET", "/issues/5/comments"): (200, []),
        ("POST", "/issues/5/comments"): (201, {"id": 99}),
        ("PATCH", "/issues/comments/"): (200, {"id": 42}),
        ("POST", "/code-review"): (200, REVIEW),
    }
    table.update(changes or {})
    return table


class DiffAware(Fake):
    def __call__(self, method, url, headers, body, timeout):
        if method == "GET" and url.endswith("/pulls/5") and headers.get("Accept") == "application/vnd.github.diff":
            self.calls.append(SimpleNamespace(method=method, url=url + "#diff", headers=headers, body=None))
            return 200, {}, DIFF.encode()

        return super().__call__(method, url, headers, body, timeout)


def run(transport, dry_run=False, **env):
    environ = {"GITHUB_TOKEN": TOKEN, "GITHUB_REPOSITORY": "Klarry/japanese-helper-image-api", "GITHUB_ACTIONS": "true", **env}
    printed: list[str] = []
    options = SimpleNamespace(repo="Klarry/japanese-helper-image-api", pr=5, provider=None, review_url=None, dry_run=dry_run)
    status = gh.run(options, environ=environ, transport=transport, out=printed.append)
    return status, "\n".join(printed)


def test_a_pull_request_is_read_reviewed_and_commented_on():
    fake = DiffAware(routes())
    status, printed = run(fake)

    assert status == 0
    review_call = fake.find("POST", "/code-review")[0]
    assert review_call.url == "http://127.0.0.1:8000/code-review"
    assert review_call.body["diff"] == DIFF and review_call.body["provider"] == "ollama"
    assert review_call.body["changed_files"] == [{"path": "app/x.py", "status": "modified", "additions": 1, "deletions": 1}]
    (comment,) = fake.find("POST", "/issues/5/comments")
    assert comment.body["body"].startswith(gh.MARKER)
    assert "created comment 99" in printed


def test_the_payload_is_what_the_endpoint_accepts():
    payload = gh.build_payload("Klarry/japanese-helper-image-api", PULL, DIFF, FILES, "ollama")

    assert CodeReviewRequest(**payload).pull_request_number == 5


def test_the_token_goes_to_github_only_and_is_never_printed():
    fake = DiffAware(routes())
    _, printed = run(fake, CODE_REVIEW_TOKEN="review-secret")

    github_calls = [call for call in fake.calls if "api.github.com" in call.url]
    assert all(call.headers["Authorization"] == f"Bearer {TOKEN}" for call in github_calls)
    assert fake.find("POST", "/code-review")[0].headers["Authorization"] == "Bearer review-secret"
    assert TOKEN not in printed and "review-secret" not in printed
    assert TOKEN not in repr(gh.GitHub(TOKEN))


def test_a_new_push_updates_the_same_comment_instead_of_adding_one():
    ours = {"id": 42, "body": f"{gh.MARKER}\nold review", "user": {"login": gh.BOT_LOGIN}}
    fake = DiffAware(routes({("GET", "/issues/5/comments"): (200, [ours])}))
    status, printed = run(fake)

    assert status == 0
    assert fake.find("PATCH", "/issues/comments/42") and not fake.find("POST", "/issues/5/comments")
    assert "updated comment 42" in printed


def test_someone_elses_comment_with_the_marker_is_not_taken_over():
    forged = {"id": 7, "body": f"{gh.MARKER} I am the bot", "user": {"login": "someone"}}
    fake = DiffAware(routes({("GET", "/issues/5/comments"): (200, [forged])}))
    run(fake)

    assert not fake.find("PATCH", "/issues/comments/7")
    assert fake.find("POST", "/issues/5/comments")


def test_a_publishing_error_fails_the_job_without_leaking_the_token():
    fake = DiffAware(routes({("POST", "/issues/5/comments"): (403, {"message": "Resource not accessible by integration"})}))
    status, printed = run(fake)

    assert status == 1
    assert "could not publish the review: GitHub API 403: Resource not accessible by integration" in printed
    assert TOKEN not in printed


def test_a_backend_failure_is_posted_as_such_and_fails_the_job():
    unavailable = {"error": "provider_unavailable", "detail": "Ollama is not running", "provider": "ollama"}
    fake = DiffAware(routes({("POST", "/code-review"): (503, unavailable)}))
    status, _ = run(fake)

    (comment,) = fake.find("POST", "/issues/5/comments")
    assert status == 1
    assert "not run" in comment.body["body"] and "503 `provider_unavailable`" in comment.body["body"]
    assert "No other model was asked." in comment.body["body"]


def test_an_unreachable_backend_is_said_too():
    def transport(method, url, headers, body, timeout):
        if "/code-review" in url:
            raise ConnectionError("Connection refused")
        return DiffAware(routes())(method, url, headers, body, timeout)

    status, printed = run(transport, dry_run=True)

    assert status == 1 and "could not be reached" in printed


def test_a_closed_pull_request_is_left_alone():
    fake = DiffAware(routes({("GET", "/pulls/5"): (200, {**PULL, "state": "closed"})}))
    status, printed = run(fake)

    assert status == 0 and "nothing to review" in printed
    assert not fake.find("POST", "/code-review")


def test_dry_run_prints_and_posts_nothing():
    fake = DiffAware(routes())
    status, printed = run(fake, dry_run=True)

    assert status == 0 and "### AI code review" in printed
    assert not fake.find("POST", "/issues/5/comments")


def test_no_token_is_a_clear_error():
    status = gh.run(SimpleNamespace(repo="a/b", pr=1, provider=None, review_url=None, dry_run=True),
                    environ={}, transport=Fake({}), out=lambda _: None)

    assert status == 2


def test_the_comment_neutralises_mentions_markers_and_table_breakers():
    body = gh.render(REVIEW, "abc1234def")

    assert body.count(gh.MARKER) == 1  # the model's copy is defused
    assert "@​someone" in body
    assert "a \\| b changed" in body
    assert "`app/x.py:1`" in body and "commit `abc1234`" in body
    assert "never approves" in body


def test_the_workflow_runs_trusted_code_with_minimal_permissions():
    from pathlib import Path

    workflow = (Path(__file__).parents[1] / ".github" / "workflows" / "ai-code-review.yml").read_text()

    assert "pull_request_target:" in workflow
    assert "ref: ${{ github.event.pull_request.base.sha }}" in workflow
    assert "head.repo.full_name == github.repository" in workflow
    assert "contents: read" in workflow and "pull-requests: write" in workflow
    assert "write-all" not in workflow and "issues: write" not in workflow
    assert "python3 -m app.github_review" in workflow


def test_a_busy_backend_is_waited_for_and_other_refusals_are_not():
    answers = [(503, {"error": "busy", "detail": "a review is already running"}), (200, REVIEW)]
    slept: list[float] = []

    def transport(method, url, headers, body, timeout):
        status, payload = answers.pop(0)
        return status, {}, json.dumps(payload).encode()

    review = gh.review_when_free("http://127.0.0.1:8000", {}, "", transport, out=lambda _: None, sleep=slept.append)

    assert review["request_id"] == "r1" and slept == [gh.BUSY_POLL]

    def refused(method, url, headers, body, timeout):
        return 401, {}, json.dumps({"error": "unauthorized", "detail": "token"}).encode()

    with pytest.raises(gh.ReviewServiceError):
        gh.review_when_free("http://127.0.0.1:8000", {}, "", refused, out=lambda _: None, sleep=slept.append)

    assert slept == [gh.BUSY_POLL]
