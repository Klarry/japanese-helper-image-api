"""Day 32: a manual code review of a local diff, from the terminal.

    git diff main...HEAD | python -m app.review_diff --provider ollama
    python -m app.review_diff changes.diff --title "Parse Retry-After" --json

The same reviewer the /code-review endpoint and the GitHub Action use - the
diff is read from a file or stdin, nothing is run, nothing is posted. The
output is the comment the Action would post (or the full JSON with --json).
"""

import argparse
import asyncio
import json
import logging
import sys

from app.github_review import render
from app.services.code_review import CodeReviewer, InvalidReviewOutput, PullRequest, ReviewRequestError, ReviewTooLarge
from app.services.llm_provider import OLLAMA, PROVIDERS, ProviderUnavailable


async def run(options) -> int:
    diff = sys.stdin.read() if options.diff in (None, "-") else open(options.diff, encoding="utf-8").read()
    pr = PullRequest(
        repository="local/working-copy",
        number=1,
        base_branch=options.base,
        head_branch=options.head,
        title=options.title,
        diff=diff,
    )

    try:
        result = (await CodeReviewer(options.provider).review(pr)).as_dict()
    except (ReviewRequestError, ReviewTooLarge) as error:
        print(f"cannot review this diff: {error}")
        return 2
    except InvalidReviewOutput as error:
        print(f"{error.provider} did not return a usable review: {error}")
        return 3
    except ProviderUnavailable as error:
        print(f"{error.provider} is unavailable ({error.reason}) - no other model was asked.\n{error.message}")
        return 3

    print(json.dumps(result, ensure_ascii=False, indent=2) if options.json else render(result))

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.review_diff", description="Review a unified diff.")
    parser.add_argument("diff", nargs="?", help="a .diff file, or - / nothing for stdin")
    parser.add_argument("--provider", default=OLLAMA, choices=list(PROVIDERS))
    parser.add_argument("--title", default="")
    parser.add_argument("--base", default="main")
    parser.add_argument("--head", default="working-copy")
    parser.add_argument("--json", action="store_true")
    options = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    return asyncio.run(run(options))


if __name__ == "__main__":
    raise SystemExit(main())
