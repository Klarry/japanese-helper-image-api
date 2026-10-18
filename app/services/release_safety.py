"""Day 35: what must never reach a model prompt or a release artifact.

The release assistant sends diffs, commit subjects and documentation
excerpts to a model and writes reports to disk. Three rules keep that safe,
and they live here once so the Git MCP server and the assistant apply the
same ones:

- a file is **withheld** - listed by path, never shown - when its name says
  it holds secrets (``.env``, ``*.pem``, ``google-services.json``,
  ``local.properties``), when it is generated or lives in a build/cache/
  output folder, or when it is binary;
- a diff is withheld when its text looks like it contains a credential (a
  private key, an API key or token of a known shape, a ``password = "..."``
  assignment with a real-looking value);
- any text written to an artifact goes through ``redact`` first.

The patterns are deliberately conservative: a false alarm withholds one
file's diff and says so; a miss would put a secret into a prompt or a report.
The reasons are the vocabulary the rest of the release code uses.
"""

import re
from pathlib import PurePosixPath

from app.services.project_files import EXCLUDED_DIRS, is_secret_name

SECRET_FILE = "secret_file"
GENERATED = "generated"
BINARY = "binary"
POSSIBLE_SECRET = "possible_secret"
TOO_LARGE = "too_large"
OVER_BUDGET = "over_budget"
UNREADABLE = "unreadable"

#: Why a file's content was not shown, in words for a report.
REASONS = {
    SECRET_FILE: "the file name says it holds secrets or credentials",
    GENERATED: "generated, build or cache output",
    BINARY: "binary file - no text to show",
    POSSIBLE_SECRET: "the diff contains text that looks like a credential",
    TOO_LARGE: "the diff of this file is larger than the per-file limit",
    OVER_BUDGET: "left out to keep the collected diff within its size limit",
    UNREADABLE: "git's diff for this file could not be matched to its path",
}

#: Folders that hold output, not source - beyond the file tools' own list.
_GENERATED_DIRS = frozenset({"output", "generated", "intermediates", "captures", ".kotlin", "outputs", "coverage", "htmlcov"})
_GENERATED_PREFIXES = ("data/index",)
_GENERATED_NAME = re.compile(
    r"""^(
        package-lock\.json | yarn\.lock | pnpm-lock\.yaml | poetry\.lock | Pipfile\.lock | Gemfile\.lock
      | gradle\.lockfile | .*\.lock | \.DS_Store | .*\.pyc | .*\.min\.(js|css) | .*\.map
    )$""",
    re.IGNORECASE | re.VERBOSE,
)
_BINARY_NAME = re.compile(
    r".*\.(png|jpe?g|gif|webp|ico|bmp|mp[34]|mov|wav|ogg|pdf|zip|gz|tgz|jar|aar|apk|aab|class|dex|so|dylib|a|o|"
    r"faiss|pkl|pickle|npy|db|sqlite3?|ttf|otf|woff2?)$",
    re.IGNORECASE,
)
#: Secrets by file name that the file tools do not already refuse.
_EXTRA_SECRET_NAME = re.compile(r"^(local\.properties|keystore\.properties|.*\.(jks|keystore|p12))$", re.IGNORECASE)


def classify_path(path: str) -> str:
    """Why this file's content must not be shown ("" when it may be)."""
    pure = PurePosixPath(path)
    name = pure.name

    if is_secret_name(name) or _EXTRA_SECRET_NAME.match(name):
        return SECRET_FILE

    parts = pure.parts

    if any(part in EXCLUDED_DIRS or part in _GENERATED_DIRS or part.startswith(".venv") or part.endswith(".egg-info") for part in parts[:-1]):
        return GENERATED

    if any(path == prefix or path.startswith(prefix + "/") for prefix in _GENERATED_PREFIXES):
        return GENERATED

    if _GENERATED_NAME.match(name):
        return GENERATED

    if _BINARY_NAME.match(name):
        return BINARY

    return ""


_PLACEHOLDER = re.compile(
    r"(your[-_ ]|changeme|change[-_]me|example|placeholder|xxxx|\*{4,}|dummy|sample|<.*>|\$\{|\{\{|%s|test[-_]key|fake|redacted)",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(
    r"""(?ix)
    \b(api[_-]?key|secret|token|passw(?:or)?d|private[_-]?key|client[_-]?secret|access[_-]?key|auth[_-]?key)\b
    ["']?\s*[:=]\s*["']([^"'\s$<{][^"'\s]{11,})["']
    """
)
_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{24,}"),
)
REDACTED = "[redacted]"


def _assignment_hits(text: str):
    for match in _ASSIGNMENT.finditer(text):
        if not _PLACEHOLDER.search(match.group(2)):
            yield match


def contains_secret(text: str) -> bool:
    """True when ``text`` has a private key, a token of a known shape, or a
    credential assignment with a real-looking value."""
    if any(pattern.search(text) for pattern in _PATTERNS):
        return True

    return next(_assignment_hits(text), None) is not None


def redact(text: str) -> str:
    """``text`` with everything ``contains_secret`` would flag replaced."""
    for pattern in _PATTERNS:
        text = pattern.sub(REDACTED, text)

    for match in reversed(list(_assignment_hits(text))):
        text = text[: match.start(2)] + REDACTED + text[match.end(2):]

    return text
