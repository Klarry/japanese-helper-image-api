"""Day 32: a pull request's unified diff, read into files, hunks and lines.

The review is only as good as its idea of what changed, so the diff is read
here once, strictly, and everything downstream works from the result: which
files the model may name, which line numbers exist, and which text a
finding's evidence has to be found in.

Every line keeps the number it has in the new file (added and context
lines) or the old one (removed lines). The prompt shows those numbers next
to the code, so the model never has to count hunk offsets - and a finding
that names a line the diff does not show can be caught instead of believed.

Paths are labels here, never opened: nothing in this module touches the
filesystem. They are still refused when they could be read as one - an
absolute path, a ``..`` segment, a control character - because a label that
looks like a path tends to end up used as one.
"""

import re
from dataclasses import dataclass, field

MAX_PATH_CHARS = 300

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")
_GIT_HEADER = re.compile(r'^diff --git "?a/(.+?)"? "?b/(.+?)"?$')
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

ADDED = "added"
MODIFIED = "modified"
REMOVED = "removed"
RENAMED = "renamed"
STATUSES = (ADDED, MODIFIED, REMOVED, RENAMED)


class DiffError(ValueError):
    """The text is not a diff this service can review."""


def safe_path(path: str) -> str:
    """``path`` as a repository-relative label, or DiffError."""
    text = (path or "").strip()

    if not text:
        raise DiffError("a file in the diff has no path")

    if len(text) > MAX_PATH_CHARS:
        raise DiffError(f"path longer than {MAX_PATH_CHARS} characters")

    if _CONTROL.search(text) or "\\" in text:
        raise DiffError(f"path {text[:80]!r} contains control characters or backslashes")

    if text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        raise DiffError(f"path {text[:80]!r} is absolute; only repository-relative paths are accepted")

    if any(part in ("..", ".") for part in text.split("/")) or "//" in text:
        raise DiffError(f"path {text[:80]!r} is not a plain repository-relative path")

    return text


@dataclass(frozen=True)
class DiffLine:
    kind: str  # "+", "-" or " "
    text: str
    new: int | None  # number in the new file: added and context lines
    old: int | None  # number in the old file: removed and context lines


@dataclass
class Hunk:
    header: str
    lines: list[DiffLine] = field(default_factory=list)


@dataclass
class FileDiff:
    path: str
    status: str = MODIFIED
    old_path: str | None = None
    binary: bool = False
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def additions(self) -> int:
        return sum(1 for hunk in self.hunks for line in hunk.lines if line.kind == "+")

    @property
    def deletions(self) -> int:
        return sum(1 for hunk in self.hunks for line in hunk.lines if line.kind == "-")

    @property
    def lines(self) -> list[DiffLine]:
        return [line for hunk in self.hunks for line in hunk.lines]

    @property
    def new_numbers(self) -> set[int]:
        """The new-file line numbers this diff actually shows."""
        return {line.new for line in self.lines if line.new is not None}

    @property
    def is_test(self) -> bool:
        name = self.path.rsplit("/", 1)[-1]
        return self.path.startswith("tests/") or "/tests/" in self.path or name.startswith("test_")

    def label(self) -> str:
        moved = f" (from {self.old_path})" if self.status == RENAMED and self.old_path else ""
        kind = "binary" if self.binary else f"+{self.additions} -{self.deletions}"
        return f"{self.path}{moved} ({self.status}, {kind})"

    def render(self, hunks: list[Hunk] | None = None) -> str:
        """The file as the prompt shows it: every line with its number."""
        out = [f"### {self.label()}"]

        if self.binary:
            out.append("(binary file - no text to review)")

        for hunk in self.hunks if hunks is None else hunks:
            out.append(hunk.header)

            for line in hunk.lines:
                number = line.new if line.new is not None else None
                out.append(f"{number if number is not None else '':>5} {line.kind} {line.text}")

        return "\n".join(out)


def _strip_prefix(path: str) -> str:
    return path[2:] if path[:2] in ("a/", "b/") else path


def parse_diff(text: str) -> list[FileDiff]:
    """Every file in a unified diff (``git diff`` / GitHub's ``.diff``).

    Raises DiffError for text that has no file in it, and for a path that is
    not a plain repository-relative one.
    """
    if not text or not text.strip():
        raise DiffError("the diff is empty")

    files: list[FileDiff] = []
    current: FileDiff | None = None
    hunk: Hunk | None = None
    old_left = new_left = 0
    old_no = new_no = 0

    def start(path: str) -> FileDiff:
        made = FileDiff(path=path)
        files.append(made)
        return made

    for raw in text.splitlines():
        if raw.startswith("diff --git "):
            # A new file always ends the hunk before it, even one whose
            # header promised more lines than it had.
            hunk = None

        if hunk is not None and (old_left > 0 or new_left > 0):
            if raw.startswith("\\"):  # "\ No newline at end of file"
                continue

            kind = raw[:1] if raw[:1] in ("+", "-", " ") else " "
            body = raw[1:] if raw[:1] in ("+", "-", " ") else raw

            if kind == "+":
                hunk.lines.append(DiffLine("+", body, new_no, None))
                new_no += 1
                new_left -= 1
            elif kind == "-":
                hunk.lines.append(DiffLine("-", body, None, old_no))
                old_no += 1
                old_left -= 1
            else:
                hunk.lines.append(DiffLine(" ", body, new_no, old_no))
                new_no += 1
                old_no += 1
                new_left -= 1
                old_left -= 1

            continue

        hunk = None if not raw.startswith("@@") else hunk

        if raw.startswith("diff --git "):
            match = _GIT_HEADER.match(raw)
            current = start(safe_path(match.group(2)) if match else "")
            continue

        if raw.startswith("--- "):
            source = raw[4:].split("\t")[0].strip().strip('"')

            if current is None or current.hunks:
                current = start("")

            if source != "/dev/null":
                current.old_path = safe_path(_strip_prefix(source))
            else:
                current.status = ADDED

            continue

        if raw.startswith("+++ ") and current is not None:
            target = raw[4:].split("\t")[0].strip().strip('"')

            if target == "/dev/null":
                current.status = REMOVED
                current.path = current.old_path or current.path
            else:
                current.path = safe_path(_strip_prefix(target))

            continue

        if current is None:
            continue

        if raw.startswith("new file mode"):
            current.status = ADDED
        elif raw.startswith("deleted file mode"):
            current.status = REMOVED
        elif raw.startswith("rename from "):
            current.status = RENAMED
            current.old_path = safe_path(raw[len("rename from "):])
        elif raw.startswith("rename to "):
            current.path = safe_path(raw[len("rename to "):])
        elif raw.startswith("Binary files ") or raw.startswith("GIT binary patch"):
            current.binary = True
        elif raw.startswith("@@"):
            match = _HUNK.match(raw)

            if not match:
                raise DiffError(f"unreadable hunk header in {current.path or 'a file'}: {raw[:80]!r}")

            old_no, new_no = int(match.group(1)), int(match.group(3))
            old_left = int(match.group(2)) if match.group(2) is not None else 1
            new_left = int(match.group(4)) if match.group(4) is not None else 1
            hunk = Hunk(header=raw)
            current.hunks.append(hunk)

    files = [item for item in files if item.path]

    if not files:
        raise DiffError("no file sections found - expected a unified diff (git diff / GitHub .diff)")

    for item in files:
        if item.status == MODIFIED and item.old_path and item.old_path != item.path:
            item.status = RENAMED

    return files
