"""Day 34: the project's files, as the file assistant may see and change them.

Everything the assistant does to a file goes through this module - the
``project-files`` MCP server is a thin wrapper around it - so the rules live
in one place:

- **One root.** Paths are relative to the project root (``PROJECT_FILES_ROOT``,
  this repository by default). An absolute path, a ``..`` and a symlink that
  resolves outside the root are refused before anything is opened.
- **Ignore rules.** In a Git repository the files are the ones Git lists -
  tracked files, and untracked ones ``.gitignore`` does not exclude; anywhere
  else, a walk of the tree. Either way the denylist below applies on top:
  ``.git``, virtual environments, caches, build output, generated indexes
  (``PROJECT_FILES_EXCLUDE``) and the assistant's own output folder, so a
  report never becomes evidence for the next one.
- **Secrets are never read.** ``.env`` files, keys, keystores and credential
  files are refused by name, and a file holding a private key is refused by
  content. Search skips both.
- **Limits.** A file over the size limit is refused, not truncated. Results,
  excerpts and listings are capped and say when they were.
- **Writes are narrow.** The output folder (``PROJECT_FILES_OUTPUT_DIR``,
  ``output/`` by default) is the assistant's to write. Any other path needs an
  explicit confirmation, and an existing file also needs the sha256 the change
  was prepared against - so a file edited after it was read is not overwritten.
  Writes are atomic, never go through a symlink, and keep the file's newline
  style and byte-order mark.

Nothing here executes anything it reads. Git is run once, as a fixed argument
list, to list files.
"""

import difflib
import fnmatch
import hashlib
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from app.services.json_store import write_bytes

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_EXCLUDE = ("data/index",)

#: Folder names never listed, searched, read or written, wherever they are.
EXCLUDED_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
        ".ruff_cache", ".tox", ".idea", ".vscode", ".gradle", "build", "dist", ".eggs", "site-packages",
    }
)

#: File names that hold secrets. Refused by name, whatever they contain.
_SECRET_NAME = re.compile(
    r"""^(
        \.env(\..+)?
      | .*\.(pem|key|p12|pfx|jks|keystore|gpg|tfvars)
      | id_(rsa|dsa|ecdsa|ed25519)(\.pub)?
      | \.netrc | \.pypirc | \.npmrc | \.git-credentials
      | .*credential.* | .*secret.* | service[-_]account.*\.json | google-services\.json
    )$""",
    re.IGNORECASE | re.VERBOSE,
)
#: Templates that document a secret's name without holding its value.
_SECRET_TEMPLATE = re.compile(r"^\.env\.(example|sample|template)$", re.IGNORECASE)
_PRIVATE_KEY = re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_BINARY_PROBE = 8192
_MAX_WALK = 20_000


class ProjectFileError(Exception):
    """A file operation was refused. ``kind`` is the short reason a caller
    branches on: invalid_path, outside_root, protected_path, secret_file,
    not_found, not_a_file, binary_file, not_utf8, too_large, invalid_query,
    write_not_authorized, hash_required, hash_mismatch, parent_missing."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message


@dataclass(frozen=True)
class FileLimits:
    max_file_bytes: int = 262_144
    max_results: int = 200
    max_listed: int = 300
    max_excerpt_chars: int = 200

    @classmethod
    def from_env(cls) -> "FileLimits":
        return cls(
            max_file_bytes=_env_int("PROJECT_FILES_MAX_FILE_BYTES", cls.max_file_bytes),
            max_results=_env_int("PROJECT_FILES_MAX_RESULTS", cls.max_results),
        )


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else default


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def is_secret_name(name: str) -> bool:
    return bool(_SECRET_NAME.match(name)) and not _SECRET_TEMPLATE.match(name)


def _inside(path: Path, folder: Path) -> bool:
    return path == folder or folder in path.parents


def _newline_of(text: str) -> str:
    return "\r\n" if text.count("\r\n") * 2 > text.count("\n") else "\n"


class ProjectFiles:
    """The project's files under the rules above."""

    def __init__(
        self,
        root: Path | str | None = None,
        output_dir: str | None = None,
        limits: FileLimits | None = None,
        exclude: tuple[str, ...] | None = None,
    ) -> None:
        self.root = Path(root or os.getenv("PROJECT_FILES_ROOT") or PROJECT_ROOT).resolve()

        if not self.root.is_dir():
            raise ProjectFileError("invalid_path", f"the project root {self.root.name!r} is not a folder")

        self.limits = limits or FileLimits.from_env()
        configured = os.getenv("PROJECT_FILES_EXCLUDE")
        names = exclude if exclude is not None else (configured.split(",") if configured else DEFAULT_EXCLUDE)
        self.exclude = tuple(name.strip().strip("/") for name in names if name.strip().strip("/"))
        output = self._clean(output_dir or os.getenv("PROJECT_FILES_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)
        self._check(output, folder=True)
        self.output_rel = output.as_posix()
        self.output = (self.root / output).resolve()

        if not _inside(self.output, self.root) or self.output == self.root:
            raise ProjectFileError("outside_root", "the output folder must be a folder inside the project root")

    # --- paths --------------------------------------------------------------------

    def _clean(self, path: str) -> PurePosixPath:
        if not isinstance(path, str) or not path.strip():
            raise ProjectFileError("invalid_path", "the path is empty")

        text = path.strip()

        if "\x00" in text or "\\" in text:
            raise ProjectFileError("invalid_path", "the path contains a NUL byte or a backslash")

        pure = PurePosixPath(text)

        if pure.is_absolute() or text.startswith("~"):
            raise ProjectFileError("outside_root", "paths are relative to the project root; absolute paths are refused")

        if ".." in pure.parts:
            raise ProjectFileError("outside_root", "'..' is not allowed in a path")

        parts = [part for part in pure.parts if part != "."]

        if not parts:
            raise ProjectFileError("invalid_path", "the path names the project root itself")

        return PurePosixPath(*parts)

    def _excluded(self, relative: PurePosixPath) -> bool:
        if any(part in EXCLUDED_DIRS or part.startswith(".venv") or part.endswith(".egg-info") for part in relative.parts):
            return True

        text = relative.as_posix()

        return any(text == prefix or text.startswith(prefix + "/") for prefix in self.exclude)

    def _check(self, relative: PurePosixPath, folder: bool = False) -> None:
        if self._excluded(relative):
            raise ProjectFileError("protected_path", f"{relative.as_posix()} is in a protected or ignored folder")

        if not folder and is_secret_name(relative.name):
            raise ProjectFileError("secret_file", f"{relative.as_posix()} may hold secrets and is never read or written")

    def resolve(self, path: str) -> tuple[str, Path]:
        """The relative path and the real file behind it - refused when either
        is outside the root, protected or a secret (a link to .env is .env)."""
        relative = self._clean(path)
        self._check(relative)
        real = (self.root / relative).resolve()

        if not _inside(real, self.root):
            raise ProjectFileError("outside_root", f"{relative.as_posix()} resolves outside the project root")

        target = PurePosixPath(real.relative_to(self.root).as_posix())

        if target != relative:
            self._check(target)

        return relative.as_posix(), real

    def in_output(self, path: str) -> bool:
        _, real = self.resolve(path)
        return _inside(real, self.output)

    def _hidden(self, relative: PurePosixPath, folder: bool = False) -> bool:
        if self._excluded(relative):
            return True

        text = relative.as_posix()

        if text == self.output_rel or text.startswith(self.output_rel + "/"):
            return True

        return not folder and is_secret_name(relative.name)

    # --- listing ------------------------------------------------------------------

    def list_files(self) -> list[str]:
        """Every file the assistant may look at, as sorted relative paths."""
        names = self._git_files()

        if names is None:
            names = list(self._walk())

        kept = set()

        for name in names:
            relative = PurePosixPath(name)

            if self._hidden(relative):
                continue

            path = self.root / relative

            if path.is_symlink():
                # A link is listed only when what it points at could be listed
                # itself: inside the root, not ignored, not a secret.
                target = path.resolve()

                if not _inside(target, self.root) or target == self.root:
                    continue

                if self._hidden(PurePosixPath(target.relative_to(self.root).as_posix())):
                    continue

            if path.is_file():
                kept.add(relative.as_posix())

        return sorted(kept)

    def _git_files(self) -> list[str] | None:
        """What Git lists - tracked files and untracked ones it does not
        ignore - or None outside a Git work tree."""
        if shutil.which("git") is None:
            return None

        try:
            completed = subprocess.run(
                ["git", "-C", str(self.root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                capture_output=True,
                timeout=10,
                check=False,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

        if completed.returncode != 0:
            return None

        return [name for name in completed.stdout.decode("utf-8", "replace").split("\0") if name]

    def _walk(self):
        count = 0

        for directory, folders, files in os.walk(self.root, followlinks=False):
            base = PurePosixPath(Path(directory).relative_to(self.root).as_posix())
            folders[:] = sorted(name for name in folders if not self._hidden(base / name, folder=True))

            for name in sorted(files):
                count += 1

                if count > _MAX_WALK:
                    return

                yield (base / name).as_posix()

    def _scope(self, directory: str) -> list[str]:
        files = self.list_files()

        if not (directory or "").strip():
            return files

        relative = self._clean(directory)
        self._check(relative, folder=True)
        real = (self.root / relative).resolve()

        if not _inside(real, self.root):
            raise ProjectFileError("outside_root", f"{relative.as_posix()} resolves outside the project root")

        if not real.is_dir():
            raise ProjectFileError("not_found", f"there is no folder {relative.as_posix()}")

        prefix = relative.as_posix() + "/"

        return [name for name in files if name.startswith(prefix)]

    @staticmethod
    def _matches(path: str, pattern: str) -> bool:
        if any(char in pattern for char in "*?["):
            return fnmatch.fnmatchcase(path, pattern) or fnmatch.fnmatchcase(PurePosixPath(path).name, pattern)

        return pattern.lower() in path.lower()

    # --- the operations ---------------------------------------------------------------

    def search_files(self, pattern: str, directory: str = "") -> dict:
        """Files whose path matches a glob (``*.md``) or contains a fragment."""
        wanted = (pattern or "").strip()

        if not wanted or len(wanted) > 200:
            raise ProjectFileError("invalid_query", "give a glob or a part of a file name (1-200 characters)")

        found = [name for name in self._scope(directory) if self._matches(name, wanted)]
        cap = self.limits.max_listed

        return {"pattern": wanted, "directory": directory or "", "files": found[:cap], "total": len(found),
                "truncated": len(found) > cap}

    def search_text(self, query: str, file_pattern: str = "", directory: str = "", ignore_case: bool = False) -> dict:
        """Lines that contain ``query`` (literal text, not a regex)."""
        wanted = query if isinstance(query, str) else ""

        if not wanted.strip() or len(wanted) > 200 or "\n" in wanted:
            raise ProjectFileError("invalid_query", "give one line of text to search for (1-200 characters)")

        needle = wanted.lower() if ignore_case else wanted
        files = [name for name in self._scope(directory) if not file_pattern or self._matches(name, file_pattern)]
        matches: list[dict] = []
        matched_files: set[str] = set()
        skipped = {"binary": 0, "too_large": 0, "secret": 0}
        searched = 0
        truncated = False

        for name in files:
            data = self._searchable(self.root / name, skipped)

            if data is None:
                continue

            searched += 1

            for number, line in enumerate(data.splitlines(), start=1):
                at = (line.lower() if ignore_case else line).find(needle)

                if at == -1:
                    continue

                if len(matches) >= self.limits.max_results:
                    truncated = True
                    break

                matches.append({"path": name, "line": number, "excerpt": self._excerpt(line, at, len(needle))})
                matched_files.add(name)

            if truncated:
                break

        return {
            "query": wanted, "ignore_case": bool(ignore_case), "file_pattern": file_pattern or "",
            "directory": directory or "", "matches": matches, "files_searched": searched,
            "files_matched": len(matched_files), "truncated": truncated, "skipped": skipped,
        }

    def _searchable(self, path: Path, skipped: dict) -> str | None:
        """A file's text for searching, or None (counted) when it is too
        large, binary, not UTF-8 or holds a private key."""
        try:
            if path.stat().st_size > self.limits.max_file_bytes:
                skipped["too_large"] += 1
                return None

            data = path.read_bytes()
        except OSError:
            return None

        if b"\x00" in data[:_BINARY_PROBE]:
            skipped["binary"] += 1
            return None

        if _PRIVATE_KEY.search(data):
            skipped["secret"] += 1
            return None

        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            skipped["binary"] += 1
            return None

    def _excerpt(self, line: str, at: int, length: int) -> str:
        cap = self.limits.max_excerpt_chars
        text = line.rstrip("\r")

        if len(text) <= cap:
            return text.strip()

        start = max(0, min(at - cap // 3, len(text) - cap))

        return ("…" if start else "") + text[start : start + cap].strip() + ("…" if start + cap < len(text) else "")

    def _load(self, path: str) -> tuple[str, Path, bytes]:
        relative, real = self.resolve(path)

        if not real.exists():
            raise ProjectFileError("not_found", f"there is no file {relative}")

        if not real.is_file():
            raise ProjectFileError("not_a_file", f"{relative} is not a file")

        if real.stat().st_size > self.limits.max_file_bytes:
            raise ProjectFileError("too_large", f"{relative} is larger than {self.limits.max_file_bytes} bytes")

        data = real.read_bytes()

        if b"\x00" in data[:_BINARY_PROBE]:
            raise ProjectFileError("binary_file", f"{relative} is not a text file")

        if _PRIVATE_KEY.search(data):
            raise ProjectFileError("secret_file", f"{relative} holds a private key and is never read")

        return relative, real, data

    @staticmethod
    def _text(relative: str, data: bytes) -> str:
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ProjectFileError("not_utf8", f"{relative} is not UTF-8 text") from error

    def read(self, path: str, start_line: int = 0, end_line: int = 0) -> dict:
        """A text file, whole or a range of lines (1-based, inclusive). The
        sha256 is always of the whole file, for a later write to check."""
        relative, _, data = self._load(path)
        text = self._text(relative, data)
        lines = text.splitlines()
        first, last = 1, len(lines)

        if start_line or end_line:
            first = max(1, int(start_line or 1))
            last = min(len(lines), int(end_line or len(lines)))

            if first > last:
                raise ProjectFileError("invalid_query", f"{relative} has {len(lines)} lines; the range {start_line}-{end_line} is empty")

            content = "\n".join(lines[first - 1 : last])
        else:
            content = text

        logger.info("[ProjectFiles] read %s lines=%d-%d sha256=%s", relative, first, last, sha256_bytes(data)[:12])

        return {"path": relative, "content": content, "sha256": sha256_bytes(data), "size": len(data),
                "total_lines": len(lines), "start_line": first, "end_line": last, "newline": _newline_of(text)}

    def write(self, path: str, content: str, expected_sha256: str = "", confirmed: bool = False) -> dict:
        """Create or replace a text file, under the write rules above."""
        relative, real = self.resolve(path)

        if not isinstance(content, str):
            raise ProjectFileError("invalid_query", "content must be text")

        if (self.root / relative).is_symlink():
            raise ProjectFileError("protected_path", f"{relative} is a symlink; writes never follow links")

        in_output = _inside(real, self.output)
        exists = real.exists()

        if exists and not real.is_file():
            raise ProjectFileError("not_a_file", f"{relative} is not a file")

        if not in_output and not confirmed:
            raise ProjectFileError(
                "write_not_authorized",
                f"{relative} is outside {self.output_rel}/; changing it needs an explicit confirmation",
            )

        current = self._load(relative)[2] if exists else None

        if current is not None and not in_output and not expected_sha256:
            raise ProjectFileError("hash_required", f"{relative} exists; give the sha256 the change was prepared against")

        if expected_sha256:
            if current is None:
                raise ProjectFileError("hash_mismatch", f"{relative} no longer exists")

            if sha256_bytes(current) != expected_sha256.strip().lower():
                raise ProjectFileError("hash_mismatch", f"{relative} changed after it was read; read it again")

        text = content.replace("\r\n", "\n")
        previous = None

        if current is not None:
            previous = self._text(relative, current)

            if _newline_of(previous) == "\r\n":
                text = text.replace("\n", "\r\n")

        payload = text.encode("utf-8")

        if current is not None and current.startswith(b"\xef\xbb\xbf"):
            payload = b"\xef\xbb\xbf" + payload

        if len(payload) > self.limits.max_file_bytes:
            raise ProjectFileError("too_large", f"the content is larger than {self.limits.max_file_bytes} bytes")

        if not real.parent.exists():
            if not in_output:
                raise ProjectFileError("parent_missing", f"the folder of {relative} does not exist")

            real.parent.mkdir(parents=True, exist_ok=True)

        write_bytes(real, payload, mode=real.stat().st_mode & 0o777 if exists else None)
        digest = sha256_bytes(payload)
        logger.info("[ProjectFiles] wrote %s created=%s bytes=%d sha256=%s", relative, not exists, len(payload), digest[:12])

        return {"path": relative, "sha256": digest, "bytes": len(payload), "created": not exists,
                "previous_sha256": sha256_bytes(current) if current is not None else None, "in_output": in_output}

    def diff(self, path: str, proposed: str) -> dict:
        """A unified diff from the file as it is now to ``proposed``."""
        relative, real = self.resolve(path)
        current, base = "", None

        if real.exists():
            _, _, data = self._load(relative)
            current, base = self._text(relative, data), sha256_bytes(data)

        text = unified_diff(current.replace("\r\n", "\n"), (proposed or "").replace("\r\n", "\n"), relative, base is not None)
        added = sum(1 for line in text.splitlines() if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in text.splitlines() if line.startswith("-") and not line.startswith("---"))

        return {"path": relative, "exists": base is not None, "base_sha256": base, "diff": text,
                "added": added, "removed": removed, "identical": not text}


# --- diffs ------------------------------------------------------------------------------

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_NO_NEWLINE = "\\ No newline at end of file"


def unified_diff(old: str, new: str, path: str, exists: bool = True) -> str:
    """``git apply``-compatible unified diff of two texts; empty when equal."""
    lines = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}" if exists else "/dev/null",
        tofile=f"b/{path}",
        n=3,
    )

    return "".join(line if line.endswith("\n") else f"{line}\n{_NO_NEWLINE}\n" for line in lines)


def apply_unified_diff(original: str, diff: str) -> str:
    """Apply a diff made by unified_diff to ``original``. Raises ValueError
    when a context or removed line does not match - which is how a diff is
    validated: applied to the original, it has to give the proposal back."""
    source = original.splitlines(keepends=True)
    hunks: list[tuple[int, int, list[tuple[str, str]]]] = []

    for line in diff.splitlines(keepends=True):
        match = _HUNK.match(line)

        if match:
            hunks.append((int(match.group(1)), int(match.group(2) or "1"), []))
        elif not hunks:
            continue
        elif line.rstrip("\n") == _NO_NEWLINE:
            tag, body = hunks[-1][2][-1]
            hunks[-1][2][-1] = (tag, body[:-1] if body.endswith("\n") else body)
        elif line[:1] in (" ", "-", "+"):
            hunks[-1][2].append((line[0], line[1:]))
        else:
            raise ValueError(f"not a diff line: {line[:40]!r}")

    result: list[str] = []
    position = 0

    for start, count, body in hunks:
        begin = start - 1 if count else start

        if begin < position:
            raise ValueError("hunks overlap or are out of order")

        result.extend(source[position:begin])
        position = begin

        for tag, text in body:
            if tag == "+":
                result.append(text)
                continue

            if position >= len(source) or source[position] != text:
                raise ValueError(f"line {position + 1} does not match the diff")

            if tag == " ":
                result.append(text)

            position += 1

    result.extend(source[position:])

    return "".join(result)
