"""Day 35: run the checks a person configured, and report what happened.

A release is only as ready as its checks say, so this module is strict about
three things:

- **what runs.** Only a check listed in ``data/release/release_config.json``,
  only as the argument list written there (``{python}`` becomes the Python
  that runs the assistant), never through a shell, never with text the model
  or the repository supplied. Running a repository's tests does run the
  repository's test code - that is why a check is opt-in, and why the list
  is a file a person edits;
- **where.** In the repository being released, with a small environment:
  the search path, the home folder and the Java/Android/Gradle variables a
  build needs - no API keys, no tokens;
- **what is reported.** The real exit status, how long it took, the tail of
  its output and the lines that look like errors. A check that did not
  start is ``could_not_run`` with the reason; one that started and was
  stopped is ``timed_out``; one the caller did not ask for is ``not_run``.
  None of these is ever reported as passed. Nothing here edits production
  code to make a check pass.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.services.release_config import CheckSpec
from app.services.release_safety import redact

PASSED = "passed"
FAILED = "failed"
TIMED_OUT = "timed_out"
COULD_NOT_RUN = "could_not_run"
NOT_RUN = "not_run"

TAIL_CHARS = 2_500
MAX_ERROR_LINES = 12
ERROR_LINE_CHARS = 220

#: The only variables a check inherits. Proxies are included so a build can
#: download its dependencies; credentials are not.
ENVIRONMENT = (
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "TERM",
    "JAVA_HOME", "ANDROID_HOME", "ANDROID_SDK_ROOT", "GRADLE_USER_HOME", "GRADLE_OPTS", "VIRTUAL_ENV",
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy", "SSL_CERT_FILE",
)

_ERROR_LINE = re.compile(r"(?i)(\w*error\b|\bfailed\b|\bfailure\b|\w*exception\b|traceback|e: file://)")
_MISSING_MODULE = re.compile(r"No module named '?([\w.]+)'?")


@dataclass
class CheckResult:
    id: str
    title: str
    status: str
    command: str
    documented_in: str = ""
    exit_code: int | None = None
    seconds: float = 0.0
    reason: str = ""
    output_tail: str = ""
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "command": self.command,
            "documented_in": self.documented_in,
            "exit_code": self.exit_code,
            "seconds": round(self.seconds, 1),
            "reason": self.reason,
            "output_tail": self.output_tail,
            "errors": self.errors,
        }


def display_command(spec: CheckSpec) -> str:
    """The command as a person would type it, without this machine's interpreter path."""
    return " ".join(shlex.quote("python" if part == "{python}" else part) for part in spec.argv)


def check_environment() -> dict[str, str]:
    environment = {name: os.environ[name] for name in ENVIRONMENT if name in os.environ}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    return environment


def _result(spec: CheckSpec, status: str, **values) -> CheckResult:
    return CheckResult(id=spec.id, title=spec.title, status=status, command=display_command(spec), documented_in=spec.documented_in, **values)


def not_run(spec: CheckSpec, reason: str) -> CheckResult:
    return _result(spec, NOT_RUN, reason=reason)


def _executable(argv0: str, root: Path) -> str | None:
    """The program to start, or None when it cannot be started."""
    if argv0 == "{python}":
        return sys.executable

    if argv0.startswith("./"):
        path = root / argv0[2:]
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None

    return shutil.which(argv0)


def _tail(text: str) -> str:
    return redact(text.strip()[-TAIL_CHARS:])


def _errors(text: str) -> list[str]:
    lines = []

    for line in text.splitlines():
        if _ERROR_LINE.search(line) and line.strip():
            lines.append(redact(line.strip())[:ERROR_LINE_CHARS])

        if len(lines) >= MAX_ERROR_LINES:
            break

    return lines


def run_check(spec: CheckSpec, root: Path) -> CheckResult:
    """Run one check in ``root`` and say what happened. Blocking: call it in a thread."""
    for marker in spec.requires:
        if not (root / marker).exists():
            return _result(spec, COULD_NOT_RUN, reason=f"required path {marker!r} does not exist in this repository")

    program = _executable(spec.argv[0], root)

    if program is None:
        name = spec.argv[0]
        why = "is not executable (chmod +x gradlew)" if name.startswith("./") and (root / name[2:]).is_file() else "was not found"
        return _result(spec, COULD_NOT_RUN, reason=f"{name} {why}")

    argv = [program, *(sys.executable if part == "{python}" else part for part in spec.argv[1:])]
    started = time.perf_counter()

    try:
        completed = subprocess.run(
            argv,
            cwd=str(root),
            env=check_environment(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=spec.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or b"")
        output = output.decode("utf-8", "replace") if isinstance(output, bytes) else str(output)
        return _result(
            spec,
            TIMED_OUT,
            seconds=time.perf_counter() - started,
            reason=f"stopped after {spec.timeout_seconds} s",
            output_tail=_tail(output),
        )
    except OSError as error:
        return _result(spec, COULD_NOT_RUN, reason=f"could not be started: {error.strerror or error}")

    seconds = time.perf_counter() - started
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")

    if completed.returncode == 0:
        return _result(spec, PASSED, exit_code=0, seconds=seconds, output_tail=_tail(output))

    missing = _MISSING_MODULE.search(output)

    if missing and completed.returncode in (1, 2):
        # The command did not get as far as running the project's checks.
        return _result(
            spec,
            COULD_NOT_RUN,
            exit_code=completed.returncode,
            seconds=seconds,
            reason=f"a Python module the check needs is not installed: {missing.group(1)}",
            output_tail=_tail(output),
        )

    return _result(
        spec,
        FAILED,
        exit_code=completed.returncode,
        seconds=seconds,
        output_tail=_tail(output),
        errors=_errors(output),
    )
