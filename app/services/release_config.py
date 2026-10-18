"""Day 35: the release assistant's configuration - which repositories it may
prepare a release of, where its artifacts go, and which checks it may run.

``data/release/release_config.json`` is edited by a person. Nothing the model
says and nothing in a repository being released can add a repository, a
check or an output folder; the API takes names from this file, the CLI also
takes a path. A missing file means no named repositories and no checks, and
is never created here (unlike a loader that writes its defaults - a read of
configuration must not change the repository).
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from app.services.project_files import DEFAULT_OUTPUT_DIR as PROJECT_OUTPUT_DIR
from app.services.project_files import PROJECT_ROOT, ProjectFileError, ProjectFiles

CONFIG_PATH = PROJECT_ROOT / "data" / "release" / "release_config.json"
DEFAULT_OUTPUT_DIR = "output/release"


class ReleaseConfigError(ValueError):
    """The configuration, or a choice made against it, is not usable."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass(frozen=True)
class RepositorySpec:
    key: str
    path: Path
    title: str
    documentation: tuple[str, ...] = ()
    user_facing: tuple[str, ...] = ()
    version_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckSpec:
    id: str
    repository: str
    title: str
    argv: tuple[str, ...]
    documented_in: str = ""
    timeout_seconds: int = 600
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReleaseConfig:
    output_dir: str = DEFAULT_OUTPUT_DIR
    repositories: dict[str, RepositorySpec] = field(default_factory=dict)
    checks: tuple[CheckSpec, ...] = ()

    def repository(self, name: str) -> RepositorySpec:
        """A configured repository by key, or - for the CLI - any existing folder by path."""
        chosen = (name or "").strip() or "backend"

        if chosen in self.repositories:
            return self.repositories[chosen]

        raise ReleaseConfigError(
            "unknown_repository",
            f"{chosen!r} is not a configured repository (configured: {', '.join(sorted(self.repositories)) or 'none'})",
        )

    def checks_for(self, repository: str) -> list[CheckSpec]:
        return [check for check in self.checks if check.repository == repository]


def load_config(path: Path | None = None) -> ReleaseConfig:
    target = path or CONFIG_PATH

    if not target.is_file():
        return ReleaseConfig()

    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseConfigError("invalid_config", f"{target.name} cannot be read: {error}") from error

    base = target.resolve().parents[2] if target == CONFIG_PATH else target.resolve().parent
    repositories = {}

    for key, item in (document.get("repositories") or {}).items():
        repositories[key] = RepositorySpec(
            key=key,
            path=(base / item.get("path", ".")).resolve(),
            title=item.get("title", key),
            documentation=tuple(item.get("documentation", ())),
            user_facing=tuple(item.get("user_facing", ())),
            version_files=tuple(item.get("version_files", ())),
        )

    checks = []

    for item in document.get("checks") or ():
        argv = tuple(str(part) for part in item.get("argv", ()))

        if not item.get("id") or not argv:
            raise ReleaseConfigError("invalid_config", "every check needs an id and an argv list")

        checks.append(
            CheckSpec(
                id=item["id"],
                repository=item.get("repository", "backend"),
                title=item.get("title", item["id"]),
                argv=argv,
                documented_in=item.get("documented_in", ""),
                timeout_seconds=int(item.get("timeout_seconds", 600)),
                requires=tuple(item.get("requires", ())),
            )
        )

    return ReleaseConfig(
        output_dir=str(document.get("output_dir") or DEFAULT_OUTPUT_DIR), repositories=repositories, checks=tuple(checks)
    )


def output_dir_for(config: ReleaseConfig, override: str = "") -> str:
    """The folder the artifacts go in: the caller's choice, else
    ``RELEASE_OUTPUT_DIR``, else the configuration's. It must be inside the
    project's output folder (``output/``): the file tools' rules are checked
    here before any server is started, and a release run never gets to write
    into a source, documentation or data folder."""
    chosen = (override or os.getenv("RELEASE_OUTPUT_DIR") or config.output_dir or DEFAULT_OUTPUT_DIR).strip()

    try:
        resolved = ProjectFiles(root=PROJECT_ROOT, output_dir=chosen).output_rel
    except ProjectFileError as error:
        raise ReleaseConfigError(error.kind, f"output directory {chosen!r} refused: {error.message}") from error

    if resolved != PROJECT_OUTPUT_DIR and not resolved.startswith(PROJECT_OUTPUT_DIR + "/"):
        raise ReleaseConfigError(
            "outside_output", f"output directory {chosen!r} refused: release artifacts go inside {PROJECT_OUTPUT_DIR}/"
        )

    return resolved
