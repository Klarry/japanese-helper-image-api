"""The small JSON files this project keeps on disk, read and written in one
place.

Every store here wants the same two things. A write another process can
never catch half-finished: the text goes to a temporary file first and is
moved into place with ``os.replace``, which is atomic, so a reader either
sees the old file or the new one and never a truncated one. And a read that
treats a missing or unreadable file as "nothing yet" rather than an error,
because that is what it means on the first run and after a disk that went
wrong - neither is worth failing a request over.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_json(path: Path, data: dict[str, Any]) -> int:
    """Write ``data`` to ``path``, atomically. Returns the bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)

    return len(payload.encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path``; an empty dict when there is nothing
    readable there."""
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Could not read %s: %s", path, error)
        return {}

    return data if isinstance(data, dict) else {}
