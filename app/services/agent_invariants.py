"""Day 14: the rules the agent is not allowed to break.

The memory layers say what was said, the profile says how to answer, the
task state says where the work has got to. None of them says what must
never happen. "Move the Gemini call into the app" is a perfectly coherent
suggestion - it is just not allowed in this project, and an agent that does
not know that will cheerfully help you do it.

So invariants are their own layer: a small set of project rules in four
categories, kept in their own file apart from every memory layer and from
the conversation, and put in front of the model on every single request.
They are not memory - nothing in a conversation writes them, and clearing a
conversation cannot forget them. They are changed deliberately, through the
API, because that is what a rule is.

The section carries the protocol as well as the rules: when a request would
break one, the agent names the rule, explains why it exists, and offers a
permitted alternative if there is one - rather than quietly helping.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import AGENT_INVARIANTS_FILE_PATH
from app.schemas.agent import InvariantCategory

logger = logging.getLogger(__name__)

_MAX_RULE_LENGTH = 300


_CATEGORY_TITLES = {
    InvariantCategory.ARCHITECTURE: "Architecture",
    InvariantCategory.TECHNOLOGY_STACK: "Technology stack",
    InvariantCategory.TECHNICAL_DECISIONS: "Technical decisions",
    InvariantCategory.BUSINESS_RULES: "Business rules",
}

_INVARIANTS_CAPTION = (
    "INVARIANTS (rules of this project that must never be broken, whatever is asked). "
    "If the request would break one, do not propose it and do not help work around it. "
    "Instead: 1) name the rule it conflicts with, 2) explain why the rule exists, "
    "3) offer an alternative that respects the rules, if one exists."
)


@dataclass(frozen=True)
class Invariant:
    """One rule, with an id so it can be changed or removed on its own."""

    id: str
    category: InvariantCategory
    rule: str


# The rules this project actually runs under. They are written to the file
# the first time it is read, so a fresh install starts with them in place
# and a rule deleted afterwards stays deleted.
PROJECT_INVARIANTS: tuple[Invariant, ...] = (
    Invariant("arch-layers", InvariantCategory.ARCHITECTURE, "Android layering: ViewModel -> Repository -> API"),
    Invariant("stack-backend", InvariantCategory.TECHNOLOGY_STACK, "Backend: FastAPI"),
    Invariant("stack-llm", InvariantCategory.TECHNOLOGY_STACK, "LLM: Gemini"),
    Invariant("stack-storage", InvariantCategory.TECHNOLOGY_STACK, "Storage: JSON files"),
    Invariant("stack-android", InvariantCategory.TECHNOLOGY_STACK, "Android: Kotlin"),
    Invariant(
        "decision-llm-via-backend",
        InvariantCategory.TECHNICAL_DECISIONS,
        "The LLM API is called only from the backend, never from Android",
    ),
    Invariant("decision-no-sqlite", InvariantCategory.TECHNICAL_DECISIONS, "SQLite is not used"),
    Invariant(
        "decision-single-llm",
        InvariantCategory.TECHNICAL_DECISIONS,
        "No new LLM integration is added; everything goes through the existing gemini_service",
    ),
)


def _as_text(value: Any, limit: int = _MAX_RULE_LENGTH) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def invariant_from_json(value: Any) -> Invariant | None:
    if not isinstance(value, dict):
        return None

    identifier = _as_text(value.get("id"), limit=80)
    rule = _as_text(value.get("rule"))

    if not identifier or not rule:
        return None

    try:
        category = InvariantCategory(value.get("category"))
    except ValueError:
        logger.warning("Unknown invariant category %r in %r; skipping it", value.get("category"), identifier)
        return None

    return Invariant(id=identifier, category=category, rule=rule)


def invariant_as_json(invariant: Invariant) -> dict[str, str]:
    return {"id": invariant.id, "category": invariant.category.value, "rule": invariant.rule}


def as_section(invariants: list[Invariant]) -> str:
    """The rules as the model reads them, grouped by category.

    Empty when there are no rules at all, so an agent without invariants
    gets exactly the prompt it got before this layer existed.
    """
    if not invariants:
        return ""

    lines = []

    for category in InvariantCategory:
        rules = [item for item in invariants if item.category is category]

        if not rules:
            continue

        lines.append(f"{_CATEGORY_TITLES[category]}:")
        lines += [f"- {item.rule}" for item in rules]

    return f"{_INVARIANTS_CAPTION}\n" + "\n".join(lines)


class AgentInvariantsStorage:
    """The invariants on disk, in their own file.

    Same fail-safe rules as the other stores: a broken or unreadable file
    reads as "no rules" rather than raising. It differs in one way - a file
    that is not there yet is seeded with the project's own rules and saved,
    so a fresh install starts out constrained rather than unconstrained,
    while a rule deleted later stays deleted.
    """

    def __init__(self, file_path: str = AGENT_INVARIANTS_FILE_PATH) -> None:
        self._path = Path(file_path)

    def load(self) -> list[Invariant]:
        if not self._path.exists():
            seeded = list(PROJECT_INVARIANTS)
            self.save(seeded)
            return seeded

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read invariants from %s: %s", self._path, exc)
            return []

        raw = data.get("invariants") if isinstance(data, dict) else None

        if not isinstance(raw, list):
            logger.warning("Invariants at %s are malformed; reading as none", self._path)
            return []

        parsed = [invariant_from_json(entry) for entry in raw]

        return [item for item in parsed if item is not None]

    def save(self, invariants: list[Invariant]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                {"invariants": [invariant_as_json(item) for item in invariants]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


invariants_storage = AgentInvariantsStorage()
