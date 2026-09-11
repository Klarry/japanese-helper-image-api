"""Sticky Facts: a small key-value memory instead of a summary.

The summary strategy compresses the conversation by rewriting it as prose.
This one does something different on purpose: it keeps only the handful of
things that stay true later - the learner's goals, constraints, preferences
and decisions - as ``key: value`` pairs, and sends those alongside the newest
messages. Nothing here writes or reads a summary.

The extraction runs through the same gemini_service every other feature
uses; there is no second path to the LLM.
"""

import json
import logging
from typing import Any, NamedTuple

from app.core.config import AGENT_FACTS_LIMIT
from app.services.gemini_service import generate_text_with_usage

logger = logging.getLogger(__name__)

_MAX_VALUE_LENGTH = 200

_FACTS_INSTRUCTIONS = (
    "You keep a small key-value memory of a Japanese-learning conversation, so a "
    "tutor can be reminded of what matters without replaying the whole transcript. "
    "Update that memory from the learner's newest message.\n\n"
    "Keep only what stays useful later: the learner's goals, constraints, "
    "preferences, level, and decisions they have already made. Merge new "
    "information into keys that already exist rather than adding near-duplicates; "
    "drop a fact only when the new message contradicts it; never record anything "
    "the conversation does not state.\n\n"
    "Answer with a JSON object and nothing else - no prose, no code fences: at most "
    "{limit} entries, snake_case keys, short plain-text values, no nesting."
)


class FactsUpdate(NamedTuple):
    """The refreshed memory, and what refreshing it cost - so the strategy's
    own token spend is visible in the comparison instead of hiding inside it."""

    facts: dict[str, str]
    tokens_used: int


class FactsExtractor:
    """Rewrites the key-value memory from the learner's newest message."""

    def __init__(self, facts_limit: int = AGENT_FACTS_LIMIT) -> None:
        self._facts_limit = facts_limit

    async def update(self, facts: dict[str, str], message: str) -> FactsUpdate:
        """Ask Gemini for the updated memory.

        Raises if Gemini fails or answers with something that is not a flat
        JSON object - the caller decides what to do about it, and keeps the
        previous facts rather than storing a half-parsed mess.
        """
        generated = await generate_text_with_usage(self._build_prompt(facts, message))

        return FactsUpdate(
            facts=self._parse(generated.text),
            tokens_used=(generated.input_tokens or 0) + (generated.output_tokens or 0),
        )

    def _build_prompt(self, facts: dict[str, str], message: str) -> str:
        current = json.dumps(facts, ensure_ascii=False, indent=2) if facts else "{}"

        return (
            f"{_FACTS_INSTRUCTIONS.format(limit=self._facts_limit)}\n\n"
            f"Current memory:\n{current}\n\n"
            f"Learner's newest message:\n{message}"
        )

    def _parse(self, text: str) -> dict[str, str]:
        try:
            parsed = json.loads(self._strip_code_fence(text))
        except json.JSONDecodeError as error:
            raise ValueError(f"facts update was not valid JSON: {error}") from error

        if not isinstance(parsed, dict):
            raise ValueError("facts update was not a JSON object")

        facts: dict[str, str] = {}

        for key, value in parsed.items():
            if not isinstance(key, str) or isinstance(value, (dict, list)):
                continue

            facts[key] = str(value)[:_MAX_VALUE_LENGTH]

            if len(facts) == self._facts_limit:
                break

        return facts

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Models wrap JSON in ```json fences often enough to be worth
        handling here rather than losing the whole update to it."""
        stripped = text.strip()

        if not stripped.startswith("```"):
            return stripped

        without_opening = stripped.split("\n", 1)[-1]
        return without_opening.rsplit("```", 1)[0].strip()
