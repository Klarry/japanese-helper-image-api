"""Deciding which memory layer a message belongs in.

Short-term needs no decision: every message is the conversation, and is
already stored as one. The question is only whether a message also said
something that should outlive the next few turns, and if so, whether it
belongs to the task being worked on (working memory) or to the learner
(long-term memory).

That is a judgement about meaning, so it is made by the model - through the
same gemini_service every other feature uses, with no second path to the
LLM. The router returns the layers it changed, in full; a layer it does not
mention is left exactly as it was, so an ordinary question ("explain this
grammar") writes nothing anywhere.
"""

import json
import logging
from typing import Any, NamedTuple

from app.core.config import AGENT_MEMORY_ENTRY_LIMIT
from app.services.agent_memory import (
    LongTermMemory,
    WorkingMemory,
    long_term_memory_as_json,
    long_term_memory_from_json,
    working_memory_as_json,
    working_memory_from_json,
)
from app.services.gemini_service import generate_text_with_usage

logger = logging.getLogger(__name__)

_ROUTER_INSTRUCTIONS = (
    "You maintain the memory of a Japanese-learning assistant. The memory has "
    "three layers, and they are kept apart on purpose:\n\n"
    "- short-term: the current conversation. Handled automatically - never write to it.\n"
    "- working: the task being worked on right now. Its goals, requirements, "
    "constraints and the decisions already taken for it. Forgotten when the task "
    "is over.\n"
    "- long-term: the learner themselves. Profile, lasting preferences, important "
    "decisions and knowledge that will still be true in another conversation.\n\n"
    "Decide what the learner's newest message adds, if anything.\n\n"
    "Rules:\n"
    "- Wording decides the layer. \"remember for a long time\", \"remember about me\", "
    "\"from now on\" means long-term; \"for the current task\", \"for this exercise\", "
    "\"for now\" means working.\n"
    "- A request that only asks for help - explain this, make an example, translate - "
    "adds nothing. Return an empty object.\n"
    "- Never move something between layers on your own, and never copy the same "
    "item into both.\n"
    "- Record only what the message states, in the learner's own language.\n\n"
    "Answer with a JSON object and nothing else - no prose, no code fences. Include "
    "a \"working\" key only if working memory changes, and a \"long_term\" key only if "
    "long-term memory changes; give the full updated layer for each key you include, "
    "merging the new information into what is already there. At most {limit} entries "
    "per list, short plain-text values.\n\n"
    "working: {{\"goals\": [], \"requirements\": [], \"constraints\": [], \"decisions\": []}}\n"
    "long_term: {{\"profile\": {{}}, \"preferences\": [], \"decisions\": [], \"knowledge\": []}}"
)


class MemoryRouting(NamedTuple):
    """What the router decided. ``None`` means "this layer does not change",
    which is different from an empty layer - a layer is only ever emptied
    deliberately, through the memory API."""

    working: WorkingMemory | None
    long_term: LongTermMemory | None
    tokens_used: int

    @property
    def layers_written(self) -> list[str]:
        written = []

        if self.working is not None:
            written.append("working")

        if self.long_term is not None:
            written.append("long_term")

        return written


class MemoryRouter:
    """Routes one message into working and/or long-term memory."""

    def __init__(self, entry_limit: int = AGENT_MEMORY_ENTRY_LIMIT) -> None:
        self._entry_limit = entry_limit

    async def route(
        self,
        message: str,
        working: WorkingMemory,
        long_term: LongTermMemory,
    ) -> MemoryRouting:
        """Ask Gemini which layers this message changes, and how.

        Raises if Gemini fails or answers with something that is not a JSON
        object - the caller keeps the memory it already had rather than
        storing a half-parsed one.
        """
        generated = await generate_text_with_usage(self._build_prompt(message, working, long_term))
        parsed = self._parse(generated.text)

        return MemoryRouting(
            working=working_memory_from_json(parsed["working"]) if "working" in parsed else None,
            long_term=long_term_memory_from_json(parsed["long_term"]) if "long_term" in parsed else None,
            tokens_used=(generated.input_tokens or 0) + (generated.output_tokens or 0),
        )

    def _build_prompt(self, message: str, working: WorkingMemory, long_term: LongTermMemory) -> str:
        return (
            f"{_ROUTER_INSTRUCTIONS.format(limit=self._entry_limit)}\n\n"
            f"Current working memory:\n{json.dumps(working_memory_as_json(working), ensure_ascii=False)}\n\n"
            f"Current long-term memory:\n"
            f"{json.dumps(long_term_memory_as_json(long_term), ensure_ascii=False)}\n\n"
            f"Learner's newest message:\n{message}"
        )

    def _parse(self, text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(self._strip_code_fence(text))
        except json.JSONDecodeError as error:
            raise ValueError(f"memory routing was not valid JSON: {error}") from error

        if not isinstance(parsed, dict):
            raise ValueError("memory routing was not a JSON object")

        return {key: parsed[key] for key in ("working", "long_term") if isinstance(parsed.get(key), dict)}

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()

        if not stripped.startswith("```"):
            return stripped

        without_opening = stripped.split("\n", 1)[-1]
        return without_opening.rsplit("```", 1)[0].strip()
