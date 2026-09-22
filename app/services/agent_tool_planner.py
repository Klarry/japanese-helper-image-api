"""Day 17: deciding whether a message needs a tool, and which one.

MCP tells the agent what tools exist - names, descriptions, argument
schemas, straight from list_tools(). It does not say when to use them. That
is a judgement about what the learner asked, so the model makes it, through
the same gemini_service every other part of the agent uses: there is no
second path to the LLM here, and no provider-specific function-calling
format either. The model is shown the tools exactly as the server described
them and answers with the calls it wants, as JSON.

The planner only *proposes*. Calls to tools the server does not have are
dropped here, and the arguments are checked by the server itself against its
own schema - a confused plan can waste a lookup, never do something the
server did not offer.
"""

import json
import logging
from collections.abc import Sequence
from typing import Any, NamedTuple

from app.services.gemini_service import generate_text_with_usage
from app.services.mcp_client import McpToolInfo

logger = logging.getLogger(__name__)

# One message rarely needs more; a message comparing two words needs two.
MAX_CALLS = 3

_PLANNER_INSTRUCTIONS = (
    "You decide whether answering a Japanese learner's message needs data from the tools "
    "below. The tools look things up in a real dictionary, and your own memory is not a "
    "substitute when the learner asks about a specific word.\n\n"
    "Rules:\n"
    "- Call a tool when the message asks about one or more specific Japanese words or "
    "kanji: their meaning, translation, reading, romaji or JLPT level.\n"
    "- Pass each word exactly as it is written in Japanese in the message - never romaji, "
    "never a translation. If the message refers back to a word from the recent "
    "conversation ('this word', 'it'), pass that word.\n"
    "- Do not call a tool for grammar explanations, exercises, small talk, or anything "
    "that is not about a specific word.\n"
    f"- At most {MAX_CALLS} calls.\n\n"
    "Answer with a JSON object and nothing else - no prose, no code fences:\n"
    '{"calls": [{"tool": "<tool name>", "arguments": {...}}]}\n'
    'Use {"calls": []} when no tool is needed.'
)


class PlannedCall(NamedTuple):
    tool: str
    arguments: dict[str, Any]


class ToolPlan(NamedTuple):
    """What the planner proposes, and what proposing it cost."""

    calls: tuple[PlannedCall, ...]
    tokens_used: int


class ToolPlanner:
    """Proposes the tool calls a message needs."""

    async def plan(
        self,
        message: str,
        tools: Sequence[McpToolInfo],
        recent: Sequence[dict[str, str]] = (),
    ) -> ToolPlan:
        """Ask Gemini which tools, if any, to call for ``message``.

        No tools means no call to Gemini at all. Raises if Gemini fails or
        answers with something that is not a plan - the caller then simply
        answers without tools.
        """
        if not tools:
            return ToolPlan((), 0)

        generated = await generate_text_with_usage(self._build_prompt(message, tools, recent))
        calls = self._calls(self._parse(generated.text), {tool.name for tool in tools})

        return ToolPlan(
            calls=calls,
            tokens_used=(generated.input_tokens or 0) + (generated.output_tokens or 0),
        )

    @staticmethod
    def _build_prompt(message: str, tools: Sequence[McpToolInfo], recent: Sequence[dict[str, str]]) -> str:
        catalogue = "\n".join(
            f"- {tool.name}: {tool.description}\n"
            f"  arguments (JSON schema): {json.dumps(tool.input_schema, ensure_ascii=False)}"
            for tool in tools
        )
        parts = [_PLANNER_INSTRUCTIONS, f"Tools:\n{catalogue}"]

        if recent:
            transcript = "\n".join(f"{turn['role']}: {turn['content'][:400]}" for turn in recent)
            parts.append(f"Recent conversation (only to resolve references):\n{transcript}")

        parts.append(f"Learner's message:\n{message}")

        return "\n\n".join(parts)

    @staticmethod
    def _calls(parsed: dict[str, Any], known: set[str]) -> tuple[PlannedCall, ...]:
        raw = parsed.get("calls")

        if not isinstance(raw, list):
            raise ValueError("tool plan has no 'calls' list")

        calls: list[PlannedCall] = []

        for item in raw:
            if not isinstance(item, dict):
                continue

            tool = item.get("tool")
            arguments = item.get("arguments")

            if tool not in known:
                logger.warning("Dropping a planned call to unknown tool %r", tool)
                continue

            if not isinstance(arguments, dict):
                logger.warning("Dropping a planned call to %s without an arguments object", tool)
                continue

            calls.append(PlannedCall(tool, arguments))

        return tuple(calls[:MAX_CALLS])

    def _parse(self, text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(self._strip_code_fence(text))
        except json.JSONDecodeError as error:
            raise ValueError(f"tool plan was not valid JSON: {error}") from error

        if not isinstance(parsed, dict):
            raise ValueError("tool plan was not a JSON object")

        return parsed

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()

        if not stripped.startswith("```"):
            return stripped

        without_opening = stripped.split("\n", 1)[-1]
        return without_opening.rsplit("```", 1)[0].strip()
