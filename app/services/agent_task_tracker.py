"""Deciding where a task has got to.

The state machine (agent_task_state) says which moves are legal. It does
not say when to make one - that is a judgement about what the learner just
asked for, so the model makes it, through the same gemini_service every
other feature uses. There is no second path to the LLM here.

The tracker only ever *proposes*: it answers with a stage, the two step
fields and what it noticed about the stage's own work - a plan the learner
approved, a check that came out right. The state machine decides whether the
move is allowed. A proposal to skip a stage, or to start executing a plan
nobody approved, is refused there, not here, so a confused model can never
put the task in an impossible place.
"""

import json
import logging
from typing import Any, NamedTuple

from app.services.agent_task_state import (
    TaskStage,
    TaskState,
    allowed_next,
    as_stage,
    next_requirement,
)
from app.services.gemini_service import generate_text_with_usage

logger = logging.getLogger(__name__)

_TRACKER_INSTRUCTIONS = (
    "You track how far a Japanese-learning task has got. A task moves through four "
    "stages, strictly in this order:\n\n"
    "- planning: working out what the task is and how it will be done.\n"
    "- execution: actually doing the work, step by step.\n"
    "- validation: checking the result - reviewing, correcting, testing understanding.\n"
    "- done: the task is finished.\n\n"
    "Rules:\n"
    "- A stage is never skipped and never goes backwards. From the current stage you "
    "may only stay where you are or move to one of: {allowed}.\n"
    "- Order is not the only condition. Execution may only begin once the learner has "
    "approved a plan, and done may only be reached once validation has passed.{requirement}\n"
    "- Most messages keep the stage and only move the step forward. Move to the next "
    "stage only when the learner's message actually calls for it.\n"
    "- If no task is running and the message does not start one (an ordinary question, "
    "a word lookup, small talk), stay idle.\n"
    "- current_step: what is being worked on right now, one short phrase.\n"
    "- expected_action: what should happen next, one short phrase.\n"
    "- plan: while planning, and only if this very message approves the plan under "
    "discussion, the agreed plan in one short line. Empty otherwise.\n"
    "- validation_passed: while validating, true if this message confirms the result "
    "is right, false if it says it is not. null otherwise.\n\n"
    "Answer with a JSON object and nothing else - no prose, no code fences:\n"
    '{{"task_stage": "...", "current_step": "...", "expected_action": "...", '
    '"plan": "", "validation_passed": null}}'
)


class TaskUpdate(NamedTuple):
    """What the tracker proposes, and what proposing it cost.

    ``plan`` and ``validation_passed`` are evidence, not decisions: the state
    machine accepts them only in the stage that produces them, and only then
    considers the move they would unlock.
    """

    stage: TaskStage
    current_step: str
    expected_action: str
    tokens_used: int
    plan: str = ""
    validation_passed: bool | None = None


class TaskTracker:
    """Proposes where the task stands after the learner's newest message."""

    async def track(self, message: str, state: TaskState) -> TaskUpdate:
        """Ask Gemini where the task is now.

        Raises if Gemini fails or answers with something that is not a JSON
        object - the caller keeps the state it already had rather than
        storing a half-parsed one.
        """
        generated = await generate_text_with_usage(self._build_prompt(message, state))
        parsed = self._parse(generated.text)

        validation_passed = parsed.get("validation_passed")

        return TaskUpdate(
            stage=as_stage(parsed.get("task_stage"), fallback=state.stage),
            current_step=self._text(parsed.get("current_step")),
            expected_action=self._text(parsed.get("expected_action")),
            tokens_used=(generated.input_tokens or 0) + (generated.output_tokens or 0),
            plan=self._text(parsed.get("plan")),
            validation_passed=validation_passed if isinstance(validation_passed, bool) else None,
        )

    @staticmethod
    def _build_prompt(message: str, state: TaskState) -> str:
        current = json.dumps(
            {
                "task_stage": state.stage.value,
                "current_step": state.current_step,
                "expected_action": state.expected_action,
                "plan": state.plan,
                "validation_passed": state.validation_passed,
            },
            ensure_ascii=False,
        )
        allowed = ", ".join(allowed_next(state.stage)) or "nothing (the task is finished)"
        missing = next_requirement(state)
        requirement = f" Right now {missing}, so that move is not available yet." if missing else ""

        return (
            f"{_TRACKER_INSTRUCTIONS.format(allowed=allowed, requirement=requirement)}\n\n"
            f"Current task state:\n{current}\n\n"
            f"Learner's newest message:\n{message}"
        )

    @staticmethod
    def _text(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _parse(self, text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(self._strip_code_fence(text))
        except json.JSONDecodeError as error:
            raise ValueError(f"task state update was not valid JSON: {error}") from error

        if not isinstance(parsed, dict):
            raise ValueError("task state update was not a JSON object")

        return parsed

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()

        if not stripped.startswith("```"):
            return stripped

        without_opening = stripped.split("\n", 1)[-1]
        return without_opening.rsplit("```", 1)[0].strip()
