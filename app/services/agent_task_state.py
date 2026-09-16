"""Day 13: where a task is, as a state machine.

Memory says what was said, the profile says how to answer - neither says
what the learner and the agent are in the middle of. A task has stages, and
they only ever go one way:

    planning -> execution -> validation -> done

Only those three moves are legal, plus starting a task (idle -> planning)
and staying where you are. Everything else is refused: no skipping
validation, no going back to planning halfway through execution, and
nothing at all after done except clearing the task. The machine is what
makes "continue from where we stopped" mean something definite - after a
restart the agent does not re-plan a task it was already executing.

Alongside the stage, two fields say where inside it the task is:
``current_step`` (what is being done now) and ``expected_action`` (what
should happen next). All three travel with the conversation in the same
file the branches live in - the task belongs to the conversation, so
clearing the conversation ends the task too.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

_MAX_VALUE_LENGTH = 300


class TaskStage(str, Enum):
    """The stages a task moves through. ``IDLE`` is the absence of a task -
    not a stage the learner is ever in the middle of, which is why it is the
    only stage a task can be started from and the one clearing returns to."""

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"


# The whole state machine, in one place. A stage maps to the stages it may
# move to next; staying in the same stage is always allowed and is not
# listed here (a task usually takes several messages per stage).
ALLOWED_TRANSITIONS: dict[TaskStage, frozenset[TaskStage]] = {
    TaskStage.IDLE: frozenset({TaskStage.PLANNING}),
    TaskStage.PLANNING: frozenset({TaskStage.EXECUTION}),
    TaskStage.EXECUTION: frozenset({TaskStage.VALIDATION}),
    TaskStage.VALIDATION: frozenset({TaskStage.DONE}),
    TaskStage.DONE: frozenset(),
}


def can_transition(current: TaskStage, proposed: TaskStage) -> bool:
    """Whether the task may move from ``current`` to ``proposed``.

    Staying put is always fine; every other move has to be in the table.
    """
    if proposed is current:
        return True

    return proposed in ALLOWED_TRANSITIONS[current]


def allowed_next(current: TaskStage) -> list[str]:
    return sorted(stage.value for stage in ALLOWED_TRANSITIONS[current])


@dataclass
class TaskState:
    """The task in progress: which stage, what is being done in it, and what
    is expected next."""

    stage: TaskStage = TaskStage.IDLE
    current_step: str = ""
    expected_action: str = ""

    def is_active(self) -> bool:
        return self.stage is not TaskStage.IDLE

    def as_section(self) -> str:
        """The task state as the model reads it. Empty while no task is
        running, so an ordinary question is unaffected by this feature."""
        if not self.is_active():
            return ""

        lines = [f"- Stage: {self.stage.value}"]

        if self.current_step:
            lines.append(f"- Current step: {self.current_step}")

        if self.expected_action:
            lines.append(f"- Expected next action: {self.expected_action}")

        return (
            "TASK STATE (a task is already in progress - continue it from this exact "
            "step; do not start it over and do not repeat explanations already given):\n"
            + "\n".join(lines)
        )

    def with_update(
        self,
        stage: TaskStage,
        current_step: str,
        expected_action: str,
    ) -> "TaskState":
        """The state after a proposed update, or this state unchanged when
        the move is not a legal one.

        An illegal move is refused whole rather than in part: a proposal to
        jump from planning straight to done describes a task that finished,
        and keeping its step text while refusing its stage would leave the
        two contradicting each other.
        """
        if not can_transition(self.stage, stage):
            logger.warning(
                "Refusing task transition %s -> %s; keeping the current state",
                self.stage.value,
                stage.value,
            )
            return self

        return TaskState(
            stage=stage,
            current_step=current_step[:_MAX_VALUE_LENGTH],
            expected_action=expected_action[:_MAX_VALUE_LENGTH],
        )


def _as_text(value: Any) -> str:
    return value.strip()[:_MAX_VALUE_LENGTH] if isinstance(value, str) else ""


def as_stage(value: Any, fallback: TaskStage = TaskStage.IDLE) -> TaskStage:
    """A stage name from a file or a model, or the fallback when it is not
    one of the five - a task never lands in a stage that does not exist."""
    try:
        return TaskStage(value)
    except ValueError:
        if value not in (None, ""):
            logger.warning("Unknown task stage %r; using %r", value, fallback.value)

        return fallback


def task_state_from_json(value: Any) -> TaskState:
    if not isinstance(value, dict):
        return TaskState()

    return TaskState(
        stage=as_stage(value.get("task_stage")),
        current_step=_as_text(value.get("current_step")),
        expected_action=_as_text(value.get("expected_action")),
    )


def task_state_as_json(state: TaskState) -> dict[str, Any]:
    return {
        "task_stage": state.stage.value,
        "current_step": state.current_step,
        "expected_action": state.expected_action,
    }
