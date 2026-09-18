"""Day 13, then Day 15: where a task is, as a state machine with guarded
transitions.

Memory says what was said, the profile says how to answer, the invariants say
what may never be done - none of them says what the learner and the agent are
in the middle of. A task has stages, and they only ever go one way:

    planning -> execution -> validation -> done

Day 13 built that table. Day 15 adds the half a table cannot express: a move
can be legal in shape and still wrong in fact. Leaving planning needs a plan
that was actually approved; reaching done needs a validation that actually
passed. So every change of stage goes through one check (``check_transition``)
that asks both questions - is this edge in the table, and is the stage's own
work finished - and either lets the move happen or refuses it with the four
things a refusal has to say:

    where the task is · where it was asked to go · what may come next ·
    which condition is not met yet

A refused move changes nothing about where the task is. The only thing written
is the refusal itself, on ``TaskState.blocked``, so the reason outlives the
response that reported it: the screen keeps showing why the task did not
advance, and the agent is told on the next message and can explain it in
words. Any move that does succeed clears it.

Alongside the stage, two fields say where inside it the task is:
``current_step`` (what is being done now) and ``expected_action`` (what should
happen next). All of it travels with the conversation in the same file the
branches live in - the task belongs to the conversation, so clearing the
conversation ends the task too.
"""

import logging
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

from app.schemas.agent import TaskStage

logger = logging.getLogger(__name__)

_MAX_VALUE_LENGTH = 300

# The stages in the order a task takes them. Used to tell a skipped stage
# ("planning -> done" leaves out two) from a backwards one.
STAGE_ORDER: tuple[TaskStage, ...] = (
    TaskStage.IDLE,
    TaskStage.PLANNING,
    TaskStage.EXECUTION,
    TaskStage.VALIDATION,
    TaskStage.DONE,
)

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

# What each guarded stage needs before it can be entered. The table above says
# which door exists; this says whether it is unlocked.
_PLAN_NEEDED = "the plan has not been approved yet"
_VALIDATION_NEEDED = "validation has not passed yet"


def can_transition(current: TaskStage, proposed: TaskStage) -> bool:
    """Whether the shape of the move is allowed: is this edge in the table.

    Staying put is always fine; every other move has to be listed. This
    answers only half the question - see ``check_transition`` for the half
    that looks at whether the current stage's work is finished.
    """
    if proposed is current:
        return True

    return proposed in ALLOWED_TRANSITIONS[current]


def allowed_next(current: TaskStage) -> list[str]:
    return sorted(stage.value for stage in ALLOWED_TRANSITIONS[current])


@dataclass(frozen=True)
class TransitionRefusal:
    """A move that was not made, and why.

    Four fields rather than a string, because a client (or a person) needs
    different parts of it: the current stage to show, the required next stage
    to offer, and the unmet condition to act on.
    """

    current_stage: TaskStage
    requested_stage: TaskStage
    required_next: tuple[str, ...]
    unmet_condition: str

    @property
    def message(self) -> str:
        """The four fields as one sentence, for whoever just wants to print
        something."""
        if self.required_next:
            where = (
                f"from '{self.current_stage.value}' the only next stage is "
                f"'{' or '.join(self.required_next)}'"
            )
        else:
            where = f"nothing follows '{self.current_stage.value}'"

        return (
            f"The task is in '{self.current_stage.value}' and cannot move to "
            f"'{self.requested_stage.value}': {where}, and {self.unmet_condition}."
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "current_stage": self.current_stage.value,
            "requested_stage": self.requested_stage.value,
            "required_next": list(self.required_next),
            "unmet_condition": self.unmet_condition,
        }


def refusal_from_json(value: Any) -> "TransitionRefusal | None":
    if not isinstance(value, dict):
        return None

    current = as_stage(value.get("current_stage"))
    requested = as_stage(value.get("requested_stage"))
    required = value.get("required_next")

    return TransitionRefusal(
        current_stage=current,
        requested_stage=requested,
        required_next=tuple(item for item in required if isinstance(item, str))
        if isinstance(required, list)
        else tuple(allowed_next(current)),
        unmet_condition=_as_text(value.get("unmet_condition")),
    )


def _edge_condition(current: TaskStage, requested: TaskStage) -> str:
    """Why an edge that is not in the table is not in the table."""
    if current is TaskStage.DONE:
        return "the task is already finished and has to be cleared before a new one starts"

    if STAGE_ORDER.index(requested) < STAGE_ORDER.index(current):
        return "a task never goes back to a stage it has already left"

    skipped = [stage.value for stage in STAGE_ORDER[STAGE_ORDER.index(current) + 1 : STAGE_ORDER.index(requested)]]

    return f"the task has not been through {' and '.join(skipped)} yet"


def _gate_condition(state: "TaskState", requested: TaskStage) -> str:
    """Why a legal edge is still shut: the work of the current stage is not
    finished. Empty when the way is clear."""
    if requested is TaskStage.EXECUTION and not state.plan:
        return _PLAN_NEEDED

    if requested is TaskStage.DONE and not state.validation_passed:
        return _VALIDATION_NEEDED

    return ""


def check_transition(state: "TaskState", proposed: TaskStage) -> TransitionRefusal | None:
    """The one check every change of stage goes through, whoever asked for it
    - a client through the API or the agent through the tracker.

    ``None`` means the move may happen. Anything else is the refusal, with
    the current stage, the stage that may come next and the condition that is
    not met yet.
    """
    if proposed is state.stage:
        return None

    condition = (
        _edge_condition(state.stage, proposed)
        if not can_transition(state.stage, proposed)
        else _gate_condition(state, proposed)
    )

    if not condition:
        return None

    return TransitionRefusal(
        current_stage=state.stage,
        requested_stage=proposed,
        required_next=tuple(allowed_next(state.stage)),
        unmet_condition=condition,
    )


def next_requirement(state: "TaskState") -> str:
    """What still has to happen before the task can leave the stage it is in.

    Empty when the next stage is already within reach (or when there is no
    next stage at all).
    """
    for stage in ALLOWED_TRANSITIONS[state.stage]:
        condition = _gate_condition(state, stage)

        if condition:
            return condition

    return ""


class TransitionOutcome(NamedTuple):
    """The state after a proposed move, and the refusal if there was one.

    Both are returned together because they are the same event seen twice:
    the state is what to store, the refusal is what to say.
    """

    state: "TaskState"
    refusal: TransitionRefusal | None


@dataclass
class TaskState:
    """The task in progress: which stage, what is being done in it, what is
    expected next - and the two facts that unlock the stages after it."""

    stage: TaskStage = TaskStage.IDLE
    current_step: str = ""
    expected_action: str = ""
    # The plan the task is being executed by, once it has been approved.
    # Empty means planning has not produced one, which is what keeps the task
    # out of execution.
    plan: str = ""
    validation_passed: bool = False
    validation_note: str = ""
    # The last refused move. Not part of where the task is - part of why it
    # is still there.
    blocked: TransitionRefusal | None = None

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

        if self.plan:
            lines.append(f"- Approved plan: {self.plan}")

        if self.blocked is not None:
            lines.append(
                f"- Refused move: {self.blocked.message} Tell the learner this in their "
                "own language - name the stage the task is in, the stage that has to "
                "come next and what is missing - and carry on from the current step."
            )

        return (
            "TASK STATE (a task is already in progress - continue it from this exact "
            "step; do not start it over and do not repeat explanations already given):\n"
            + "\n".join(lines)
        )

    def with_plan(self, plan: str) -> "TaskState":
        """The state with an approved plan. Approving one is a change like any
        other, so it clears a refusal that may have been about its absence."""
        return replace(self, plan=plan.strip()[:_MAX_VALUE_LENGTH], blocked=None)

    def with_validation(self, passed: bool, note: str = "") -> "TaskState":
        """The state with the outcome of validation recorded. A failed check
        is recorded too: "checked, not right yet" is not "not checked"."""
        return replace(
            self,
            validation_passed=passed,
            validation_note=note.strip()[:_MAX_VALUE_LENGTH],
            blocked=None,
        )

    def with_update(
        self,
        stage: TaskStage,
        current_step: str,
        expected_action: str,
        plan: str = "",
        validation_passed: bool | None = None,
    ) -> TransitionOutcome:
        """The state after a proposed update, or this state with the refusal
        recorded when the move is not allowed.

        ``plan`` and ``validation_passed`` are what the agent noticed in the
        conversation - the learner approving a plan, a check coming out right.
        They are only accepted in the stage that produces them, so neither can
        be used to unlock a stage from outside it.

        An illegal move is refused whole rather than in part: a proposal to
        jump from planning straight to done describes a task that finished,
        and keeping its step text while refusing its stage would leave the two
        contradicting each other.
        """
        base = self

        if plan and self.stage is TaskStage.PLANNING:
            base = base.with_plan(plan)

        if validation_passed is not None and self.stage is TaskStage.VALIDATION:
            base = base.with_validation(validation_passed, base.validation_note)

        refusal = check_transition(base, stage)

        if refusal is not None:
            logger.warning(
                "Refusing task transition %s -> %s: %s",
                self.stage.value,
                stage.value,
                refusal.unmet_condition,
            )
            return TransitionOutcome(replace(base, blocked=refusal), refusal)

        return TransitionOutcome(
            replace(
                base,
                stage=stage,
                current_step=current_step.strip()[:_MAX_VALUE_LENGTH],
                expected_action=expected_action.strip()[:_MAX_VALUE_LENGTH],
                blocked=None,
            ),
            None,
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
        plan=_as_text(value.get("plan")),
        validation_passed=value.get("validation_passed") is True,
        validation_note=_as_text(value.get("validation_note")),
        blocked=refusal_from_json(value.get("blocked")),
    )


def task_state_as_json(state: TaskState) -> dict[str, Any]:
    return {
        "task_stage": state.stage.value,
        "current_step": state.current_step,
        "expected_action": state.expected_action,
        "plan": state.plan,
        "validation_passed": state.validation_passed,
        "validation_note": state.validation_note,
        "blocked": state.blocked.as_json() if state.blocked is not None else None,
    }
