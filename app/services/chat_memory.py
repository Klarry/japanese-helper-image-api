"""Day 25: the task's memory, as one thing.

The project already remembers everything this day asks for - it just keeps
it in two places, for good reasons. ``WorkingMemory`` (Day 11) holds what the
task is: its goals, requirements, constraints, decisions and, since today,
the terms it has agreed on. ``TaskState`` (Day 13) holds where the task has
got to: which stage, which step, what is expected next. They are stored apart
because they change for different reasons - a constraint is settled by the
person, a stage by the work.

A chat needs to show them together, so this is a *view* over the two, not a
third store. Nothing here owns anything: it reads the conversation's own
state and renders it, and the changes it reports are a comparison of two
snapshots rather than a log it keeps.

Why a view rather than a new dataclass with the five fields the assignment
names: a second copy would need keeping in step with the first, and a memory
that can disagree with itself is worse than no memory at all.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.services.agent_memory import WorkingMemory
from app.services.agent_task_state import TaskState

#: How many entries of a list the summary shows before saying "and N more".
MAX_SHOWN = 8


@dataclass(frozen=True)
class TaskMemory:
    """What this conversation has settled, in the shape the day asks for."""

    goal: str = ""
    confirmed_terms: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    current_state: str = "idle"

    @classmethod
    def of(cls, working: WorkingMemory, task: TaskState) -> "TaskMemory":
        """The two stores, read as one.

        The goal is the first thing the task said it was for. Later goals are
        kept as goals in working memory but do not displace it: a
        conversation that wanders is still about what it started as, and that
        is the whole point of holding a goal separately from the last
        message.
        """
        return cls(
            goal=working.goals[0] if working.goals else "",
            confirmed_terms=tuple(working.terms),
            constraints=tuple(working.constraints),
            decisions=tuple(working.decisions),
            requirements=tuple(working.requirements),
            current_state=task.stage.value,
        )

    @property
    def is_empty(self) -> bool:
        return not (
            self.goal
            or self.confirmed_terms
            or self.constraints
            or self.decisions
            or self.requirements
        )

    def as_dict(self) -> dict:
        return {
            "goal": self.goal,
            "confirmed_terms": list(self.confirmed_terms),
            "constraints": list(self.constraints),
            "decisions": list(self.decisions),
            "requirements": list(self.requirements),
            "current_state": self.current_state,
        }

    def as_section(self) -> str:
        """What the model is shown. Empty when nothing has been settled, so
        the first question of a conversation is not padded with headings
        describing nothing."""
        if self.is_empty:
            return ""

        lines = [f"- Goal: {self.goal}"] if self.goal else []
        lines += _lines("Constraint", self.constraints)
        lines += _lines("Decision", self.decisions)
        lines += _lines("Requirement", self.requirements)
        lines += _lines("Confirmed term", self.confirmed_terms)
        lines.append(f"- Task stage: {self.current_state}")

        return (
            "TASK MEMORY (what this conversation has already settled - it holds "
            "whatever the newest message is about):\n" + "\n".join(lines)
        )

    def changes_from(self, previous: "TaskMemory") -> list[str]:
        """What this message settled, in words, for the debug view.

        Only additions and the goal are reported. A disappearing constraint
        is not called a removal here: the memory router returns a whole layer
        and may legitimately reword an entry, and announcing "constraint
        removed" for a rewording would be a lie about what happened.
        """
        reported: list[str] = []

        if self.goal and self.goal != previous.goal:
            reported.append(f"goal: {self.goal}")

        for caption, now, before in (
            ("constraint", self.constraints, previous.constraints),
            ("decision", self.decisions, previous.decisions),
            ("requirement", self.requirements, previous.requirements),
            ("term", self.confirmed_terms, previous.confirmed_terms),
        ):
            for entry in now:
                if entry not in before:
                    reported.append(f"+ {caption}: {entry}")

        if self.current_state != previous.current_state:
            reported.append(f"stage: {previous.current_state} -> {self.current_state}")

        return reported


@dataclass
class MemoryUpdate:
    """One message's effect on the task's memory."""

    before: TaskMemory
    after: TaskMemory
    changes: list[str] = field(default_factory=list)
    #: What the router cost, when it ran. Zero when nothing was asked.
    tokens_used: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.changes)

    def as_dict(self) -> dict:
        return {
            "changed": self.changed,
            "changes": self.changes,
            "task_memory": self.after.as_dict(),
            "tokens_used": self.tokens_used,
        }


def _lines(caption: str, entries: Sequence[str]) -> list[str]:
    shown = list(entries[:MAX_SHOWN])
    lines = [f"- {caption}: {entry}" for entry in shown]

    if len(entries) > MAX_SHOWN:
        lines.append(f"- {caption}: and {len(entries) - MAX_SHOWN} more")

    return lines
