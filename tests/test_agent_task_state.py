import json

import pytest

from app.services.agent_history_storage import AgentHistoryStorage, ConversationState
from app.services.agent_task_state import (
    ALLOWED_TRANSITIONS,
    TaskStage,
    TaskState,
    allowed_next,
    can_transition,
)

STAGES = list(TaskStage)
LEGAL = [
    (TaskStage.IDLE, TaskStage.PLANNING),
    (TaskStage.PLANNING, TaskStage.EXECUTION),
    (TaskStage.EXECUTION, TaskStage.VALIDATION),
    (TaskStage.VALIDATION, TaskStage.DONE),
]


# --- the machine -----------------------------------------------------------


@pytest.mark.parametrize("current,proposed", LEGAL)
def test_the_four_legal_moves_are_allowed(current, proposed):
    assert can_transition(current, proposed)


@pytest.mark.parametrize("stage", STAGES)
def test_staying_in_the_same_stage_is_always_allowed(stage):
    """A stage usually takes several messages, so standing still is normal."""
    assert can_transition(stage, stage)


@pytest.mark.parametrize(
    "current,proposed",
    [
        (TaskStage.PLANNING, TaskStage.VALIDATION),
        (TaskStage.PLANNING, TaskStage.DONE),
        (TaskStage.EXECUTION, TaskStage.DONE),
        (TaskStage.IDLE, TaskStage.EXECUTION),
        (TaskStage.IDLE, TaskStage.DONE),
    ],
)
def test_a_stage_can_never_be_skipped(current, proposed):
    assert not can_transition(current, proposed)


@pytest.mark.parametrize(
    "current,proposed",
    [
        (TaskStage.EXECUTION, TaskStage.PLANNING),
        (TaskStage.VALIDATION, TaskStage.EXECUTION),
        (TaskStage.DONE, TaskStage.VALIDATION),
        (TaskStage.PLANNING, TaskStage.IDLE),
    ],
)
def test_a_task_never_goes_backwards(current, proposed):
    assert not can_transition(current, proposed)


def test_nothing_follows_done():
    assert ALLOWED_TRANSITIONS[TaskStage.DONE] == frozenset()
    assert allowed_next(TaskStage.DONE) == []


def test_the_machine_reports_where_it_can_go_next():
    assert allowed_next(TaskStage.IDLE) == ["planning"]
    assert allowed_next(TaskStage.PLANNING) == ["execution"]
    assert allowed_next(TaskStage.EXECUTION) == ["validation"]
    assert allowed_next(TaskStage.VALIDATION) == ["done"]


# --- applying an update ----------------------------------------------------


def test_a_legal_move_is_taken_with_its_step_and_action():
    state = TaskState(stage=TaskStage.PLANNING, current_step="составляем план", expected_action="начать")

    updated = state.with_update(TaskStage.EXECUTION, "шаг 1 из 3", "написать три предложения")

    assert updated.stage is TaskStage.EXECUTION
    assert updated.current_step == "шаг 1 из 3"
    assert updated.expected_action == "написать три предложения"


def test_an_illegal_move_changes_nothing_at_all():
    """Refused whole rather than in part: a proposal that jumps to done
    describes a finished task, and keeping its step text while refusing its
    stage would leave the two contradicting each other."""
    state = TaskState(stage=TaskStage.PLANNING, current_step="составляем план", expected_action="начать")

    updated = state.with_update(TaskStage.DONE, "всё готово", "поздравить")

    assert updated == state


def test_the_step_moves_on_without_changing_the_stage():
    state = TaskState(stage=TaskStage.EXECUTION, current_step="шаг 1 из 3", expected_action="…")

    updated = state.with_update(TaskStage.EXECUTION, "шаг 2 из 3", "показать второе предложение")

    assert updated.stage is TaskStage.EXECUTION
    assert updated.current_step == "шаг 2 из 3"


# --- what the model is told ------------------------------------------------


def test_an_idle_task_says_nothing_at_all():
    """No task running means the prompt is exactly what it was before this
    feature existed."""
    assert TaskState().as_section() == ""
    assert not TaskState().is_active()


def test_an_active_task_states_the_stage_the_step_and_what_comes_next():
    section = TaskState(
        stage=TaskStage.EXECUTION,
        current_step="шаг 2 из 4 — пять предложений с 〜ながら",
        expected_action="показать предложения и спросить про проверку",
    ).as_section()

    assert section.startswith("TASK STATE")
    assert "- Stage: execution" in section
    assert "- Current step: шаг 2 из 4 — пять предложений с 〜ながら" in section
    assert "- Expected next action: показать предложения и спросить про проверку" in section


def test_the_section_tells_the_model_not_to_start_over():
    section = TaskState(stage=TaskStage.EXECUTION, current_step="шаг 2").as_section()

    assert "continue it from this exact step" in section
    assert "do not repeat explanations already given" in section


def test_empty_step_fields_are_left_out_rather_than_sent_blank():
    section = TaskState(stage=TaskStage.PLANNING).as_section()

    assert "- Stage: planning" in section
    assert "Current step" not in section
    assert "Expected next action" not in section


# --- on disk, with the conversation ----------------------------------------


def test_the_task_survives_a_fresh_storage_instance(tmp_path):
    path = str(tmp_path / "history.json")
    state = ConversationState()
    state.task_state = TaskState(
        stage=TaskStage.EXECUTION, current_step="шаг 2 из 4", expected_action="показать предложения"
    )
    AgentHistoryStorage(file_path=path).save(state)

    restored = AgentHistoryStorage(file_path=path).load()

    assert restored.task_state.stage is TaskStage.EXECUTION
    assert restored.task_state.current_step == "шаг 2 из 4"
    assert restored.task_state.expected_action == "показать предложения"


def test_the_task_is_stored_beside_the_branches_not_inside_a_transcript(tmp_path):
    path = tmp_path / "history.json"
    state = ConversationState()
    state.task_state = TaskState(stage=TaskStage.VALIDATION, current_step="проверяем предложения")
    state.current().messages.append({"role": "user", "content": "продолжим"})
    AgentHistoryStorage(file_path=str(path)).save(state)

    stored = json.loads(path.read_text(encoding="utf-8"))

    assert stored["task_state"]["task_stage"] == "validation"
    assert "validation" not in json.dumps(stored["branches"], ensure_ascii=False)


def test_a_history_file_written_before_the_task_state_existed_still_loads(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps({"summary": "", "messages": [{"role": "user", "content": "старый файл"}]}),
        encoding="utf-8",
    )

    state = AgentHistoryStorage(file_path=str(path)).load()

    assert state.task_state.stage is TaskStage.IDLE
    assert [entry["content"] for entry in state.current().messages] == ["старый файл"]


def test_an_unknown_stage_on_disk_reads_as_no_task(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps({"branches": {}, "task_state": {"task_stage": "halfway", "current_step": "?"}}),
        encoding="utf-8",
    )

    assert AgentHistoryStorage(file_path=str(path)).load().task_state.stage is TaskStage.IDLE
