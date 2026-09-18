import json

import pytest

from app.services.agent_history_storage import AgentHistoryStorage, ConversationState
from app.services.agent_task_state import (
    ALLOWED_TRANSITIONS,
    TaskStage,
    TaskState,
    allowed_next,
    can_transition,
    check_transition,
    next_requirement,
    task_state_as_json,
    task_state_from_json,
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
    state = TaskState(
        stage=TaskStage.PLANNING,
        current_step="составляем план",
        expected_action="начать",
        plan="три шага",
    )

    outcome = state.with_update(TaskStage.EXECUTION, "шаг 1 из 3", "написать три предложения")

    assert outcome.refusal is None
    assert outcome.state.stage is TaskStage.EXECUTION
    assert outcome.state.current_step == "шаг 1 из 3"
    assert outcome.state.expected_action == "написать три предложения"


def test_an_illegal_move_changes_nothing_at_all():
    """Refused whole rather than in part: a proposal that jumps to done
    describes a finished task, and keeping its step text while refusing its
    stage would leave the two contradicting each other."""
    state = TaskState(stage=TaskStage.PLANNING, current_step="составляем план", expected_action="начать")

    outcome = state.with_update(TaskStage.DONE, "всё готово", "поздравить")

    assert outcome.refusal is not None
    assert outcome.state.stage is state.stage
    assert outcome.state.current_step == state.current_step
    assert outcome.state.expected_action == state.expected_action


def test_the_step_moves_on_without_changing_the_stage():
    state = TaskState(stage=TaskStage.EXECUTION, current_step="шаг 1 из 3", expected_action="…")

    outcome = state.with_update(TaskStage.EXECUTION, "шаг 2 из 3", "показать второе предложение")

    assert outcome.refusal is None
    assert outcome.state.stage is TaskStage.EXECUTION
    assert outcome.state.current_step == "шаг 2 из 3"


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


# --- Day 15: the check every move goes through -----------------------------


def _planned(**fields) -> TaskState:
    """A task in planning with its plan approved - the state execution may
    legally be entered from."""
    return TaskState(stage=TaskStage.PLANNING, plan="план из 4 шагов", **fields)


def _validated(passed: bool = True) -> TaskState:
    return TaskState(stage=TaskStage.VALIDATION, plan="план", validation_passed=passed)


def test_the_whole_path_is_open_when_each_stage_did_its_work():
    assert check_transition(TaskState(), TaskStage.PLANNING) is None
    assert check_transition(_planned(), TaskStage.EXECUTION) is None
    assert check_transition(TaskState(stage=TaskStage.EXECUTION), TaskStage.VALIDATION) is None
    assert check_transition(_validated(), TaskStage.DONE) is None


@pytest.mark.parametrize("stage", STAGES)
def test_staying_where_you_are_is_never_refused(stage):
    assert check_transition(TaskState(stage=stage), stage) is None


def test_planning_to_done_is_refused_and_says_what_was_skipped():
    refusal = check_transition(_planned(), TaskStage.DONE)

    assert refusal is not None
    assert refusal.current_stage is TaskStage.PLANNING
    assert refusal.requested_stage is TaskStage.DONE
    assert refusal.required_next == ("execution",)
    assert refusal.unmet_condition == "the task has not been through execution and validation yet"


def test_planning_to_validation_is_refused():
    refusal = check_transition(_planned(), TaskStage.VALIDATION)

    assert refusal is not None
    assert refusal.required_next == ("execution",)
    assert refusal.unmet_condition == "the task has not been through execution yet"


def test_execution_to_done_is_refused():
    refusal = check_transition(TaskState(stage=TaskStage.EXECUTION), TaskStage.DONE)

    assert refusal is not None
    assert refusal.required_next == ("validation",)
    assert refusal.unmet_condition == "the task has not been through validation yet"


def test_a_refusal_says_all_four_things_in_one_sentence():
    """The message is the four fields put together, so a client that only
    prints something still tells the learner everything."""
    message = check_transition(TaskState(stage=TaskStage.EXECUTION), TaskStage.DONE).message

    assert "'execution'" in message
    assert "cannot move to 'done'" in message
    assert "the only next stage is 'validation'" in message
    assert "has not been through validation yet" in message


def test_execution_cannot_be_entered_without_an_approved_plan():
    """The edge exists; the stage's own work is what is missing."""
    refusal = check_transition(TaskState(stage=TaskStage.PLANNING), TaskStage.EXECUTION)

    assert refusal is not None
    assert refusal.requested_stage is TaskStage.EXECUTION
    assert refusal.required_next == ("execution",)
    assert refusal.unmet_condition == "the plan has not been approved yet"


def test_done_cannot_be_reached_without_a_validation_that_passed():
    refusal = check_transition(TaskState(stage=TaskStage.VALIDATION), TaskStage.DONE)

    assert refusal is not None
    assert refusal.unmet_condition == "validation has not passed yet"


def test_a_validation_that_failed_is_not_a_validation_that_passed():
    refusal = check_transition(_validated(passed=False), TaskStage.DONE)

    assert refusal is not None
    assert refusal.unmet_condition == "validation has not passed yet"


def test_nothing_at_all_follows_done():
    refusal = check_transition(TaskState(stage=TaskStage.DONE), TaskStage.PLANNING)

    assert refusal is not None
    assert refusal.required_next == ()
    assert "already finished" in refusal.unmet_condition
    assert "nothing follows 'done'" in refusal.message


def test_going_back_is_refused_as_going_back():
    refusal = check_transition(TaskState(stage=TaskStage.VALIDATION), TaskStage.EXECUTION)

    assert refusal is not None
    assert refusal.unmet_condition == "a task never goes back to a stage it has already left"


def test_the_state_says_what_is_still_missing_before_the_next_stage():
    assert next_requirement(TaskState(stage=TaskStage.PLANNING)) == "the plan has not been approved yet"
    assert next_requirement(_planned()) == ""
    assert next_requirement(TaskState(stage=TaskStage.EXECUTION)) == ""
    assert next_requirement(TaskState(stage=TaskStage.VALIDATION)) == "validation has not passed yet"
    assert next_requirement(_validated()) == ""
    assert next_requirement(TaskState(stage=TaskStage.DONE)) == ""


# --- Day 15: what a refused move leaves behind -----------------------------


def test_a_refused_move_records_why_and_moves_nothing():
    state = _planned(current_step="составляем план", expected_action="согласовать план")

    outcome = state.with_update(TaskStage.DONE, "всё готово", "поздравить")

    assert outcome.state.stage is TaskStage.PLANNING
    assert outcome.state.current_step == "составляем план"
    assert outcome.state.blocked == outcome.refusal


def test_the_model_is_told_about_the_refusal_and_what_to_say():
    state = TaskState(stage=TaskStage.EXECUTION, current_step="шаг 2 из 4")

    section = state.with_update(TaskStage.DONE, "готово", "").state.as_section()

    assert "- Stage: execution" in section
    assert "Refused move:" in section
    assert "cannot move to 'done'" in section
    assert "name the stage the task is in" in section


def test_a_move_that_works_clears_the_refusal():
    blocked = TaskState(stage=TaskStage.EXECUTION).with_update(TaskStage.DONE, "", "").state
    assert blocked.blocked is not None

    carried_on = blocked.with_update(TaskStage.VALIDATION, "проверяем", "подтвердить")

    assert carried_on.state.blocked is None
    assert "Refused move" not in carried_on.state.as_section()


def test_approving_the_plan_clears_a_refusal_that_was_about_it():
    blocked = TaskState(stage=TaskStage.PLANNING).with_update(TaskStage.EXECUTION, "шаг 1", "").state

    assert blocked.blocked is not None
    assert blocked.with_plan("план из 4 шагов").blocked is None


# --- Day 15: evidence is only accepted where it is produced ----------------


def test_the_plan_the_learner_approved_unlocks_execution_in_the_same_move():
    state = TaskState(stage=TaskStage.PLANNING, current_step="согласуем план")

    outcome = state.with_update(TaskStage.EXECUTION, "шаг 1 из 4", "написать", plan="план из 4 шагов")

    assert outcome.refusal is None
    assert outcome.state.stage is TaskStage.EXECUTION
    assert outcome.state.plan == "план из 4 шагов"


def test_a_plan_cannot_be_approved_from_outside_planning():
    """Otherwise the condition for leaving planning could be satisfied after
    leaving it - which would make it no condition at all."""
    state = TaskState(stage=TaskStage.EXECUTION)

    outcome = state.with_update(TaskStage.VALIDATION, "проверяем", "", plan="задним числом")

    assert outcome.state.plan == ""


def test_validation_passing_unlocks_done_in_the_same_move():
    state = TaskState(stage=TaskStage.VALIDATION, plan="план")

    outcome = state.with_update(TaskStage.DONE, "готово", "", validation_passed=True)

    assert outcome.refusal is None
    assert outcome.state.stage is TaskStage.DONE
    assert outcome.state.validation_passed


def test_validation_failing_keeps_the_task_in_validation():
    state = TaskState(stage=TaskStage.VALIDATION, plan="план")

    outcome = state.with_update(TaskStage.DONE, "готово", "", validation_passed=False)

    assert outcome.refusal is not None
    assert outcome.state.stage is TaskStage.VALIDATION
    assert outcome.state.validation_passed is False


def test_validation_cannot_be_passed_from_outside_validation():
    state = TaskState(stage=TaskStage.EXECUTION)

    outcome = state.with_update(TaskStage.VALIDATION, "проверяем", "", validation_passed=True)

    assert outcome.state.validation_passed is False


def test_the_plan_travels_with_the_task_through_the_stages():
    executing = _planned().with_update(TaskStage.EXECUTION, "шаг 1", "").state

    assert executing.plan == "план из 4 шагов"
    assert executing.with_update(TaskStage.VALIDATION, "проверяем", "").state.plan == "план из 4 шагов"


# --- Day 15: on disk -------------------------------------------------------


def test_the_plan_the_validation_and_the_refusal_survive_a_restart(tmp_path):
    path = str(tmp_path / "history.json")
    state = ConversationState()
    state.task_state = TaskState(
        stage=TaskStage.VALIDATION, current_step="проверяем", plan="план из 4 шагов"
    ).with_update(TaskStage.DONE, "готово", "").state
    AgentHistoryStorage(file_path=path).save(state)

    restored = AgentHistoryStorage(file_path=path).load().task_state

    assert restored.stage is TaskStage.VALIDATION
    assert restored.plan == "план из 4 шагов"
    assert restored.blocked is not None
    assert restored.blocked.requested_stage is TaskStage.DONE
    assert restored.blocked.unmet_condition == "validation has not passed yet"


def test_a_task_written_before_the_guards_existed_still_loads():
    """An older file has no plan and no validation - which reads as "not done
    yet", so the task keeps its stage and simply cannot skip ahead."""
    state = task_state_from_json(
        {"task_stage": "execution", "current_step": "шаг 2", "expected_action": "дальше"}
    )

    assert state.stage is TaskStage.EXECUTION
    assert state.plan == ""
    assert state.validation_passed is False
    assert state.blocked is None


def test_the_stored_shape_round_trips():
    state = TaskState(
        stage=TaskStage.VALIDATION, plan="план", validation_passed=True, validation_note="всё верно"
    )

    assert task_state_from_json(task_state_as_json(state)) == state
