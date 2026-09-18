import asyncio

import pytest

from app.services import agent_task_tracker as tracker_module
from app.services.agent_task_state import TaskStage, TaskState
from app.services.agent_task_tracker import TaskTracker
from app.services.gemini_service import GeneratedText


def _stub(monkeypatch, text: str, input_tokens=120, output_tokens=20):
    captured = {}

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        captured["prompt"] = prompt
        return GeneratedText(text=text, input_tokens=input_tokens, output_tokens=output_tokens)

    monkeypatch.setattr(tracker_module, "generate_text_with_usage", fake_generate_text_with_usage)
    return captured


def _track(message="продолжим", state=None):
    return asyncio.run(TaskTracker().track(message, state or TaskState()))


def test_a_message_that_starts_a_task_proposes_planning(monkeypatch):
    _stub(
        monkeypatch,
        '{"task_stage": "planning", "current_step": "составляем план", '
        '"expected_action": "согласовать план"}',
    )

    update = _track("Давай составим план изучения грамматики N4.")

    assert update.stage is TaskStage.PLANNING
    assert update.current_step == "составляем план"
    assert update.expected_action == "согласовать план"


def test_an_ordinary_question_keeps_the_task_idle(monkeypatch):
    _stub(monkeypatch, '{"task_stage": "idle", "current_step": "", "expected_action": ""}')

    assert _track("Что значит 学習?").stage is TaskStage.IDLE


def test_the_current_state_is_given_to_the_model_so_it_moves_on_rather_than_restarting(monkeypatch):
    captured = _stub(monkeypatch, '{"task_stage": "execution", "current_step": "шаг 3 из 4"}')
    state = TaskState(stage=TaskStage.EXECUTION, current_step="шаг 2 из 4", expected_action="дальше")

    _track("продолжаем", state)

    assert "шаг 2 из 4" in captured["prompt"]
    assert "execution" in captured["prompt"]
    assert "продолжаем" in captured["prompt"]


def test_the_model_is_told_which_moves_are_legal_from_here(monkeypatch):
    captured = _stub(monkeypatch, "{}")

    _track("продолжаем", TaskState(stage=TaskStage.EXECUTION))

    assert "you may only stay where you are or move to one of: validation" in captured["prompt"]


def test_a_finished_task_is_told_there_is_nowhere_left_to_go(monkeypatch):
    captured = _stub(monkeypatch, "{}")

    _track("спасибо", TaskState(stage=TaskStage.DONE))

    assert "nothing (the task is finished)" in captured["prompt"]


def test_a_missing_stage_keeps_the_one_the_task_is_already_in(monkeypatch):
    _stub(monkeypatch, '{"current_step": "шаг 3 из 4"}')

    update = _track("дальше", TaskState(stage=TaskStage.EXECUTION, current_step="шаг 2 из 4"))

    assert update.stage is TaskStage.EXECUTION
    assert update.current_step == "шаг 3 из 4"


def test_an_unknown_stage_keeps_the_one_the_task_is_already_in(monkeypatch):
    _stub(monkeypatch, '{"task_stage": "almost done"}')

    assert _track("дальше", TaskState(stage=TaskStage.PLANNING)).stage is TaskStage.PLANNING


def test_a_fenced_json_answer_is_still_understood(monkeypatch):
    _stub(monkeypatch, '```json\n{"task_stage": "validation", "current_step": "проверяем"}\n```')

    assert _track("проверь", TaskState(stage=TaskStage.EXECUTION)).stage is TaskStage.VALIDATION


def test_an_answer_that_is_not_json_is_rejected(monkeypatch):
    _stub(monkeypatch, "конечно, продолжаем!")

    with pytest.raises(ValueError):
        _track()


def test_an_answer_that_is_not_an_object_is_rejected(monkeypatch):
    _stub(monkeypatch, '["planning"]')

    with pytest.raises(ValueError):
        _track()


def test_tracking_reports_what_it_cost(monkeypatch):
    """The extra Gemini call this feature makes is visible in the usage log
    rather than hidden inside the answer's own numbers."""
    _stub(monkeypatch, "{}", input_tokens=180, output_tokens=25)

    assert _track().tokens_used == 205


def test_the_tracker_only_proposes_it_does_not_decide(monkeypatch):
    """It can happily suggest an illegal jump - refusing it is the state
    machine's job, and that is where it is tested."""
    _stub(monkeypatch, '{"task_stage": "done", "current_step": "всё"}')

    assert _track("хватит", TaskState(stage=TaskStage.PLANNING)).stage is TaskStage.DONE


# --- Day 15: the evidence the guarded stages need --------------------------


def test_the_plan_the_learner_approved_is_reported_with_the_move(monkeypatch):
    _stub(
        monkeypatch,
        '{"task_stage": "execution", "current_step": "шаг 1 из 4", '
        '"expected_action": "написать предложения", "plan": "4 шага: разбор, примеры, проверка, вывод"}',
    )

    update = _track("План подходит, приступаем.", TaskState(stage=TaskStage.PLANNING))

    assert update.stage is TaskStage.EXECUTION
    assert update.plan == "4 шага: разбор, примеры, проверка, вывод"


def test_a_validation_that_came_out_right_is_reported_with_the_move(monkeypatch):
    _stub(
        monkeypatch,
        '{"task_stage": "done", "current_step": "задача завершена", '
        '"expected_action": "", "validation_passed": true}',
    )

    update = _track("Да, все пять предложений верны.", TaskState(stage=TaskStage.VALIDATION))

    assert update.validation_passed is True


def test_nothing_reported_is_nothing_claimed(monkeypatch):
    """A tracker answer without the two extra fields says nothing about them
    - it must not read as "the plan was approved"."""
    _stub(monkeypatch, '{"task_stage": "planning", "current_step": "составляем план"}')

    update = _track("Давай составим план.")

    assert update.plan == ""
    assert update.validation_passed is None


def test_something_that_is_not_true_or_false_is_not_a_validation(monkeypatch):
    _stub(monkeypatch, '{"task_stage": "validation", "validation_passed": "yes"}')

    assert _track("проверь", TaskState(stage=TaskStage.VALIDATION)).validation_passed is None


def test_the_model_is_told_what_the_next_stage_is_still_waiting_for(monkeypatch):
    captured = _stub(monkeypatch, '{"task_stage": "planning", "current_step": "составляем план"}')

    _track("приступаем", TaskState(stage=TaskStage.PLANNING))

    assert "the plan has not been approved yet" in captured["prompt"]
    assert "not available yet" in captured["prompt"]


def test_an_open_way_forward_is_not_announced_as_a_condition(monkeypatch):
    captured = _stub(monkeypatch, '{"task_stage": "execution", "current_step": "шаг 2"}')

    _track("продолжаем", TaskState(stage=TaskStage.EXECUTION, plan="план"))

    assert "not available yet" not in captured["prompt"]
