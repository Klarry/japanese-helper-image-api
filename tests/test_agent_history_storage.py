import json

from app.services.agent_history_storage import (
    MAIN_BRANCH,
    AgentHistoryStorage,
    Checkpoint,
    ConversationHistory,
    ConversationState,
)


def _storage(tmp_path, name="history.json"):
    return AgentHistoryStorage(file_path=str(tmp_path / name))


def _state(messages=None, summary="", summary_tokens=None, facts=None):
    return ConversationState(
        branches={
            MAIN_BRANCH: ConversationHistory(
                summary=summary,
                messages=messages or [],
                summary_tokens=summary_tokens,
                facts=facts or {},
            )
        }
    )


def _main(state):
    return state.branches[MAIN_BRANCH]


# --- the conversation itself -----------------------------------------------


def test_load_returns_an_empty_conversation_when_file_does_not_exist(tmp_path):
    loaded = _storage(tmp_path).load()

    assert loaded.branches == {MAIN_BRANCH: ConversationHistory()}
    assert loaded.current_branch == MAIN_BRANCH


def test_save_then_load_round_trips_messages(tmp_path):
    storage = _storage(tmp_path)
    messages = [
        {"role": "user", "content": "Explain the kanji 学"},
        {"role": "assistant", "content": "学 means to study."},
    ]

    storage.save(_state(messages))

    assert _main(storage.load()).messages == messages


def test_load_returns_an_empty_conversation_for_corrupted_json(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{not valid json", encoding="utf-8")

    assert _main(AgentHistoryStorage(file_path=str(path)).load()) == ConversationHistory()


def test_load_returns_an_empty_conversation_when_the_file_is_not_an_object(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps(["nope"]), encoding="utf-8")

    assert _main(AgentHistoryStorage(file_path=str(path)).load()) == ConversationHistory()


def test_load_returns_an_empty_conversation_when_messages_key_is_missing(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"other": []}), encoding="utf-8")

    assert _main(AgentHistoryStorage(file_path=str(path)).load()) == ConversationHistory()


def test_load_returns_an_empty_conversation_when_messages_is_not_a_list(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"messages": "not-a-list"}), encoding="utf-8")

    assert _main(AgentHistoryStorage(file_path=str(path)).load()) == ConversationHistory()


def test_clear_removes_all_messages(tmp_path):
    storage = _storage(tmp_path)
    storage.save(_state([{"role": "user", "content": "hi"}]))

    storage.clear()

    assert _main(storage.load()).messages == []


def test_save_creates_missing_parent_directories(tmp_path):
    nested_path = tmp_path / "nested" / "history.json"
    storage = AgentHistoryStorage(file_path=str(nested_path))

    storage.save(_state([{"role": "user", "content": "hi"}]))

    assert nested_path.exists()
    assert _main(storage.load()).messages == [{"role": "user", "content": "hi"}]


def test_save_preserves_non_ascii_content(tmp_path):
    storage = _storage(tmp_path)
    messages = [{"role": "assistant", "content": "学 значит «учиться»"}]

    storage.save(_state(messages))

    assert _main(storage.load()).messages == messages


def test_a_new_storage_instance_reads_what_a_previous_instance_saved(tmp_path):
    """Simulates a backend restart: a fresh AgentHistoryStorage pointed at
    the same file must see what an earlier instance persisted."""
    file_path = str(tmp_path / "history.json")
    AgentHistoryStorage(file_path=file_path).save(_state([{"role": "user", "content": "hi"}]))

    restarted = AgentHistoryStorage(file_path=file_path)

    assert _main(restarted.load()).messages == [{"role": "user", "content": "hi"}]


# --- summary alongside the messages ----------------------------------------


def test_save_then_load_round_trips_the_summary(tmp_path):
    storage = _storage(tmp_path)
    storage.save(_state([{"role": "user", "content": "и ещё?"}], summary="Разбирали 学習."))

    assert _main(storage.load()).summary == "Разбирали 学習."


def test_the_summarys_own_token_count_round_trips(tmp_path):
    storage = _storage(tmp_path)
    storage.save(_state(summary="Разбирали 学習.", summary_tokens=1245))

    assert _main(storage.load()).summary_tokens == 1245


def test_a_malformed_summary_is_ignored_without_losing_the_messages(tmp_path):
    path = tmp_path / "history.json"
    messages = [{"role": "user", "content": "hi"}]
    path.write_text(json.dumps({"summary": {"not": "a string"}, "messages": messages}), encoding="utf-8")

    loaded = _main(AgentHistoryStorage(file_path=str(path)).load())

    assert loaded.summary == ""
    assert loaded.messages == messages


def test_a_malformed_summary_token_count_is_ignored(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"summary": "s", "summary_tokens": "many", "messages": []}), encoding="utf-8")

    assert AgentHistoryStorage(file_path=str(path)).load().current().summary_tokens is None


def test_clear_removes_the_summary_too(tmp_path):
    storage = _storage(tmp_path)
    storage.save(_state([{"role": "user", "content": "hi"}], summary="Разбирали 学習."))

    storage.clear()

    assert _main(storage.load()) == ConversationHistory()


# --- a file written before branches existed --------------------------------


def test_a_history_file_written_before_branches_existed_loads_as_the_main_branch(tmp_path):
    """The older on-disk shape is one conversation at the top level, with no
    branches key at all. It must keep loading rather than read as corrupt."""
    path = tmp_path / "history.json"
    messages = [{"role": "user", "content": "hi"}]
    path.write_text(
        json.dumps({"summary": "Разбирали 学習.", "summary_tokens": 42, "messages": messages}),
        encoding="utf-8",
    )

    loaded = AgentHistoryStorage(file_path=str(path)).load()

    assert loaded.current_branch == MAIN_BRANCH
    assert loaded.current().messages == messages
    assert loaded.current().summary == "Разбирали 学習."
    assert loaded.current().summary_tokens == 42


# --- facts -----------------------------------------------------------------


def test_facts_round_trip(tmp_path):
    storage = _storage(tmp_path)
    facts = {"goal": "сдать N3", "preference": "много примеров"}

    storage.save(_state(facts=facts))

    assert _main(storage.load()).facts == facts


def test_malformed_facts_entries_are_dropped_without_losing_the_rest(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps({"messages": [], "facts": {"goal": "сдать N3", "level": {"nested": "no"}}}),
        encoding="utf-8",
    )

    assert AgentHistoryStorage(file_path=str(path)).load().current().facts == {"goal": "сдать N3"}


def test_facts_survive_a_new_storage_instance(tmp_path):
    file_path = str(tmp_path / "history.json")
    AgentHistoryStorage(file_path=file_path).save(_state(facts={"goal": "сдать N3"}))

    assert AgentHistoryStorage(file_path=file_path).load().current().facts == {"goal": "сдать N3"}


# --- branches, checkpoints and the chosen strategy -------------------------


def test_branches_are_stored_separately_and_round_trip(tmp_path):
    storage = _storage(tmp_path)
    state = ConversationState(
        current_branch="formal",
        branches={
            MAIN_BRANCH: ConversationHistory(messages=[{"role": "user", "content": "main only"}]),
            "formal": ConversationHistory(messages=[{"role": "user", "content": "formal only"}]),
        },
    )

    storage.save(state)
    loaded = storage.load()

    assert sorted(loaded.branches) == ["formal", "main"]
    assert loaded.branches[MAIN_BRANCH].messages == [{"role": "user", "content": "main only"}]
    assert loaded.branches["formal"].messages == [{"role": "user", "content": "formal only"}]
    assert loaded.current_branch == "formal"
    assert loaded.current().messages == [{"role": "user", "content": "formal only"}]


def test_checkpoints_round_trip_with_the_conversation_they_captured(tmp_path):
    storage = _storage(tmp_path)
    captured = ConversationHistory(messages=[{"role": "user", "content": "before the fork"}])
    state = _state()
    state.checkpoints["cp-1"] = Checkpoint(branch=MAIN_BRANCH, history=captured)

    storage.save(state)
    loaded = storage.load()

    assert loaded.checkpoints["cp-1"].branch == MAIN_BRANCH
    assert loaded.checkpoints["cp-1"].history.messages == [{"role": "user", "content": "before the fork"}]


def test_the_chosen_strategy_round_trips(tmp_path):
    storage = _storage(tmp_path)
    state = _state()
    state.strategy = "sticky_facts"

    storage.save(state)

    assert storage.load().strategy == "sticky_facts"


def test_a_current_branch_that_no_longer_exists_falls_back_to_one_that_does(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps({"current_branch": "gone", "branches": {MAIN_BRANCH: {"messages": []}}}),
        encoding="utf-8",
    )

    loaded = AgentHistoryStorage(file_path=str(path)).load()

    assert loaded.current_branch == MAIN_BRANCH
    assert loaded.current() == ConversationHistory()


def test_a_malformed_branches_object_reads_as_an_empty_conversation(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"branches": "nope"}), encoding="utf-8")

    loaded = AgentHistoryStorage(file_path=str(path)).load()

    assert loaded.branches == {MAIN_BRANCH: ConversationHistory()}


def test_everything_survives_a_new_storage_instance(tmp_path):
    """The restart guarantee, for all of it at once: strategy, branches,
    their messages and facts, and the checkpoints taken along the way."""
    file_path = str(tmp_path / "history.json")
    state = ConversationState(
        strategy="branching",
        current_branch="casual",
        branches={
            MAIN_BRANCH: ConversationHistory(messages=[{"role": "user", "content": "shared"}]),
            "casual": ConversationHistory(
                messages=[{"role": "user", "content": "casual"}],
                facts={"tone": "разговорный"},
            ),
        },
    )
    state.checkpoints["cp-1"] = Checkpoint(branch=MAIN_BRANCH, history=ConversationHistory())
    AgentHistoryStorage(file_path=file_path).save(state)

    restarted = AgentHistoryStorage(file_path=file_path).load()

    assert restarted.strategy == "branching"
    assert restarted.current_branch == "casual"
    assert restarted.current().facts == {"tone": "разговорный"}
    assert restarted.branches[MAIN_BRANCH].messages == [{"role": "user", "content": "shared"}]
    assert "cp-1" in restarted.checkpoints
