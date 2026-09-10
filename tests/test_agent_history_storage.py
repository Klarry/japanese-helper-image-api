import json

from app.services.agent_history_storage import AgentHistoryStorage, ConversationHistory


def _storage(tmp_path, name="history.json"):
    return AgentHistoryStorage(file_path=str(tmp_path / name))


def _conversation(messages, summary=""):
    return ConversationHistory(summary=summary, messages=messages)


def test_load_returns_an_empty_conversation_when_file_does_not_exist(tmp_path):
    assert _storage(tmp_path).load() == ConversationHistory()


def test_save_then_load_round_trips_messages(tmp_path):
    storage = _storage(tmp_path)
    messages = [
        {"role": "user", "content": "Explain the kanji 学"},
        {"role": "assistant", "content": "学 means to study."},
    ]

    storage.save(_conversation(messages))

    assert storage.load().messages == messages


def test_load_returns_an_empty_conversation_for_corrupted_json(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{not valid json", encoding="utf-8")

    assert AgentHistoryStorage(file_path=str(path)).load() == ConversationHistory()


def test_load_returns_an_empty_conversation_when_messages_key_is_missing(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"other": []}), encoding="utf-8")

    assert AgentHistoryStorage(file_path=str(path)).load() == ConversationHistory()


def test_load_returns_an_empty_conversation_when_messages_is_not_a_list(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"messages": "not-a-list"}), encoding="utf-8")

    assert AgentHistoryStorage(file_path=str(path)).load() == ConversationHistory()


def test_clear_removes_all_messages(tmp_path):
    storage = _storage(tmp_path)
    storage.save(_conversation([{"role": "user", "content": "hi"}]))

    storage.clear()

    assert storage.load().messages == []


def test_save_creates_missing_parent_directories(tmp_path):
    nested_path = tmp_path / "nested" / "history.json"
    storage = AgentHistoryStorage(file_path=str(nested_path))

    storage.save(_conversation([{"role": "user", "content": "hi"}]))

    assert nested_path.exists()
    assert storage.load().messages == [{"role": "user", "content": "hi"}]


def test_save_preserves_non_ascii_content(tmp_path):
    storage = _storage(tmp_path)
    messages = [{"role": "assistant", "content": "学 значит «учиться»"}]

    storage.save(_conversation(messages))

    assert storage.load().messages == messages


def test_a_new_storage_instance_reads_what_a_previous_instance_saved(tmp_path):
    """Simulates a backend restart: a fresh AgentHistoryStorage pointed at
    the same file must see what an earlier instance persisted."""
    file_path = str(tmp_path / "history.json")
    AgentHistoryStorage(file_path=file_path).save(_conversation([{"role": "user", "content": "hi"}]))

    restarted = AgentHistoryStorage(file_path=file_path)

    assert restarted.load().messages == [{"role": "user", "content": "hi"}]


# --- summary alongside the messages ----------------------------------------


def test_the_summary_is_stored_separately_from_the_messages(tmp_path):
    path = tmp_path / "history.json"
    recent = [{"role": "user", "content": "и ещё пример?"}]
    AgentHistoryStorage(file_path=str(path)).save(_conversation(recent, summary="Разбирали 学習."))

    written = json.loads(path.read_text(encoding="utf-8"))

    assert written == {"summary": "Разбирали 学習.", "messages": recent}


def test_save_then_load_round_trips_the_summary(tmp_path):
    storage = _storage(tmp_path)
    conversation = _conversation([{"role": "user", "content": "и ещё?"}], summary="Разбирали 学習.")

    storage.save(conversation)

    assert storage.load() == conversation


def test_a_history_file_written_before_compression_existed_still_loads(tmp_path):
    """The old on-disk shape had no summary key at all - it must keep
    loading, with an empty summary, rather than being treated as corrupt."""
    path = tmp_path / "history.json"
    messages = [{"role": "user", "content": "hi"}]
    path.write_text(json.dumps({"messages": messages}), encoding="utf-8")

    loaded = AgentHistoryStorage(file_path=str(path)).load()

    assert loaded == ConversationHistory(summary="", messages=messages)


def test_a_malformed_summary_is_ignored_without_losing_the_messages(tmp_path):
    path = tmp_path / "history.json"
    messages = [{"role": "user", "content": "hi"}]
    path.write_text(json.dumps({"summary": {"not": "a string"}, "messages": messages}), encoding="utf-8")

    loaded = AgentHistoryStorage(file_path=str(path)).load()

    assert loaded == ConversationHistory(summary="", messages=messages)


def test_clear_removes_the_summary_too(tmp_path):
    storage = _storage(tmp_path)
    storage.save(_conversation([{"role": "user", "content": "hi"}], summary="Разбирали 学習."))

    storage.clear()

    assert storage.load() == ConversationHistory()


def test_the_summary_survives_a_new_storage_instance(tmp_path):
    """Scenario 5: after a restart the summary must come back with the
    recent messages, not just the messages."""
    file_path = str(tmp_path / "history.json")
    recent = [{"role": "user", "content": "и ещё пример?"}]
    AgentHistoryStorage(file_path=file_path).save(_conversation(recent, summary="Разбирали 学習."))

    restarted = AgentHistoryStorage(file_path=file_path).load()

    assert restarted.summary == "Разбирали 学習."
    assert restarted.messages == recent
