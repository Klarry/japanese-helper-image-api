import json

from app.services.agent_history_storage import AgentHistoryStorage


def test_load_returns_empty_list_when_file_does_not_exist(tmp_path):
    storage = AgentHistoryStorage(file_path=str(tmp_path / "history.json"))

    assert storage.load() == []


def test_save_then_load_round_trips_messages(tmp_path):
    storage = AgentHistoryStorage(file_path=str(tmp_path / "history.json"))
    messages = [
        {"role": "user", "content": "Explain the kanji 学"},
        {"role": "assistant", "content": "学 means to study."},
    ]

    storage.save(messages)

    assert storage.load() == messages


def test_load_returns_empty_list_for_corrupted_json(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{not valid json", encoding="utf-8")
    storage = AgentHistoryStorage(file_path=str(path))

    assert storage.load() == []


def test_load_returns_empty_list_when_messages_key_is_missing(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"other": []}), encoding="utf-8")
    storage = AgentHistoryStorage(file_path=str(path))

    assert storage.load() == []


def test_load_returns_empty_list_when_messages_is_not_a_list(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"messages": "not-a-list"}), encoding="utf-8")
    storage = AgentHistoryStorage(file_path=str(path))

    assert storage.load() == []


def test_clear_removes_all_messages(tmp_path):
    storage = AgentHistoryStorage(file_path=str(tmp_path / "history.json"))
    storage.save([{"role": "user", "content": "hi"}])

    storage.clear()

    assert storage.load() == []


def test_save_creates_missing_parent_directories(tmp_path):
    nested_path = tmp_path / "nested" / "history.json"
    storage = AgentHistoryStorage(file_path=str(nested_path))

    storage.save([{"role": "user", "content": "hi"}])

    assert nested_path.exists()
    assert storage.load() == [{"role": "user", "content": "hi"}]


def test_save_preserves_non_ascii_content(tmp_path):
    storage = AgentHistoryStorage(file_path=str(tmp_path / "history.json"))
    messages = [{"role": "assistant", "content": "学 значит «учиться»"}]

    storage.save(messages)

    assert storage.load() == messages


def test_a_new_storage_instance_reads_what_a_previous_instance_saved(tmp_path):
    """Simulates a backend restart: a fresh AgentHistoryStorage pointed at
    the same file must see what an earlier instance persisted."""
    file_path = str(tmp_path / "history.json")
    AgentHistoryStorage(file_path=file_path).save([{"role": "user", "content": "hi"}])

    restarted = AgentHistoryStorage(file_path=file_path)

    assert restarted.load() == [{"role": "user", "content": "hi"}]
