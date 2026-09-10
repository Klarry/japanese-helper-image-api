import json

from app.services.agent_usage_log import AgentUsageLog


def _record(total_tokens=100, compression_enabled=False):
    return {
        "timestamp": "2026-09-10T08:00:00+00:00",
        "compression_enabled": compression_enabled,
        "messages_sent": 4,
        "summary_used": compression_enabled,
        "current_request_tokens": 10,
        "history_tokens": 60,
        "response_tokens": 30,
        "total_tokens": total_tokens,
        "summarization_tokens": 0,
    }


def test_load_returns_nothing_when_the_file_does_not_exist(tmp_path):
    assert AgentUsageLog(file_path=str(tmp_path / "usage.json")).load() == []


def test_append_then_load_round_trips_a_record(tmp_path):
    log = AgentUsageLog(file_path=str(tmp_path / "usage.json"))
    record = _record()

    log.append(record)

    assert log.load() == [record]


def test_append_keeps_earlier_records_in_order(tmp_path):
    log = AgentUsageLog(file_path=str(tmp_path / "usage.json"))

    log.append(_record(total_tokens=100))
    log.append(_record(total_tokens=200))

    assert [entry["total_tokens"] for entry in log.load()] == [100, 200]


def test_append_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "usage.json"
    log = AgentUsageLog(file_path=str(path))

    log.append(_record())

    assert path.exists()


def test_load_returns_nothing_for_a_corrupted_file(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("{not valid json", encoding="utf-8")

    assert AgentUsageLog(file_path=str(path)).load() == []


def test_load_returns_nothing_when_entries_is_not_a_list(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text(json.dumps({"entries": "nope"}), encoding="utf-8")

    assert AgentUsageLog(file_path=str(path)).load() == []


def test_records_from_both_modes_can_be_told_apart(tmp_path):
    """The whole point of the log: comparing the two experiment runs."""
    log = AgentUsageLog(file_path=str(tmp_path / "usage.json"))
    log.append(_record(total_tokens=900, compression_enabled=False))
    log.append(_record(total_tokens=300, compression_enabled=True))

    by_mode = {entry["compression_enabled"]: entry["total_tokens"] for entry in log.load()}

    assert by_mode == {False: 900, True: 300}


def test_a_new_log_instance_reads_what_a_previous_instance_appended(tmp_path):
    file_path = str(tmp_path / "usage.json")
    AgentUsageLog(file_path=file_path).append(_record())

    assert len(AgentUsageLog(file_path=file_path).load()) == 1
