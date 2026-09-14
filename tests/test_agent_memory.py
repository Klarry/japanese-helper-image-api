import json

from app.services.agent_history_storage import AgentHistoryStorage, ConversationState
from app.services.agent_memory import (
    AgentLongTermMemoryStorage,
    LongTermMemory,
    MemoryLayers,
    ShortTermMemory,
    WorkingMemory,
)


def _layers(**kwargs):
    return MemoryLayers(**kwargs)


# --- what each layer sends -------------------------------------------------


def test_nothing_is_sent_while_every_layer_is_empty():
    assert _layers().as_sections() == []


def test_each_layer_is_sent_as_its_own_labelled_section():
    sections = _layers(
        short_term=ShortTermMemory(messages=[{"role": "user", "content": "пример со словом 学習"}]),
        working=WorkingMemory(goals=["составить пример"], constraints=["уровень N4"]),
        long_term=LongTermMemory(profile={"favorite_word": "学習"}),
    ).as_sections()

    assert len(sections) == 3
    assert sections[0].startswith("LONG-TERM MEMORY")
    assert sections[1].startswith("WORKING MEMORY")
    assert sections[2].startswith("SHORT-TERM MEMORY")


def test_the_layers_are_sent_from_the_most_durable_to_the_most_immediate():
    """Long-term first, then the task, then the conversation: what is true
    about the learner frames the task, and the task frames the dialogue."""
    sections = _layers(
        short_term=ShortTermMemory(messages=[{"role": "user", "content": "дальше"}]),
        working=WorkingMemory(goals=["цель задачи"]),
        long_term=LongTermMemory(knowledge=["любимое слово 学習"]),
    ).as_sections()

    assert "любимое слово 学習" in sections[0]
    assert "цель задачи" in sections[1]
    assert "дальше" in sections[2]


def test_an_empty_layer_is_left_out_rather_than_sent_as_an_empty_heading():
    sections = _layers(long_term=LongTermMemory(preferences=["краткие ответы"])).as_sections()

    assert len(sections) == 1
    assert "WORKING MEMORY" not in sections[0]
    assert "SHORT-TERM MEMORY" not in sections[0]


def test_a_layer_never_carries_another_layers_content():
    layers = _layers(
        short_term=ShortTermMemory(messages=[{"role": "user", "content": "объясни грамматику"}]),
        working=WorkingMemory(constraints=["уровень N4"]),
        long_term=LongTermMemory(profile={"favorite_word": "学習"}),
    )
    long_term_section, working_section, short_term_section = layers.as_sections()

    assert "уровень N4" not in long_term_section and "объясни грамматику" not in long_term_section
    assert "学習" not in working_section and "объясни грамматику" not in working_section
    assert "уровень N4" not in short_term_section and "学習" not in short_term_section


def test_the_conversation_is_rendered_the_way_every_strategy_renders_it():
    section = ShortTermMemory(
        messages=[{"role": "user", "content": "вопрос"}, {"role": "assistant", "content": "ответ"}]
    ).as_section()

    assert "Learner: вопрос" in section
    assert "Assistant: ответ" in section


# --- long-term memory on disk ----------------------------------------------


def test_long_term_memory_survives_a_fresh_storage_instance(tmp_path):
    path = str(tmp_path / "long_term.json")
    AgentLongTermMemoryStorage(file_path=path).save(
        LongTermMemory(profile={"favorite_word": "学習"}, preferences=["краткие ответы"])
    )

    restarted = AgentLongTermMemoryStorage(file_path=path).load()

    assert restarted.profile == {"favorite_word": "学習"}
    assert restarted.preferences == ["краткие ответы"]


def test_long_term_memory_is_empty_before_anything_is_remembered(tmp_path):
    assert AgentLongTermMemoryStorage(file_path=str(tmp_path / "missing.json")).load().is_empty()


def test_a_broken_long_term_memory_file_reads_as_nothing_remembered(tmp_path):
    path = tmp_path / "long_term.json"
    path.write_text("{ this is not json", encoding="utf-8")

    assert AgentLongTermMemoryStorage(file_path=str(path)).load().is_empty()


def test_clearing_long_term_memory_empties_the_file_rather_than_deleting_it(tmp_path):
    path = tmp_path / "long_term.json"
    storage = AgentLongTermMemoryStorage(file_path=str(path))
    storage.save(LongTermMemory(knowledge=["любимое слово 学習"]))

    storage.clear()

    assert storage.load().is_empty()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "profile": {},
        "preferences": [],
        "decisions": [],
        "knowledge": [],
    }


def test_long_term_memory_keeps_only_what_it_can_read(tmp_path):
    path = tmp_path / "long_term.json"
    path.write_text(
        json.dumps({"profile": {"level": "N4", "bad": {"nested": 1}}, "preferences": "not a list"}),
        encoding="utf-8",
    )

    memory = AgentLongTermMemoryStorage(file_path=str(path)).load()

    assert memory.profile == {"level": "N4"}
    assert memory.preferences == []


# --- working memory travels with the conversation --------------------------


def test_working_memory_is_persisted_with_the_conversation(tmp_path):
    path = str(tmp_path / "history.json")
    state = ConversationState()
    state.working_memory = WorkingMemory(constraints=["уровень N4"], requirements=["показывать перевод"])
    AgentHistoryStorage(file_path=path).save(state)

    restored = AgentHistoryStorage(file_path=path).load()

    assert restored.working_memory.constraints == ["уровень N4"]
    assert restored.working_memory.requirements == ["показывать перевод"]


def test_working_memory_is_kept_out_of_the_stored_transcript(tmp_path):
    """The layers are stored apart, not folded into the messages - which is
    what makes clearing one of them a local change."""
    path = tmp_path / "history.json"
    state = ConversationState()
    state.working_memory = WorkingMemory(constraints=["уровень N4"])
    state.current().messages.append({"role": "user", "content": "составь пример"})
    AgentHistoryStorage(file_path=str(path)).save(state)

    stored = json.loads(path.read_text(encoding="utf-8"))

    assert stored["working_memory"]["constraints"] == ["уровень N4"]
    assert "N4" not in json.dumps(stored["branches"], ensure_ascii=False)


def test_a_history_file_written_before_memory_layers_existed_still_loads(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps({"summary": "", "messages": [{"role": "user", "content": "старый файл"}]}),
        encoding="utf-8",
    )

    state = AgentHistoryStorage(file_path=str(path)).load()

    assert [entry["content"] for entry in state.current().messages] == ["старый файл"]
    assert state.working_memory.is_empty()
