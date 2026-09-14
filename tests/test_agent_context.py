from app.schemas.agent import ContextStrategy
from app.services.agent_context import build_context
from app.services.agent_history_storage import ConversationHistory
from app.services.agent_memory import LongTermMemory, MemoryLayers, ShortTermMemory, WorkingMemory

RECENT_KEPT = 6


def _messages(count, prefix="message"):
    return [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"{prefix} {index}"}
        for index in range(count)
    ]


def _history(count=0, **kwargs):
    return ConversationHistory(messages=_messages(count), **kwargs)


def _context(strategy, history):
    return build_context(strategy, history, RECENT_KEPT)


# --- nothing said yet ------------------------------------------------------


def test_every_strategy_sends_nothing_before_the_first_message():
    for strategy in ContextStrategy:
        assert _context(strategy, ConversationHistory()).sections == []


# --- full and branching ----------------------------------------------------


def test_full_sends_the_whole_conversation():
    window = _context(ContextStrategy.FULL, _history(20))

    assert len(window.messages) == 20
    assert "message 0" in window.text
    assert "message 19" in window.text


def test_branching_sends_the_whole_conversation_of_the_branch_it_is_given():
    window = _context(ContextStrategy.BRANCHING, _history(20))

    assert len(window.messages) == 20
    assert "message 0" in window.text


# --- summary (unchanged behaviour) -----------------------------------------


def test_summary_sends_the_summary_and_the_stored_messages():
    history = _history(6, summary="Разбирали 学習.", summary_tokens=452)

    window = _context(ContextStrategy.SUMMARY, history)

    assert "Summary of the earlier part of the conversation:" in window.text
    assert "Разбирали 学習." in window.text
    assert "Conversation so far:" in window.text
    assert window.summary_tokens == 452
    assert len(window.messages) == 6


def test_summary_reports_no_summary_tokens_before_anything_is_summarised():
    assert _context(ContextStrategy.SUMMARY, _history(4)).summary_tokens == 0


# --- sliding window --------------------------------------------------------


def test_sliding_window_sends_only_the_newest_messages():
    """The required guarantee: older messages are out of the context."""
    window = _context(ContextStrategy.SLIDING_WINDOW, _history(20))

    assert len(window.messages) == RECENT_KEPT
    assert "message 19" in window.text
    assert "message 14" in window.text
    assert "message 13" not in window.text
    assert "message 0" not in window.text


def test_sliding_window_leaves_the_stored_conversation_alone():
    """Only the context is windowed. The messages stay on disk, so the same
    dialogue can still be replayed under another strategy - which is the
    whole point of being able to compare them."""
    history = _history(20)

    _context(ContextStrategy.SLIDING_WINDOW, history)

    assert len(history.messages) == 20


def test_sliding_window_sends_everything_while_the_conversation_is_short():
    window = _context(ContextStrategy.SLIDING_WINDOW, _history(4))

    assert len(window.messages) == 4
    assert "message 0" in window.text


def test_sliding_window_never_sends_a_summary():
    history = _history(20, summary="Разбирали 学習.", summary_tokens=452)

    window = _context(ContextStrategy.SLIDING_WINDOW, history)

    assert "Разбирали 学習." not in window.text
    assert window.summary_tokens == 0


# --- sticky facts ----------------------------------------------------------


def test_sticky_facts_sends_the_facts_and_the_newest_messages():
    history = _history(20, facts={"goal": "сдать N3", "preference": "много примеров"})

    window = _context(ContextStrategy.STICKY_FACTS, history)

    assert "goal: сдать N3" in window.text
    assert "preference: много примеров" in window.text
    assert len(window.messages) == RECENT_KEPT
    assert "message 19" in window.text
    assert "message 0" not in window.text


def test_sticky_facts_never_sends_a_summary():
    history = _history(20, summary="Разбирали 学習.", summary_tokens=452, facts={"goal": "сдать N3"})

    window = _context(ContextStrategy.STICKY_FACTS, history)

    assert "Разбирали 学習." not in window.text
    assert "Summary" not in window.text
    assert window.summary_tokens == 0


def test_sticky_facts_sends_just_the_window_while_there_are_no_facts_yet():
    window = _context(ContextStrategy.STICKY_FACTS, _history(8))

    assert window.facts == {}
    assert "Known facts" not in window.text
    assert len(window.messages) == RECENT_KEPT


def test_sticky_facts_reports_the_facts_it_sent():
    facts = {"goal": "сдать N3"}

    assert _context(ContextStrategy.STICKY_FACTS, _history(2, facts=facts)).facts == facts


# --- what the strategies cost relative to each other -----------------------


def test_the_windowed_strategies_send_less_than_the_full_one():
    history = _history(40, facts={"goal": "сдать N3"})

    full = _context(ContextStrategy.FULL, history)
    window = _context(ContextStrategy.SLIDING_WINDOW, history)
    facts = _context(ContextStrategy.STICKY_FACTS, history)

    assert len(window.text) < len(full.text)
    assert len(facts.text) < len(full.text)
    # Facts cost a little more than the bare window - that is what they buy.
    assert len(facts.text) > len(window.text)


# --- layered memory --------------------------------------------------------


def _layered(messages=None, working=None, long_term=None):
    history = ConversationHistory(messages=messages or [])

    return build_context(
        ContextStrategy.LAYERED_MEMORY,
        history,
        RECENT_KEPT,
        MemoryLayers(
            short_term=ShortTermMemory(messages=history.messages),
            working=working or WorkingMemory(),
            long_term=long_term or LongTermMemory(),
        ),
    )


def test_layered_memory_sends_each_layer_as_its_own_named_section():
    window = _layered(
        messages=_messages(2),
        working=WorkingMemory(constraints=["уровень N4"]),
        long_term=LongTermMemory(profile={"favorite_word": "学習"}),
    )

    assert len(window.sections) == 3
    assert "LONG-TERM MEMORY" in window.text
    assert "WORKING MEMORY" in window.text
    assert "SHORT-TERM MEMORY" in window.text


def test_layered_memory_sends_the_whole_conversation_as_short_term_memory():
    """Short-term memory is the current conversation, not a window over it -
    trimming is what the other strategies are for."""
    window = _layered(messages=_messages(20))

    assert len(window.messages) == 20
    assert "message 0" in window.text


def test_layered_memory_sends_the_other_layers_even_with_nothing_said_yet():
    window = _layered(long_term=LongTermMemory(knowledge=["любимое слово 学習"]))

    assert "学習" in window.text
    assert "SHORT-TERM MEMORY" not in window.text


def test_layered_memory_uses_no_summary_and_no_sticky_facts():
    window = _layered(messages=_messages(4), working=WorkingMemory(goals=["цель"]))

    assert window.summary_tokens == 0
    assert window.facts == {}


def test_layered_memory_without_any_memory_sends_nothing():
    assert build_context(ContextStrategy.LAYERED_MEMORY, _history(4), RECENT_KEPT).sections == []


def test_the_other_strategies_ignore_the_memory_layers():
    """Passing memory in must not change what the earlier strategies send -
    they were never about layers."""
    memory = MemoryLayers(
        short_term=ShortTermMemory(messages=_messages(2)),
        working=WorkingMemory(constraints=["уровень N4"]),
        long_term=LongTermMemory(profile={"favorite_word": "学習"}),
    )

    for strategy in (ContextStrategy.FULL, ContextStrategy.SLIDING_WINDOW, ContextStrategy.STICKY_FACTS):
        text = build_context(strategy, _history(4), RECENT_KEPT, memory).text

        assert "уровень N4" not in text
        assert "学習" not in text
