import asyncio

import pytest
from fastapi import HTTPException

from app.services import agent_history_compressor as compressor_module
from app.services import japanese_learning_agent as agent_module
from app.services.agent_history_compressor import HistoryCompressor
from app.services.agent_history_storage import AgentHistoryStorage
from app.services.agent_usage_log import AgentUsageLog
from app.services.gemini_service import GeneratedText
from app.services.japanese_learning_agent import JapaneseLearningAgent


def _stub_generate(monkeypatch, handler):
    """handler(prompt) -> GeneratedText. Replaces generate_text_with_usage."""
    calls = []

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        calls.append(prompt)
        return handler(prompt)

    monkeypatch.setattr(agent_module, "generate_text_with_usage", fake_generate_text_with_usage)
    return calls


def _stub_summary(monkeypatch, handler):
    """handler(prompt) -> GeneratedText. Replaces the Gemini call the
    compressor makes - a different module, so it is stubbed separately."""
    calls = []

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        calls.append(prompt)
        return handler(prompt)

    monkeypatch.setattr(compressor_module, "generate_text_with_usage", fake_generate_text_with_usage)
    return calls


def _stub_count_tokens(monkeypatch, handler):
    """handler(text) -> int. Replaces count_tokens."""
    calls = []

    async def fake_count_tokens(text, model=None):
        calls.append(text)
        return handler(text)

    monkeypatch.setattr(agent_module, "count_tokens", fake_count_tokens)
    return calls


def _generated(text="answer", input_tokens=10, output_tokens=5):
    return GeneratedText(text=text, input_tokens=input_tokens, output_tokens=output_tokens)


def _agent(tmp_path, name="history.json", compression_enabled=False, **compressor_kwargs):
    """An agent wired to throwaway files. Compression is stated explicitly
    rather than inherited from the environment, so these tests mean the same
    thing whatever AGENT_COMPRESSION_ENABLED happens to be set to."""
    return JapaneseLearningAgent(
        history_storage=AgentHistoryStorage(file_path=str(tmp_path / name)),
        usage_log=AgentUsageLog(file_path=str(tmp_path / f"usage-{name}")),
        compressor=HistoryCompressor(**compressor_kwargs) if compressor_kwargs else None,
        compression_enabled=compression_enabled,
    )


def _compressing_agent(tmp_path, name="compressed.json", **compressor_kwargs):
    return _agent(tmp_path, name=name, compression_enabled=True, **compressor_kwargs)


def _hold_a_conversation(agent, turns, start=0):
    """Run `turns` numbered turns and return every response."""
    return [asyncio.run(agent.run(f"question {index}")) for index in range(start, start + turns)]


def _numbered_answers(prefix="answer"):
    counter = iter(range(1000))
    return lambda prompt: _generated(text=f"{prefix} {next(counter)}")


# --- prompt building / persistence (unchanged behaviour) -------------------


def test_run_builds_a_prompt_that_includes_the_learners_message(monkeypatch, tmp_path):
    calls = _stub_generate(monkeypatch, lambda prompt: _generated())

    asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert len(calls) == 1
    assert "Explain the kanji 学." in calls[0]


def test_run_instructs_the_model_to_answer_in_russian_by_default(monkeypatch, tmp_path):
    calls = _stub_generate(monkeypatch, lambda prompt: _generated())

    asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert "Russian" in calls[0]


def test_run_calls_the_existing_gemini_service(monkeypatch, tmp_path):
    calls = _stub_generate(monkeypatch, lambda prompt: _generated())

    asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert len(calls) == 1


def test_run_returns_the_gemini_answer_unchanged(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, lambda prompt: _generated(text="Кандзи 学 значит «учиться»."))

    result = asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert result.response == "Кандзи 学 значит «учиться»."


def test_module_exposes_a_ready_to_use_agent_instance():
    assert isinstance(agent_module.agent, JapaneseLearningAgent)


def test_run_with_no_prior_history_does_not_mention_a_conversation(monkeypatch, tmp_path):
    calls = _stub_generate(monkeypatch, lambda prompt: _generated())

    asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert "Conversation so far" not in calls[0]


def test_run_saves_both_the_user_message_and_the_assistant_response(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, lambda prompt: _generated(text="学 means 'to study'."))
    agent = _agent(tmp_path)

    asyncio.run(agent.run("Explain the kanji 学."))

    assert agent.get_history().messages == [
        {"role": "user", "content": "Explain the kanji 学."},
        {"role": "assistant", "content": "学 means 'to study'."},
    ]


def test_run_includes_previous_turns_in_the_next_gemini_request(monkeypatch, tmp_path):
    responses = iter([_generated(text="学 means 'to study'."), _generated(text="学校で使う言葉です。")])
    calls = _stub_generate(monkeypatch, lambda prompt: next(responses))
    _stub_count_tokens(monkeypatch, lambda text: 50)
    agent = _agent(tmp_path)

    asyncio.run(agent.run("Explain the kanji 学."))
    asyncio.run(agent.run("Give me another example sentence for it."))

    second_prompt = calls[1]
    assert "Explain the kanji 学." in second_prompt
    assert "学 means 'to study'." in second_prompt
    assert "Give me another example sentence for it." in second_prompt


def test_history_survives_agent_recreation(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, lambda prompt: _generated())
    file_path = str(tmp_path / "history.json")
    first_agent = JapaneseLearningAgent(history_storage=AgentHistoryStorage(file_path=file_path))

    asyncio.run(first_agent.run("Explain the kanji 学."))

    restarted_agent = JapaneseLearningAgent(history_storage=AgentHistoryStorage(file_path=file_path))

    assert restarted_agent.get_history().messages == [
        {"role": "user", "content": "Explain the kanji 学."},
        {"role": "assistant", "content": "answer"},
    ]


def test_run_does_not_persist_anything_when_gemini_fails(monkeypatch, tmp_path):
    async def failing_generate(prompt, model=None, temperature=None):
        raise HTTPException(status_code=502, detail="No text found in Gemini response")

    monkeypatch.setattr(agent_module, "generate_text_with_usage", failing_generate)
    agent = _agent(tmp_path)

    with pytest.raises(HTTPException):
        asyncio.run(agent.run("Explain the kanji 学."))

    assert agent.get_history().messages == []


def test_clear_history_removes_all_messages(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, lambda prompt: _generated())
    agent = _agent(tmp_path)
    asyncio.run(agent.run("Explain the kanji 学."))

    agent.clear_history()

    assert agent.get_history().messages == []


# --- token usage -------------------------------------------------------


def test_the_first_message_reports_zero_history_tokens_without_calling_count_tokens(monkeypatch, tmp_path):
    """Scenario 1 (short dialogue): nothing has been said yet, so history is
    genuinely zero - no need to ask Gemini to count tokens in an empty string.
    """
    _stub_generate(monkeypatch, lambda prompt: _generated(input_tokens=30, output_tokens=12))
    count_calls = _stub_count_tokens(monkeypatch, lambda text: 999)

    result = asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert count_calls == []
    assert result.usage.history_tokens == 0
    assert result.usage.current_request_tokens == 30
    assert result.usage.response_tokens == 12
    assert result.usage.total_tokens == 42


def test_history_tokens_come_from_a_real_count_tokens_call_on_the_history_prefix(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, lambda prompt: _generated(input_tokens=100, output_tokens=20))
    count_calls = _stub_count_tokens(monkeypatch, lambda text: 64)
    agent = _agent(tmp_path)

    asyncio.run(agent.run("first message"))
    result = asyncio.run(agent.run("Give me another example for it."))

    assert len(count_calls) == 1
    assert "first message" in count_calls[0]
    assert "answer" in count_calls[0]
    assert "Conversation so far" in count_calls[0]
    # The new message itself is not part of the history being measured.
    assert "Give me another example for it." not in count_calls[0]
    assert result.usage.history_tokens == 64
    assert result.usage.current_request_tokens == 100 - 64
    assert result.usage.total_tokens == 120


def test_history_tokens_grow_as_the_conversation_grows(monkeypatch, tmp_path):
    """Scenario 2 (long dialogue): each additional turn makes the history
    prefix longer, so its real token count must keep increasing.
    """
    _stub_generate(monkeypatch, lambda prompt: _generated(input_tokens=200, output_tokens=20))
    # A real countTokens call on a longer prefix returns a bigger number -
    # simulate that by keying off the prefix's own length.
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    agent = _agent(tmp_path)

    history_token_counts = []
    for message in ["turn one", "turn two", "turn three", "turn four"]:
        result = asyncio.run(agent.run(message))
        history_token_counts.append(result.usage.history_tokens)

    assert history_token_counts == sorted(history_token_counts)
    assert history_token_counts[0] == 0
    assert history_token_counts[-1] > history_token_counts[0]
    assert len(set(history_token_counts)) > 1


def test_usage_is_null_where_gemini_does_not_report_prompt_tokens(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, lambda prompt: _generated(input_tokens=None, output_tokens=20))

    result = asyncio.run(_agent(tmp_path).run("Explain the kanji 学."))

    assert result.usage.current_request_tokens is None
    assert result.usage.total_tokens is None
    assert result.usage.response_tokens == 20
    assert result.usage.history_tokens == 0


def test_a_history_token_count_failure_does_not_fail_the_chat_response(monkeypatch, tmp_path):
    """count_tokens is a best-effort measurement for the usage experiment -
    it must never take down a chat turn that otherwise succeeded.
    """
    _stub_generate(monkeypatch, lambda prompt: _generated(input_tokens=50, output_tokens=10))

    async def failing_count_tokens(text, model=None):
        raise HTTPException(status_code=502, detail="token count unavailable")

    monkeypatch.setattr(agent_module, "count_tokens", failing_count_tokens)
    agent = _agent(tmp_path)
    asyncio.run(agent.run("first message"))

    result = asyncio.run(agent.run("Give me another example for it."))

    assert result.response == "answer"
    assert result.usage.history_tokens is None
    assert result.usage.current_request_tokens is None
    # response tokens and history persistence are unaffected by the failure.
    assert result.usage.response_tokens == 10
    assert len(agent.get_history().messages) == 4


def test_a_context_limit_error_from_gemini_propagates_as_a_proper_error(monkeypatch, tmp_path):
    """Scenario 3: a dialogue too long for the model's context window must
    surface as a clean HTTP error, not a crash, and must not be persisted.
    """
    async def failing_generate(prompt, model=None, temperature=None):
        raise HTTPException(
            status_code=400,
            detail="The input token count exceeds the maximum number of tokens allowed",
        )

    monkeypatch.setattr(agent_module, "generate_text_with_usage", failing_generate)
    agent = _agent(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(agent.run("a very long message"))

    assert exc_info.value.status_code == 400
    assert agent.get_history().messages == []


# --- compression: off (the unchanged baseline) -----------------------------


def test_without_compression_the_whole_conversation_is_kept(monkeypatch, tmp_path):
    """Compression scenario 1: the baseline run. Nothing is summarised and
    nothing is dropped - eight turns are still eight turns on disk."""
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    agent = _agent(tmp_path)

    _hold_a_conversation(agent, turns=8)
    history = agent.get_history()

    assert len(history.messages) == 16
    assert history.summary == ""
    assert history.messages[0] == {"role": "user", "content": "question 0"}


def test_without_compression_even_the_oldest_turn_is_still_sent(monkeypatch, tmp_path):
    calls = _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))

    _hold_a_conversation(_agent(tmp_path), turns=8)

    assert "question 0" in calls[-1]
    assert "Summary of the earlier part" not in calls[-1]


def test_without_compression_gemini_is_never_asked_for_a_summary(monkeypatch, tmp_path):
    """No summarisation calls means the baseline costs exactly what it
    always did - nothing extra creeps into the comparison."""
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    summary_calls = _stub_summary(monkeypatch, lambda prompt: _generated(text="сводка"))

    _hold_a_conversation(_agent(tmp_path), turns=8)

    assert summary_calls == []


# --- compression: on -------------------------------------------------------


def test_with_compression_old_messages_are_replaced_by_the_summary(monkeypatch, tmp_path):
    """Compression scenario 2: the same eight turns, stored as a summary
    plus the recent window instead of the full transcript."""
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    _stub_summary(monkeypatch, lambda prompt: _generated(text="Разбирали 学習, 勉強 и 〜につれて."))
    agent = _compressing_agent(tmp_path)

    _hold_a_conversation(agent, turns=8)
    history = agent.get_history()

    assert history.summary == "Разбирали 学習, 勉強 и 〜につれて."
    assert len(history.messages) == 6
    assert {"role": "user", "content": "question 0"} not in history.messages


def test_the_six_newest_messages_are_stored_untouched(monkeypatch, tmp_path):
    """Scenario 4: whatever compression does to the older part, the last six
    messages stay exactly as they were written."""
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    _stub_summary(monkeypatch, lambda prompt: _generated(text="сводка"))
    agent = _compressing_agent(tmp_path)

    _hold_a_conversation(agent, turns=8)

    assert agent.get_history().messages == [
        {"role": "user", "content": "question 5"},
        {"role": "assistant", "content": "answer 5"},
        {"role": "user", "content": "question 6"},
        {"role": "assistant", "content": "answer 6"},
        {"role": "user", "content": "question 7"},
        {"role": "assistant", "content": "answer 7"},
    ]


def test_with_compression_gemini_receives_the_summary_and_the_recent_messages(monkeypatch, tmp_path):
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    _stub_summary(monkeypatch, lambda prompt: _generated(text="Разбирали 学習 и 勉強."))
    agent = _compressing_agent(tmp_path)
    calls = _stub_generate(monkeypatch, _numbered_answers())

    _hold_a_conversation(agent, turns=9)
    last_prompt = calls[-1]

    assert "Разбирали 学習 и 勉強." in last_prompt
    assert "question 7" in last_prompt
    assert "question 0" not in last_prompt


def test_nothing_is_summarised_before_the_threshold_is_reached(monkeypatch, tmp_path):
    """Seven turns is fourteen messages - eight past the recent window, two
    short of the ten that trigger a rewrite."""
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    summary_calls = _stub_summary(monkeypatch, lambda prompt: _generated(text="сводка"))
    agent = _compressing_agent(tmp_path)

    _hold_a_conversation(agent, turns=7)

    assert summary_calls == []
    assert agent.get_history().summary == ""
    assert len(agent.get_history().messages) == 14


def test_the_summary_is_rewritten_once_every_ten_messages(monkeypatch, tmp_path):
    """Thirteen turns cross the threshold twice: at sixteen messages, and
    again ten messages later."""
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    summary_calls = _stub_summary(monkeypatch, _numbered_answers(prefix="сводка"))
    agent = _compressing_agent(tmp_path)

    _hold_a_conversation(agent, turns=13)

    assert len(summary_calls) == 2
    assert agent.get_history().summary == "сводка 1"
    assert len(agent.get_history().messages) == 6


def test_each_rewrite_builds_on_the_previous_summary(monkeypatch, tmp_path):
    """Scenario 3: context from the start of the conversation is carried
    forward through every rewrite instead of being dropped batch by batch."""
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    summary_calls = _stub_summary(monkeypatch, _numbered_answers(prefix="сводка"))

    _hold_a_conversation(_compressing_agent(tmp_path), turns=13)

    assert "question 0" in summary_calls[0]
    assert "сводка 0" in summary_calls[1]
    assert "question 7" in summary_calls[1]


def test_the_summary_and_recent_messages_survive_a_restart(monkeypatch, tmp_path):
    """Scenario 5: a fresh agent pointed at the same file picks the
    conversation up where it was left, summary included."""
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    _stub_summary(monkeypatch, lambda prompt: _generated(text="Учащийся разбирал 学習."))
    _hold_a_conversation(_compressing_agent(tmp_path), turns=8)

    restarted = _compressing_agent(tmp_path)
    history = restarted.get_history()

    assert history.summary == "Учащийся разбирал 学習."
    assert history.messages[-1] == {"role": "assistant", "content": "answer 7"}
    assert len(history.messages) == 6


def test_after_a_restart_the_summary_is_sent_to_gemini_again(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    _stub_summary(monkeypatch, lambda prompt: _generated(text="Учащийся разбирал 学習."))
    _hold_a_conversation(_compressing_agent(tmp_path), turns=8)

    calls = _stub_generate(monkeypatch, _numbered_answers())
    asyncio.run(_compressing_agent(tmp_path).run("а что там было про 学習?"))

    assert "Учащийся разбирал 学習." in calls[0]


def test_a_failed_summarisation_neither_fails_the_turn_nor_loses_messages(monkeypatch, tmp_path):
    """The learner's answer already arrived; a summarising hiccup must not
    turn it into an error, and must not throw away the messages it failed to
    summarise - they stay verbatim and the rewrite is retried next turn."""
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))

    async def failing_summary(prompt, model=None, temperature=None):
        raise HTTPException(status_code=502, detail="No text found in Gemini response")

    monkeypatch.setattr(compressor_module, "generate_text_with_usage", failing_summary)
    agent = _compressing_agent(tmp_path)

    responses = _hold_a_conversation(agent, turns=8)
    history = agent.get_history()

    assert responses[-1].response == "answer 7"
    assert history.summary == ""
    assert len(history.messages) == 16
    assert history.messages[0] == {"role": "user", "content": "question 0"}


def test_a_recovered_summarisation_compresses_on_the_next_turn(monkeypatch, tmp_path):
    async def failing_summary(prompt, model=None, temperature=None):
        raise HTTPException(status_code=502, detail="No text found in Gemini response")

    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    monkeypatch.setattr(compressor_module, "generate_text_with_usage", failing_summary)
    agent = _compressing_agent(tmp_path)
    _hold_a_conversation(agent, turns=8)

    _stub_summary(monkeypatch, lambda prompt: _generated(text="сводка"))
    _hold_a_conversation(agent, turns=1, start=8)

    assert agent.get_history().summary == "сводка"
    assert len(agent.get_history().messages) == 6


# --- compression: the token comparison -------------------------------------


def _measure_dialogue(monkeypatch, agent, turns):
    """Run the same dialogue and report the history tokens each turn sent.
    Token counts key off the real length of the text, so the numbers move
    the way a real countTokens call would."""
    answers = iter(
        f"answer {index}: " + "подробное объяснение грамматики и примеры " * 10 for index in range(turns)
    )
    _stub_generate(monkeypatch, lambda prompt: _generated(text=next(answers), input_tokens=len(prompt)))
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    _stub_summary(
        monkeypatch,
        lambda prompt: _generated(text="Сводка: разбирали 学習, 勉強, 〜につれて.", input_tokens=len(prompt)),
    )

    return [result.usage.history_tokens for result in _hold_a_conversation(agent, turns=turns)]


def test_compression_sends_far_fewer_tokens_over_the_same_dialogue(monkeypatch, tmp_path):
    """Scenario 6, the point of the whole feature: the same twenty-four
    turns, measured both ways."""
    baseline = _measure_dialogue(monkeypatch, _agent(tmp_path, name="plain.json"), turns=24)
    compressed = _measure_dialogue(monkeypatch, _compressing_agent(tmp_path), turns=24)

    # The last turn of a long dialogue is where the two modes differ most.
    assert compressed[-1] < baseline[-1] / 3
    # And the dialogue as a whole costs substantially less to send.
    assert sum(compressed) < sum(baseline) / 2


def test_without_compression_the_context_sent_grows_without_bound(monkeypatch, tmp_path):
    baseline = _measure_dialogue(monkeypatch, _agent(tmp_path, name="plain.json"), turns=24)

    assert baseline == sorted(baseline)
    assert baseline[-1] > baseline[len(baseline) // 2] > baseline[1]


def test_with_compression_the_context_sent_stays_bounded(monkeypatch, tmp_path):
    """It still rises and falls between rewrites, but it never runs away:
    the peak of the second half is no worse than the peak of the first."""
    compressed = _measure_dialogue(monkeypatch, _compressing_agent(tmp_path), turns=24)
    first_half, second_half = compressed[:12], compressed[12:]

    assert max(second_half) <= max(first_half) * 1.5


# --- the usage log ---------------------------------------------------------


def test_every_request_is_recorded_for_the_comparison(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    agent = _agent(tmp_path)

    _hold_a_conversation(agent, turns=3)

    assert len(agent.get_usage()) == 3


def test_a_usage_record_carries_the_real_token_counts(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, lambda prompt: _generated(input_tokens=40, output_tokens=8))
    agent = _agent(tmp_path)

    asyncio.run(agent.run("Explain 学."))
    record = agent.get_usage()[0]

    assert record["current_request_tokens"] == 40
    assert record["history_tokens"] == 0
    assert record["response_tokens"] == 8
    assert record["total_tokens"] == 48
    assert record["timestamp"]


def test_a_usage_record_says_which_mode_produced_it(monkeypatch, tmp_path):
    """Without this the two experiment runs could not be told apart."""
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    _stub_summary(monkeypatch, lambda prompt: _generated(text="сводка"))
    plain, compressed = _agent(tmp_path, name="plain.json"), _compressing_agent(tmp_path)

    _hold_a_conversation(plain, turns=1)
    _hold_a_conversation(compressed, turns=1)

    assert plain.get_usage()[0]["compression_enabled"] is False
    assert compressed.get_usage()[0]["compression_enabled"] is True


def test_a_usage_record_says_how_much_context_was_sent(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    _stub_summary(monkeypatch, lambda prompt: _generated(text="сводка"))
    agent = _compressing_agent(tmp_path)

    _hold_a_conversation(agent, turns=9)
    first, last = agent.get_usage()[0], agent.get_usage()[-1]

    assert (first["messages_sent"], first["summary_used"]) == (0, False)
    assert (last["messages_sent"], last["summary_used"]) == (6, True)


def test_a_usage_record_includes_what_the_summarising_itself_cost(monkeypatch, tmp_path):
    """Compression is not free, so its own token spend is recorded next to
    the saving rather than quietly left out of the comparison."""
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    _stub_summary(monkeypatch, lambda prompt: _generated(text="сводка", input_tokens=300, output_tokens=40))
    agent = _compressing_agent(tmp_path)

    _hold_a_conversation(agent, turns=8)
    summarisation_costs = [record["summarization_tokens"] for record in agent.get_usage()]

    assert summarisation_costs == [0, 0, 0, 0, 0, 0, 0, 340]


def test_a_failure_to_record_usage_does_not_fail_the_turn(monkeypatch, tmp_path):
    _stub_generate(monkeypatch, lambda prompt: _generated())
    agent = _agent(tmp_path)

    def failing_append(entry):
        raise OSError("disk full")

    monkeypatch.setattr(agent._usage_log, "append", failing_append)

    result = asyncio.run(agent.run("Explain 学."))

    assert result.response == "answer"
    assert len(agent.get_history().messages) == 2


# --- the mode can be chosen per request ------------------------------------


def _compressed_dialogue(monkeypatch, agent, turns=8, compression_enabled=None):
    _stub_generate(monkeypatch, _numbered_answers())
    _stub_count_tokens(monkeypatch, lambda text: len(text))
    _stub_summary(monkeypatch, lambda prompt: _generated(text="сводка", output_tokens=42))

    return [
        asyncio.run(agent.run(f"question {index}", compression_enabled))
        for index in range(turns)
    ]


def test_a_request_can_ask_for_compression_the_agent_is_not_configured_for(monkeypatch, tmp_path):
    """The client states the mode; the configured default is the fallback."""
    agent = _agent(tmp_path)  # configured off

    _compressed_dialogue(monkeypatch, agent, compression_enabled=True)

    assert agent.get_history().summary == "сводка"
    assert len(agent.get_history().messages) == 6


def test_a_request_can_opt_out_of_compression_the_agent_is_configured_for(monkeypatch, tmp_path):
    agent = _compressing_agent(tmp_path)  # configured on

    _compressed_dialogue(monkeypatch, agent, compression_enabled=False)

    assert agent.get_history().summary == ""
    assert len(agent.get_history().messages) == 16


def test_a_request_that_states_no_mode_uses_the_configured_one(monkeypatch, tmp_path):
    agent = _compressing_agent(tmp_path)

    _compressed_dialogue(monkeypatch, agent, compression_enabled=None)

    assert agent.get_history().summary == "сводка"


def test_the_response_reports_the_mode_and_what_is_being_kept(monkeypatch, tmp_path):
    results = _compressed_dialogue(monkeypatch, _agent(tmp_path), compression_enabled=True)
    status = results[-1].compression

    assert status.enabled is True
    assert status.summary_tokens == 42
    assert status.recent_messages == 6


def test_the_status_reports_no_summary_when_compression_is_off(monkeypatch, tmp_path):
    results = _compressed_dialogue(monkeypatch, _agent(tmp_path), turns=3, compression_enabled=False)
    status = results[-1].compression

    assert status.enabled is False
    assert status.summary_tokens == 0
    assert status.recent_messages == 6


def test_the_summarys_token_count_survives_a_restart(monkeypatch, tmp_path):
    _compressed_dialogue(monkeypatch, _compressing_agent(tmp_path))

    assert _compressing_agent(tmp_path).get_history().summary_tokens == 42


def test_the_usage_log_records_the_mode_the_request_asked_for(monkeypatch, tmp_path):
    agent = _agent(tmp_path)

    _compressed_dialogue(monkeypatch, agent, turns=1, compression_enabled=True)

    assert agent.get_usage()[0]["compression_enabled"] is True
