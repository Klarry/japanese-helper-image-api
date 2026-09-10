import asyncio

import pytest
from fastapi import HTTPException

from app.services import agent_history_compressor as compressor_module
from app.services.agent_history_compressor import HistoryCompressor
from app.services.gemini_service import GeneratedText


def _stub_summary(monkeypatch, handler):
    """handler(prompt) -> GeneratedText. Replaces the shared gemini_service
    call the compressor uses to write a summary."""
    calls = []

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        calls.append(prompt)
        return handler(prompt)

    monkeypatch.setattr(compressor_module, "generate_text_with_usage", fake_generate_text_with_usage)
    return calls


def _generated(text="краткое содержание", input_tokens=40, output_tokens=12):
    return GeneratedText(text=text, input_tokens=input_tokens, output_tokens=output_tokens)


def _messages(count, prefix="message"):
    return [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"{prefix} {index}"}
        for index in range(count)
    ]


def _compressor(recent_messages_kept=6, summary_update_threshold=10):
    return HistoryCompressor(
        recent_messages_kept=recent_messages_kept,
        summary_update_threshold=summary_update_threshold,
    )


# --- when the summary is due ------------------------------------------------


def test_nothing_is_compressed_while_only_the_recent_window_exists():
    assert _compressor().needs_compression(_messages(6)) is False


def test_nothing_is_compressed_until_the_threshold_of_old_messages_is_reached():
    """Fifteen messages means nine older than the recent six - one short."""
    assert _compressor().needs_compression(_messages(15)) is False


def test_compression_is_due_once_ten_messages_are_older_than_the_recent_six():
    assert _compressor().needs_compression(_messages(16)) is True


def test_the_thresholds_are_configurable():
    assert _compressor(recent_messages_kept=2, summary_update_threshold=4).needs_compression(_messages(6))


# --- what compression does --------------------------------------------------


def test_the_six_newest_messages_come_back_untouched(monkeypatch):
    """Scenario 4: compression never rewrites, trims or reorders the
    messages that are still inside the recent window."""
    _stub_summary(monkeypatch, lambda prompt: _generated())
    messages = _messages(16)

    result = asyncio.run(_compressor().compress("", messages))

    assert result.messages == messages[-6:]
    assert len(result.messages) == 6


def test_the_older_messages_are_replaced_by_the_generated_summary(monkeypatch):
    _stub_summary(monkeypatch, lambda prompt: _generated(text="Разбирали 学習 и 勉強."))

    result = asyncio.run(_compressor().compress("", _messages(16)))

    assert result.summary == "Разбирали 学習 и 勉強."


def test_the_summary_prompt_contains_the_messages_being_dropped(monkeypatch):
    """Scenario 3: what is about to leave the context window has to be in
    front of the model that writes the summary."""
    prompts = _stub_summary(monkeypatch, lambda prompt: _generated())

    asyncio.run(_compressor().compress("", _messages(16)))

    assert "message 0" in prompts[0]
    assert "message 9" in prompts[0]


def test_the_summary_prompt_does_not_contain_the_recent_messages(monkeypatch):
    """They are still sent verbatim on every turn - summarising them too
    would just pay for the same context twice."""
    prompts = _stub_summary(monkeypatch, lambda prompt: _generated())

    asyncio.run(_compressor().compress("", _messages(16)))

    assert "message 15" not in prompts[0]


def test_the_previous_summary_is_folded_into_the_new_one(monkeypatch):
    """Scenario 3: the summary is rolling, so context from the very start of
    the conversation survives every later rewrite instead of being dropped."""
    prompts = _stub_summary(monkeypatch, lambda prompt: _generated())

    asyncio.run(_compressor().compress("Учащийся готовится к N4.", _messages(16)))

    assert "Учащийся готовится к N4." in prompts[0]


def test_the_summary_prompt_asks_to_keep_what_matters(monkeypatch):
    prompts = _stub_summary(monkeypatch, lambda prompt: _generated())

    asyncio.run(_compressor().compress("", _messages(16)))

    assert "kanji" in prompts[0]
    assert "grammar" in prompts[0]


def test_compression_uses_the_shared_gemini_service_once(monkeypatch):
    """No second mechanism for talking to the LLM - the same
    generate_text_with_usage every other feature uses, one call per rewrite."""
    prompts = _stub_summary(monkeypatch, lambda prompt: _generated())

    asyncio.run(_compressor().compress("", _messages(16)))

    assert len(prompts) == 1


def test_compression_reports_what_the_rewrite_itself_cost(monkeypatch):
    _stub_summary(monkeypatch, lambda prompt: _generated(input_tokens=400, output_tokens=90))

    result = asyncio.run(_compressor().compress("", _messages(16)))

    assert result.tokens_used == 490


def test_compression_reports_how_big_the_new_summary_is(monkeypatch):
    """The rewrite's output-token count is the summary's own size - a real
    number that comes back with the call, so nothing has to be recounted."""
    _stub_summary(monkeypatch, lambda prompt: _generated(input_tokens=400, output_tokens=90))

    result = asyncio.run(_compressor().compress("", _messages(16)))

    assert result.summary_tokens == 90


def test_unreported_token_counts_are_treated_as_zero_cost(monkeypatch):
    _stub_summary(monkeypatch, lambda prompt: _generated(input_tokens=None, output_tokens=None))

    result = asyncio.run(_compressor().compress("", _messages(16)))

    assert result.tokens_used == 0


def test_a_gemini_failure_propagates_to_the_caller(monkeypatch):
    """The compressor does not decide what to do about a failed rewrite -
    the agent does, and it keeps the messages instead."""

    async def failing(prompt, model=None, temperature=None):
        raise HTTPException(status_code=502, detail="No text found in Gemini response")

    monkeypatch.setattr(compressor_module, "generate_text_with_usage", failing)

    with pytest.raises(HTTPException):
        asyncio.run(_compressor().compress("", _messages(16)))
