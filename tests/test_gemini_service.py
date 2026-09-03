import asyncio

from app.services import gemini_service


def test_generate_text_without_temperature_omits_it_from_payload(monkeypatch):
    """Regression guard: existing callers (description, kanji word set) call
    generate_text(prompt) with no temperature and must keep sending the exact
    same payload shape as before this field was added.
    """
    captured_payload = {}

    async def fake_post_to_gemini(payload):
        captured_payload.update(payload)
        return {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}

    monkeypatch.setattr(gemini_service, "_post_to_gemini", fake_post_to_gemini)

    result = asyncio.run(gemini_service.generate_text("a prompt"))

    assert result == "hello"
    assert captured_payload == {"model": gemini_service.TEXT_MODEL, "input": "a prompt"}
    assert "temperature" not in captured_payload


def test_generate_text_with_temperature_includes_it_in_payload(monkeypatch):
    captured_payload = {}

    async def fake_post_to_gemini(payload):
        captured_payload.update(payload)
        return {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}

    monkeypatch.setattr(gemini_service, "_post_to_gemini", fake_post_to_gemini)

    result = asyncio.run(gemini_service.generate_text("a prompt", temperature=0.7))

    assert result == "hello"
    assert captured_payload["temperature"] == 0.7
