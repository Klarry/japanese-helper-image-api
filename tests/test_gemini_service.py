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


def test_generate_text_with_temperature_uses_generate_content_endpoint(monkeypatch):
    """The Interactions API (_post_to_gemini) has no temperature parameter -
    confirmed by Google's docs and by the API itself ("Unknown parameter
    'temperature'"). A temperature-controlled call must go to the classic
    generateContent endpoint instead, via generationConfig.temperature.
    """
    captured = {}

    async def fake_post(url, payload):
        captured["url"] = url
        captured["payload"] = payload
        return {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}

    async def fail_post_to_gemini(payload):
        raise AssertionError("temperature calls must not use the Interactions API")

    monkeypatch.setattr(gemini_service, "_post", fake_post)
    monkeypatch.setattr(gemini_service, "_post_to_gemini", fail_post_to_gemini)

    result = asyncio.run(gemini_service.generate_text("a prompt", temperature=0.7))

    assert result == "hello"
    assert captured["url"] == (
        f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_service.TEXT_MODEL}:generateContent"
    )
    assert captured["payload"] == {
        "contents": [{"parts": [{"text": "a prompt"}]}],
        "generationConfig": {"temperature": 0.7},
    }


def test_generate_text_defaults_to_the_configured_text_model(monkeypatch):
    """The existing endpoints call generate_text(prompt) with no model, and
    must keep hitting TEXT_MODEL exactly as before.
    """
    captured_payload = {}

    async def fake_post_to_gemini(payload):
        captured_payload.update(payload)
        return {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}

    monkeypatch.setattr(gemini_service, "_post_to_gemini", fake_post_to_gemini)

    asyncio.run(gemini_service.generate_text("a prompt"))

    assert captured_payload["model"] == gemini_service.TEXT_MODEL


def test_generate_text_propagates_an_explicit_model(monkeypatch):
    captured_payload = {}

    async def fake_post_to_gemini(payload):
        captured_payload.update(payload)
        return {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}

    monkeypatch.setattr(gemini_service, "_post_to_gemini", fake_post_to_gemini)

    result = asyncio.run(gemini_service.generate_text("a prompt", model="some-other-model"))

    assert result == "hello"
    assert captured_payload == {"model": "some-other-model", "input": "a prompt"}


def test_generate_text_propagates_the_model_into_the_generate_content_url(monkeypatch):
    captured = {}

    async def fake_post(url, payload):
        captured["url"] = url
        return {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}

    monkeypatch.setattr(gemini_service, "_post", fake_post)

    asyncio.run(
        gemini_service.generate_text("a prompt", temperature=0.7, model="some-other-model")
    )

    assert captured["url"].endswith("/models/some-other-model:generateContent")


def test_generate_text_with_usage_reads_interactions_token_usage(monkeypatch):
    async def fake_post_to_gemini(payload):
        return {
            "output": [{"type": "text", "data": "hello"}],
            "usage": {
                "total_input_tokens": 42,
                "total_output_tokens": 18,
                "total_tokens": 60,
            },
        }

    monkeypatch.setattr(gemini_service, "_post_to_gemini", fake_post_to_gemini)

    generated = asyncio.run(gemini_service.generate_text_with_usage("a prompt"))

    assert generated.text == "hello"
    assert generated.input_tokens == 42
    assert generated.output_tokens == 18


def test_generate_text_with_usage_reads_generate_content_token_usage(monkeypatch):
    async def fake_post(url, payload):
        return {
            "candidates": [{"content": {"parts": [{"text": "hello"}]}}],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 7,
                "totalTokenCount": 18,
            },
        }

    monkeypatch.setattr(gemini_service, "_post", fake_post)

    generated = asyncio.run(
        gemini_service.generate_text_with_usage("a prompt", temperature=0.0)
    )

    assert generated.input_tokens == 11
    assert generated.output_tokens == 7


def test_generate_text_with_usage_returns_none_when_usage_is_absent(monkeypatch):
    async def fake_post_to_gemini(payload):
        return {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}

    monkeypatch.setattr(gemini_service, "_post_to_gemini", fake_post_to_gemini)

    generated = asyncio.run(gemini_service.generate_text_with_usage("a prompt"))

    assert generated.input_tokens is None
    assert generated.output_tokens is None
