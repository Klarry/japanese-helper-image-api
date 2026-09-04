import asyncio

from fastapi import HTTPException

from app.services import model_comparison_service as service
from app.services.gemini_service import GeneratedText


def _stub_generation(monkeypatch, handler):
    calls = []

    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        return handler(model)

    monkeypatch.setattr(service, "generate_text_with_usage", fake_generate_text_with_usage)
    return calls


def test_build_prompt_substitutes_the_placeholder():
    prompt = service._build_prompt("学", "Write a sentence using {KANJI}, please.")

    assert prompt == "Write a sentence using 学, please."


def test_build_prompt_appends_the_kanji_when_there_is_no_placeholder():
    prompt = service._build_prompt("学", "Write a sentence.")

    assert prompt.startswith("Write a sentence.")
    assert prompt.endswith("学")


def test_every_model_receives_the_identical_prompt(monkeypatch):
    calls = _stub_generation(
        monkeypatch,
        lambda model: GeneratedText(f"answer from {model}", 42, 18),
    )

    response = asyncio.run(
        service.compare_models("学", "Use {KANJI}.", ["model-a", "model-b", "model-c"])
    )

    assert [call["model"] for call in calls] == ["model-a", "model-b", "model-c"]
    assert {call["prompt"] for call in calls} == {"Use 学."}
    assert response.prompt == "Use 学."
    assert [result.model for result in response.results] == ["model-a", "model-b", "model-c"]
    assert [result.text for result in response.results] == [
        "answer from model-a",
        "answer from model-b",
        "answer from model-c",
    ]


def test_token_usage_is_passed_through(monkeypatch):
    _stub_generation(monkeypatch, lambda model: GeneratedText("answer", 42, 18))

    response = asyncio.run(service.compare_models("学", "Use {KANJI}.", ["model-a"]))

    assert response.results[0].input_tokens == 42
    assert response.results[0].output_tokens == 18


def test_missing_token_usage_stays_none(monkeypatch):
    _stub_generation(monkeypatch, lambda model: GeneratedText("answer", None, None))

    response = asyncio.run(service.compare_models("学", "Use {KANJI}.", ["model-a"]))

    assert response.results[0].input_tokens is None
    assert response.results[0].output_tokens is None


def test_response_time_is_measured_per_model(monkeypatch):
    _stub_generation(monkeypatch, lambda model: GeneratedText("answer", None, None))

    response = asyncio.run(service.compare_models("学", "Use {KANJI}.", ["model-a"]))

    assert response.results[0].response_time_ms >= 0


def test_one_failing_model_does_not_sink_the_others(monkeypatch):
    def handler(model):
        if model == "broken":
            raise HTTPException(status_code=404, detail="model not found")

        return GeneratedText(f"answer from {model}", 1, 2)

    _stub_generation(monkeypatch, handler)

    response = asyncio.run(
        service.compare_models("学", "Use {KANJI}.", ["broken", "model-b"])
    )

    broken, working = response.results

    assert broken.model == "broken"
    assert broken.text is None
    assert broken.error == "model not found"
    assert broken.response_time_ms >= 0

    assert working.model == "model-b"
    assert working.text == "answer from model-b"
    assert working.error is None
