from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.services import model_comparison_service as service
from app.services.gemini_service import GeneratedText

client = TestClient(app)


def _stub_generation(monkeypatch, handler):
    async def fake_generate_text_with_usage(prompt, model=None, temperature=None):
        return handler(model)

    monkeypatch.setattr(service, "generate_text_with_usage", fake_generate_text_with_usage)


def test_model_comparison_success(monkeypatch):
    _stub_generation(monkeypatch, lambda model: GeneratedText(f"answer from {model}", 42, 18))

    response = client.post(
        "/model-comparison",
        json={"kanji": "学", "prompt": "Use {KANJI}.", "models": ["model-a", "model-b"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prompt"] == "Use 学."
    assert len(body["results"]) == 2

    first = body["results"][0]
    assert first["model"] == "model-a"
    assert first["text"] == "answer from model-a"
    assert first["input_tokens"] == 42
    assert first["output_tokens"] == 18
    assert first["error"] is None
    assert isinstance(first["response_time_ms"], int)


def test_model_comparison_returns_partial_results(monkeypatch):
    def handler(model):
        if model == "broken":
            raise HTTPException(status_code=400, detail="unknown model")

        return GeneratedText("ok", None, None)

    _stub_generation(monkeypatch, handler)

    response = client.post(
        "/model-comparison",
        json={"kanji": "学", "prompt": "Use {KANJI}.", "models": ["broken", "model-b"]},
    )

    assert response.status_code == 200
    broken, working = response.json()["results"]
    assert broken["error"] == "unknown model"
    assert broken["text"] is None
    assert working["text"] == "ok"
    assert working["error"] is None


def test_model_comparison_reports_missing_token_usage_as_null(monkeypatch):
    _stub_generation(monkeypatch, lambda model: GeneratedText("ok", None, None))

    response = client.post(
        "/model-comparison",
        json={"kanji": "学", "prompt": "Use {KANJI}.", "models": ["model-a"]},
    )

    result = response.json()["results"][0]
    assert result["input_tokens"] is None
    assert result["output_tokens"] is None


def test_model_comparison_rejects_empty_models():
    response = client.post(
        "/model-comparison",
        json={"kanji": "学", "prompt": "Use {KANJI}.", "models": []},
    )

    assert response.status_code == 422


def test_model_comparison_rejects_blank_kanji_and_prompt():
    for payload in (
        {"kanji": " ", "prompt": "Use {KANJI}.", "models": ["model-a"]},
        {"kanji": "学", "prompt": "  ", "models": ["model-a"]},
    ):
        assert client.post("/model-comparison", json=payload).status_code == 422


def test_model_comparison_rejects_too_many_models():
    response = client.post(
        "/model-comparison",
        json={
            "kanji": "学",
            "prompt": "Use {KANJI}.",
            "models": [f"model-{index}" for index in range(6)],
        },
    )

    assert response.status_code == 422
