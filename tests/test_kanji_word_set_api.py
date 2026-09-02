from fastapi.testclient import TestClient

from app.main import app
from app.services import kanji_word_set_service as service

client = TestClient(app)


def _mock_generate_text(monkeypatch, response_text: str):
    async def fake_generate_text(prompt: str) -> str:
        return response_text

    monkeypatch.setattr(service, "generate_text", fake_generate_text)


def test_kanji_word_set_direct(monkeypatch):
    _mock_generate_text(monkeypatch, "words: 学生, 学校 cost: 4 value: 16")

    response = client.post(
        "/kanji-word-set",
        json={"kanji": "学", "experimentType": "DIRECT"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["words"] == ["学生", "学校"]
    assert body["cost"] == 4
    assert body["value"] == 16
    assert "学" in body["prompt"]
    assert "{KANJI}" not in body["prompt"]


def test_kanji_word_set_step_by_step(monkeypatch):
    _mock_generate_text(monkeypatch, "words: 火山, 火事 cost: 4 value: 14")

    response = client.post(
        "/kanji-word-set",
        json={"kanji": "火", "experimentType": "STEP_BY_STEP"},
    )

    assert response.status_code == 200
    assert response.json()["words"] == ["火山", "火事"]


def test_kanji_word_set_prompt_generation(monkeypatch):
    _mock_generate_text(monkeypatch, "words: 水曜日, 水 cost: 3 value: 15")

    response = client.post(
        "/kanji-word-set",
        json={"kanji": "水", "experimentType": "PROMPT"},
    )

    assert response.status_code == 200
    assert response.json()["cost"] == 3


def test_kanji_word_set_experts(monkeypatch):
    _mock_generate_text(monkeypatch, "words: 木曜日, 木 cost: 3 value: 15")

    response = client.post(
        "/kanji-word-set",
        json={"kanji": "木", "experimentType": "EXPERTS"},
    )

    assert response.status_code == 200
    assert response.json()["value"] == 15


def test_kanji_word_set_invalid_experiment_type():
    response = client.post(
        "/kanji-word-set",
        json={"kanji": "学", "experimentType": "NOT_A_TYPE"},
    )

    assert response.status_code == 422


def test_kanji_word_set_empty_kanji():
    response = client.post(
        "/kanji-word-set",
        json={"kanji": "  ", "experimentType": "DIRECT"},
    )

    assert response.status_code == 422


def test_kanji_word_set_llm_parse_failure(monkeypatch):
    _mock_generate_text(monkeypatch, "I refuse to answer this request.")

    response = client.post(
        "/kanji-word-set",
        json={"kanji": "学", "experimentType": "DIRECT"},
    )

    assert response.status_code == 502
