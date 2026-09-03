from fastapi.testclient import TestClient

from app.main import app
from app.services import temperature_description_service as service

client = TestClient(app)


def _mock_generate_text(monkeypatch, response_text: str):
    async def fake_generate_text(prompt: str, temperature: float | None = None) -> str:
        return response_text

    monkeypatch.setattr(service, "generate_text", fake_generate_text)


def test_temperature_description_success(monkeypatch):
    _mock_generate_text(monkeypatch, "sentence: 学校に行きます。 translation: Я иду в школу.")

    response = client.post(
        "/temperature-description",
        json={"kanji": "学", "temperature": 0.7},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sentence"] == "学校に行きます。"
    assert body["translation"] == "Я иду в школу."
    assert body["temperature"] == 0.7


def test_temperature_description_accepts_all_supported_temperatures(monkeypatch):
    _mock_generate_text(monkeypatch, "sentence: 学校に行きます。 translation: Я иду в школу.")

    for temperature in (0.0, 0.7, 1.2):
        response = client.post(
            "/temperature-description",
            json={"kanji": "学", "temperature": temperature},
        )
        assert response.status_code == 200
        assert response.json()["temperature"] == temperature


def test_temperature_description_rejects_unsupported_temperature():
    response = client.post(
        "/temperature-description",
        json={"kanji": "学", "temperature": 0.5},
    )

    assert response.status_code == 422


def test_temperature_description_rejects_empty_kanji():
    response = client.post(
        "/temperature-description",
        json={"kanji": "  ", "temperature": 0.0},
    )

    assert response.status_code == 422


def test_temperature_description_llm_parse_failure(monkeypatch):
    _mock_generate_text(monkeypatch, "I refuse to answer this request.")

    response = client.post(
        "/temperature-description",
        json={"kanji": "学", "temperature": 1.2},
    )

    assert response.status_code == 502
