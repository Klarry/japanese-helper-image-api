"""Day 30, against a real running service - skipped unless asked for.

    DAY30_BASE_URL=http://192.168.10.35:8000 pytest -m integration tests/test_day30_network.py

This is the test that actually crosses a network and reaches a real model.
The unit tests in ``test_day30_service.py`` use a stand-in for Ollama and do
not claim to.
"""

import os

import httpx
import pytest

BASE_URL = os.getenv("DAY30_BASE_URL", "").rstrip("/")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not BASE_URL, reason="set DAY30_BASE_URL to the running service"),
]


def test_the_service_is_reachable_and_ready():
    response = httpx.get(f"{BASE_URL}/local-llm/health", params={"strict": "true"}, timeout=10)

    assert response.status_code == 200, response.text
    assert response.json()["ready"] is True


def test_a_real_model_answers_over_the_network():
    response = httpx.post(f"{BASE_URL}/local-chat", json={"message": "Reply with one word: hello."}, timeout=300)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provider"] == "ollama"
    assert body["response"].strip()
    assert body["timing"]["generation_ms"] > 0
