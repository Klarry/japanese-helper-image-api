"""The backend's client for the JLPT vocabulary API, in-process."""

import asyncio

import httpx
import pytest

from app.services import jlpt_vocab_api
from app.services.jlpt_vocab_api import JlptVocabApiError, search_words

from tests.jlpt_api_standin import LIVE_ENTRIES


def _answer(status=200, json_body=None, text=None):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=json_body)

    return httpx.MockTransport(handler), seen


def _search(word, transport):
    return asyncio.run(search_words(word, transport=transport))


def test_a_word_is_looked_up_on_the_api_the_app_uses(monkeypatch):
    monkeypatch.delenv("JLPT_VOCAB_API_URL", raising=False)
    transport, seen = _answer(json_body={"total": 1, "offset": 0, "limit": 10, "words": LIVE_ENTRIES["学習"]})

    words = _search("学習", transport)

    assert seen[0].url.host == "jlpt-vocab-api.vercel.app"
    assert seen[0].url.path == "/api/words"
    assert seen[0].url.params["word"] == "学習"
    assert words[0].furigana == "がくしゅう"
    assert words[0].meaning == "study, learning"
    assert words[0].level == 3


def test_every_entry_spelled_that_way_comes_back():
    transport, _ = _answer(json_body={"total": 2, "words": LIVE_ENTRIES["勉強"]})

    assert [word.furigana for word in _search("勉強", transport)] == ["べんきょうする", "べんきょう"]


def test_no_such_word_is_an_empty_answer_not_an_error():
    transport, _ = _answer(json_body={"total": 0, "offset": 0, "limit": 10, "words": []})

    assert _search("ぷりんたー", transport) == []


def test_an_error_status_is_an_api_error():
    transport, _ = _answer(status=500, text="Internal Server Error")

    with pytest.raises(JlptVocabApiError, match="status 500"):
        _search("学習", transport)


def test_a_body_that_is_not_json_is_an_api_error():
    transport, _ = _answer(text="<html>maintenance</html>")

    with pytest.raises(JlptVocabApiError, match="not JSON"):
        _search("学習", transport)


def test_json_that_is_not_a_word_list_is_an_api_error():
    transport, _ = _answer(json_body={"error": "something else entirely"})

    with pytest.raises(JlptVocabApiError, match="not a word list"):
        _search("学習", transport)


def test_a_random_word_comes_back_from_the_endpoint_the_app_already_uses():
    transport, seen = _answer(json_body=LIVE_ENTRIES["学習"][0])

    word = asyncio.run(jlpt_vocab_api.random_word(3, transport=transport))

    assert seen[0].url.path == "/api/words/random"
    assert seen[0].url.params["level"] == "3"
    assert word.word == "学習"
    assert word.level == 3


def test_a_random_word_without_a_level_asks_for_any_level():
    transport, seen = _answer(json_body=LIVE_ENTRIES["学"][0])

    asyncio.run(jlpt_vocab_api.random_word(None, transport=transport))

    assert "level" not in seen[0].url.params


def test_an_unreachable_api_is_an_api_error():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(JlptVocabApiError, match="unreachable"):
        _search("学習", httpx.MockTransport(refuse))


def test_the_address_can_be_pointed_elsewhere(monkeypatch):
    monkeypatch.setenv("JLPT_VOCAB_API_URL", "http://127.0.0.1:9/")

    assert jlpt_vocab_api.api_url() == "http://127.0.0.1:9/"
