"""Day 33: the support assistant - MCP case, help-centre RAG, the endpoint.

The CRM call and the model are stood in for in most tests; what is checked is
what the assistant does with them. Marked "real" below: the actual
support-crm MCP server started over stdio, the actual JSON files, and the
help centre indexed the way the indexing command does it.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.routes import support as route
from app.index_documents import index_strategy
from app.main import app
from app.services import gemini_service, mcp_client, ollama_service
from app.services.chunking import STRUCTURAL
from app.services.document_loader import PROJECT_ROOT, SUPPORT_CORPUS, load_documents
from app.services.embedding_service import EmbeddingService
from app.services.llm_provider import OLLAMA, Generated, OllamaProvider, ProviderUnavailable
from app.services.mcp_client import McpConnectionError, McpToolCallResult
from app.services.rag_enhanced import EnhancedRetriever
from app.services.rag_retriever import RAGRetriever, Retrieval, RetrievedChunk
from app.services.support_assistant import (
    ANSWERED,
    INSUFFICIENT,
    SETTINGS,
    SupportAssistant,
    SupportError,
    build_prompt,
    support_profile,
    support_provider,
)
from app.services.vector_index import VectorIndexError
from mcp_servers.support_crm import SupportDataError, find_ticket, find_user, find_user_tickets

client = TestClient(app)
DATA = PROJECT_ROOT / "data" / "support"

TICKET = {
    "ticket_id": "TICKET-1042", "user_id": "USER-501", "status": "open", "category": "authentication",
    "subject": "Не могу войти после сброса пароля",
    "description": "После сброса пароля приложение пишет «Неверный email или пароль». Раньше я входила через Google.",
    "platform": "android", "app_version": "2.3.1",
    "previous_attempts": ["Сбросила пароль по ссылке из письма", "Переустановила приложение"],
}
USER = {"user_id": "USER-501", "language": "ru", "subscription": "premium"}
GOOGLE = "Если раньше вы входили кнопкой «Войти через Google», сброс пароля на вход не влияет."
DOC = f"## «Неверный email или пароль» после сброса пароля (authentication)\n\n3. {GOOGLE} Нажмите «Войти через Google» на экране входа."


def chunk(text=DOC, score=0.45, chunk_id="docs-support-troubleshooting_1"):
    return RetrievedChunk(chunk_id=chunk_id, text=text, source="support", file="docs/support/troubleshooting.md",
                          title="troubleshooting.md", section="Japanese Helper — решение проблем > «Неверный email или пароль»",
                          strategy="structural", position=1, score=score)


class Index:
    """The vector search, stood in for; the filter and reranker are the real ones."""

    def __init__(self, chunks=(chunk(),), error=None):
        self.chunks, self.error, self.queries = list(chunks), error, []

    async def retrieve(self, query, top_k=10):
        self.queries.append(query)
        if self.error:
            raise self.error
        return Retrieval(query=query, chunks=tuple(self.chunks), top_k=top_k, embedding_model="local-hashing",
                         strategy="structural", seconds=0.01)


class CRM:
    """The MCP call, stood in for: records what was asked, answers from dicts."""

    def __init__(self, tickets=(TICKET,), users=(USER,), down=False):
        self.tickets = {item["ticket_id"]: item for item in tickets}
        self.users = {item["user_id"]: item for item in users}
        self.down, self.calls = down, []

    async def __call__(self, name, arguments, server):
        self.calls.append((name, arguments))
        if self.down:
            raise McpConnectionError("server did not start")
        table, key, kind = (self.tickets, "ticket_id", "ticket_not_found") if name == "get_ticket" else (self.users, "user_id", "user_not_found")
        record = table.get(arguments[key])
        if record is None:
            return McpToolCallResult(name=name, arguments=arguments, ok=False, error=f"Error executing tool {name}: {kind}: there is no {arguments[key]}")
        return McpToolCallResult(name=name, arguments=arguments, ok=True, data=dict(record))


class Model:
    name = OLLAMA

    def __init__(self, reply):
        self.model, self.profile, self.reply, self.prompts = "qwen3:4b-instruct-2507-q4_K_M", support_profile(), reply, []

    async def generate(self, prompt, temperature=None):
        self.prompts.append(prompt)
        return Generated(text=self.reply, model=self.model, input_tokens=10, output_tokens=20)


def reply(answer="Судя по обращению, вы входили через Google.", quotes=(GOOGLE,), facts=("description", "previous_attempts")):
    return json.dumps({"answer": answer, "citations": [{"source": 1, "quote": quote} for quote in quotes],
                       "ticket_facts_used": list(facts)}, ensure_ascii=False)


def assistant(model=None, crm=None, index=None):
    model = model or Model(reply())
    crm = crm or CRM()
    retriever = EnhancedRetriever(retriever=index or Index(), settings=SETTINGS)
    return SupportAssistant(provider=model, retriever=retriever, crm_call=crm), model, crm


def ask(message="Почему не работает авторизация?", ticket="TICKET-1042", **parts):
    helper, model, crm = assistant(**parts)
    return asyncio.run(helper.ask(message, ticket)), model, crm


def no_cloud(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the support assistant called Gemini text generation")

    for name in ("_generate", "_post_to_gemini", "generate_text", "generate_text_with_usage"):
        monkeypatch.setattr(gemini_service, name, forbidden)


# --- 1. a successful answer: ticket context and documentation ----------------------------------


def test_an_answer_uses_the_ticket_and_the_documentation(monkeypatch):
    no_cloud(monkeypatch)
    answer, model, crm = ask()

    assert answer.status == ANSWERED
    assert [call[0] for call in crm.calls] == ["get_ticket", "get_user"]
    assert crm.calls[1][1] == {"user_id": "USER-501"}  # the ticket's own user, never the caller's
    assert answer.citations[0]["quote"] == GOOGLE and answer.sources[0]["file"] == "docs/support/troubleshooting.md"
    assert answer.ticket["ticket_id"] == "TICKET-1042" and answer.ticket["user_language"] == "ru"
    assert "user_id" not in answer.ticket
    assert answer.ticket_facts_used == ["description", "previous_attempts"]
    assert [tool.name for tool in answer.tools_used] == ["get_ticket", "get_user"]
    assert set(answer.timing) == {"crm_ms", "retrieval_ms", "generation_ms", "total_ms"}

    prompt = model.prompts[0]
    assert "Answer in Russian" in prompt
    assert "subject: Не могу войти после сброса пароля" in prompt
    assert "[1] file: docs/support/troubleshooting.md" in prompt
    assert prompt.index("<question>") > prompt.index("<documentation>") > prompt.index("<ticket>")


def test_the_search_uses_the_question_and_the_tickets_own_words():
    index = Index()
    ask(index=index)

    assert "Почему не работает авторизация?" in index.queries[0]
    assert "Не могу войти после сброса пароля" in index.queries[0] and "authentication" in index.queries[0]


def test_the_answer_language_follows_the_user():
    english = {**USER, "user_id": "USER-502", "language": "en"}
    ticket = {**TICKET, "ticket_id": "TICKET-1043", "user_id": "USER-502"}
    answer, model, _ = ask(ticket="TICKET-1043", crm=CRM(tickets=(ticket,), users=(english,)))

    assert "Answer in English" in model.prompts[0]


# --- 2-3. a missing ticket, a ticket without its user ----------------------------------------


def test_a_missing_ticket_is_404_and_no_model_is_asked():
    model = Model(reply())

    with pytest.raises(SupportError) as raised:
        ask(ticket="TICKET-9999", model=model)

    assert (raised.value.kind, raised.value.status) == ("ticket_not_found", 404)
    assert raised.value.message == "there is no TICKET-9999"
    assert model.prompts == []


def test_a_ticket_whose_user_does_not_exist_is_refused_not_guessed():
    orphan = {**TICKET, "ticket_id": "TICKET-1047", "user_id": "USER-599"}
    model = Model(reply())

    with pytest.raises(SupportError) as raised:
        ask(ticket="TICKET-1047", crm=CRM(tickets=(orphan,)), model=model)

    assert (raised.value.kind, raised.value.status) == ("ticket_user_not_found", 409)
    assert model.prompts == []


# --- 4. previous attempts --------------------------------------------------------------------


def test_previous_attempts_reach_the_prompt_with_the_rule_about_them():
    _, model, _ = ask()
    prompt = model.prompts[0]

    assert "previous_attempts: Сбросила пароль по ссылке из письма; Переустановила приложение" in prompt
    assert "Do not suggest steps listed in previous_attempts as if they were new" in prompt


def test_no_attempts_is_said_not_left_blank():
    fresh = {**TICKET, "previous_attempts": []}
    _, model, _ = ask(crm=CRM(tickets=(fresh,)))

    assert "previous_attempts: none recorded" in model.prompts[0]


# --- 5. the help centre, indexed and searched for real -------------------------------------------


@pytest.fixture(scope="module")
def help_centre(tmp_path_factory):
    """docs/support indexed exactly as the indexing command does it."""
    directory = tmp_path_factory.mktemp("support-index")
    embeddings = EmbeddingService(backend="local")
    asyncio.run(index_strategy(load_documents(SUPPORT_CORPUS), STRUCTURAL, embeddings, directory, 1200, 150))
    return EnhancedRetriever(retriever=RAGRetriever(strategy=STRUCTURAL, index_dir=directory, embeddings=embeddings), settings=SETTINGS)


def test_real_the_help_centre_answers_a_sign_in_question(help_centre):
    found = asyncio.run(help_centre.retrieve(
        "Почему не работает авторизация?\nНе могу войти после сброса пароля\nПосле сброса пароля приложение пишет «Неверный email или пароль».\nauthentication",
        settings=SETTINGS,
    ))

    files = {item.chunk.chunk_id for item in found.final}
    assert "docs-support-troubleshooting_1" in files  # «Неверный email или пароль» после сброса пароля
    assert all(item.chunk.source == "support" for item in found.final)


def test_real_an_off_topic_question_finds_nothing_in_the_help_centre(help_centre):
    found = asyncio.run(help_centre.retrieve("Можно ли оплатить подписку криптовалютой?", settings=SETTINGS))

    assert found.nothing_relevant


def test_real_the_support_corpus_is_the_help_centre_only():
    files = {document.file for document in load_documents(SUPPORT_CORPUS)}

    assert files == {"docs/support/faq.md", "docs/support/troubleshooting.md", "docs/support/product_guide.md"}


# --- 6. not enough documentation -----------------------------------------------------------------


def test_without_documentation_the_answer_is_marked_and_cites_nothing():
    model = Model(json.dumps({"answer": "В документации этого нет. Уточните, пожалуйста…", "citations": []}, ensure_ascii=False))
    answer, model, _ = ask(index=Index(chunks=()), model=model)

    assert answer.status == INSUFFICIENT
    assert answer.sources == [] and answer.citations == []
    assert "(nothing in the help centre matched this question)" in model.prompts[0]
    assert "say plainly that the documentation does not describe it" in model.prompts[0]


def test_a_missing_help_centre_index_is_503_with_the_command():
    with pytest.raises(SupportError) as raised:
        ask(index=Index(error=VectorIndexError("no index")))

    assert raised.value.status == 503 and "--corpus support" in raised.value.message


# --- 7. malformed support data ----------------------------------------------------------------


def _data(tmp_path, tickets, users='{"users": []}'):
    (tmp_path / "tickets.json").write_text(tickets, encoding="utf-8")
    (tmp_path / "users.json").write_text(users, encoding="utf-8")
    return tmp_path


def test_broken_json_is_unavailable_data_not_an_empty_answer(tmp_path):
    with pytest.raises(SupportDataError) as raised:
        find_ticket("TICKET-1", _data(tmp_path, "{not json"))

    assert raised.value.kind == "support_data_unavailable"


def test_a_record_without_its_fields_is_malformed_and_others_still_read(tmp_path):
    data = _data(tmp_path, json.dumps({"tickets": [{"ticket_id": "TICKET-1"}, {**TICKET, "ticket_id": "TICKET-2"}, "junk"]}))

    with pytest.raises(SupportDataError) as raised:
        find_ticket("TICKET-1", data)

    assert raised.value.kind == "malformed_record"
    assert find_ticket("ticket-2", data).ticket_id == "TICKET-2"


def test_a_missing_file_and_a_bad_id_are_named(tmp_path):
    with pytest.raises(SupportDataError) as missing:
        find_user("USER-1", tmp_path)

    with pytest.raises(SupportDataError) as invalid:
        find_ticket("../../etc/passwd", tmp_path)

    assert (missing.value.kind, invalid.value.kind) == ("support_data_unavailable", "invalid_ticket_id")


def test_a_user_record_returns_only_support_fields(tmp_path):
    users = json.dumps({"users": [{**USER, "email": "anna@example.com", "password_hash": "x"}]})

    assert find_user("USER-501", _data(tmp_path, '{"tickets": []}', users)).model_dump() == USER


# --- 8. MCP failures --------------------------------------------------------------------------


def test_an_unreachable_crm_is_503():
    with pytest.raises(SupportError) as raised:
        ask(crm=CRM(down=True))

    assert (raised.value.kind, raised.value.status) == ("crm_unavailable", 503)


def test_an_unexpected_tool_error_is_502_not_a_guess():
    class Broken(CRM):
        async def __call__(self, name, arguments, server):
            return McpToolCallResult(name=name, arguments=arguments, ok=False, error="something odd")

    with pytest.raises(SupportError) as raised:
        ask(crm=Broken())

    assert (raised.value.kind, raised.value.status) == ("crm_error", 502)


# --- 9-10. the local model, and no cloud ------------------------------------------------------------


def test_ollama_down_is_provider_unavailable_and_never_gemini(monkeypatch):
    no_cloud(monkeypatch)

    async def down(prompt, model="", options=None):
        raise ollama_service.OllamaUnavailable("not_running", "Ollama is not running")

    monkeypatch.setattr(ollama_service, "generate", down)
    helper, _, _ = assistant(model=OllamaProvider(profile=support_profile()))

    with pytest.raises(ProviderUnavailable) as raised:
        asyncio.run(helper.ask("Почему не работает авторизация?", "TICKET-1042"))

    assert raised.value.provider == "ollama"


def test_the_local_provider_is_ollama_with_json_and_no_cloud_call(monkeypatch):
    no_cloud(monkeypatch)
    sent = {}

    async def local(prompt, model="", options=None):
        sent["fields"] = options.request_fields()
        return ollama_service.LocalAnswer(response=reply(), model="qwen3:4b-instruct-2507-q4_K_M", seconds=1.0)

    monkeypatch.setattr(ollama_service, "generate", local)
    assert isinstance(support_provider(OLLAMA), OllamaProvider)
    helper, _, _ = assistant(model=support_provider(OLLAMA))
    answer = asyncio.run(helper.ask("Почему не работает авторизация?", "TICKET-1042"))

    assert answer.provider == "ollama" and sent["fields"]["format"] == "json" and sent["fields"]["think"] is False


# --- 11. citations ------------------------------------------------------------------------------


def test_invented_quotes_and_ticket_text_are_not_documentation():
    model = Model(reply(quotes=(GOOGLE, "Сбросьте кэш приложения в настройках телефона.", TICKET["subject"])))
    answer, _, _ = ask(model=model)

    assert [citation["quote"] for citation in answer.citations] == [GOOGLE]
    assert answer.rejected_citations == 2
    assert [source["chunk_id"] for source in answer.sources] == ["docs-support-troubleshooting_1"]


def test_only_real_ticket_fields_are_reported_as_used():
    answer, _, _ = ask(model=Model(reply(facts=("previous_attempts", "password", "user_id"))))

    assert answer.ticket_facts_used == ["previous_attempts"]


# --- 12. one user's case does not leak another's --------------------------------------------------


def test_the_prompt_holds_only_this_tickets_user(monkeypatch):
    other_ticket = {**TICKET, "ticket_id": "TICKET-1043", "user_id": "USER-502", "subject": "Pictures are not generated"}
    other_user = {"user_id": "USER-502", "language": "en", "subscription": "free"}
    answer, model, crm = ask(crm=CRM(tickets=(TICKET, other_ticket), users=(USER, other_user)))

    assert "Pictures are not generated" not in model.prompts[0] and "USER-502" not in model.prompts[0]
    assert all(arguments.get("user_id") != "USER-502" for _, arguments in crm.calls)


def test_real_a_users_tickets_are_only_theirs():
    tickets = find_user_tickets("USER-501", DATA).tickets

    assert {item.ticket_id for item in tickets} == {"TICKET-1042", "TICKET-1046"}


def test_the_request_cannot_name_a_user():
    response = client.post("/support/chat", json={"message": "hi", "ticket_id": "TICKET-1042", "user_id": "USER-502"})

    assert response.status_code == 422


# --- the real MCP server ------------------------------------------------------------------------


def test_real_mcp_server_reads_the_ticket_and_says_not_found(monkeypatch):
    monkeypatch.setenv("SUPPORT_DATA_DIR", str(DATA))
    parameters = mcp_client.support_crm_server_parameters()

    async def calls():
        found = await mcp_client.call_server_tool("get_ticket", {"ticket_id": "TICKET-1042"}, parameters)
        missing = await mcp_client.call_server_tool("get_ticket", {"ticket_id": "TICKET-9999"}, parameters)
        tools = await mcp_client.list_server_tools(parameters)
        return found, missing, tools

    found, missing, tools = asyncio.run(calls())

    assert found.ok and found.data["user_id"] == "USER-501" and found.data["previous_attempts"]
    assert not missing.ok and "ticket_not_found" in missing.error
    assert {tool.name for tool in tools.tools} == {"get_ticket", "get_user", "get_user_tickets"}


def test_the_crm_is_registered_with_the_client_not_the_agent():
    from app.services import mcp_registry

    assert "support-crm" in mcp_client.SERVERS
    assert "support-crm" not in str(getattr(mcp_registry, "REGISTERED_SERVERS", ""))


# --- the endpoint -------------------------------------------------------------------------------


@pytest.fixture
def endpoint(monkeypatch):
    model = Model(reply())
    monkeypatch.setattr(route, "assistant_for", lambda name: assistant(model=model)[0])
    return model


def test_the_endpoint_answers_with_sources_and_the_ticket(endpoint):
    response = client.post("/support/chat", json={"message": "Почему не работает авторизация?", "ticket_id": "ticket-1042", "provider": "ollama"})

    assert response.status_code == 200
    data = response.json()
    assert data["ticket_id"] == "TICKET-1042" and data["status"] == "answered"
    assert data["citations"][0]["chunk_id"] == "docs-support-troubleshooting_1"
    assert [tool["name"] for tool in data["tools_used"]] == ["get_ticket", "get_user"]
    assert data["ticket"]["previous_attempts"] and data["request_id"] == response.headers["X-Request-ID"]


def test_the_endpoint_says_404_for_a_missing_ticket(endpoint):
    response = client.post("/support/chat", json={"message": "Почему?", "ticket_id": "TICKET-9999"})

    assert response.status_code == 404
    assert response.json()["error"] == "ticket_not_found" and response.json()["ticket_id"] == "TICKET-9999"
    assert endpoint.prompts == []


@pytest.mark.parametrize(
    "body",
    [
        {"message": "", "ticket_id": "TICKET-1042"},
        {"message": "   ", "ticket_id": "TICKET-1042"},
        {"message": "hi", "ticket_id": "../tickets.json"},
        {"message": "hi", "ticket_id": "TICKET-1042", "provider": "openai"},
        {"message": "hi"},
    ],
)
def test_the_endpoint_validates_the_request(body):
    assert client.post("/support/chat", json=body).status_code == 422


def test_real_mcp_and_index_with_a_stand_in_model(monkeypatch, help_centre):
    """The whole pipeline but the model: the real MCP server over stdio, the
    real JSON files, the help centre indexed for real."""
    monkeypatch.setenv("SUPPORT_DATA_DIR", str(DATA))
    no_cloud(monkeypatch)
    quote = "Если раньше вы входили кнопкой «Войти через Google», сброс пароля на вход не влияет."
    model = Model(json.dumps({"answer": "…", "citations": [{"source": 1, "quote": quote}]}, ensure_ascii=False))
    helper = SupportAssistant(provider=model, retriever=help_centre)
    answer = asyncio.run(helper.ask("Почему не работает авторизация?", "TICKET-1042"))

    assert [(tool.name, tool.ok) for tool in answer.tools_used] == [("get_ticket", True), ("get_user", True)]
    assert answer.ticket["previous_attempts"] == ["Сбросила пароль по ссылке из письма", "Переустановила приложение"]
    assert answer.retrieval["chunks"][0]["chunk_id"] == "docs-support-troubleshooting_1"
    assert answer.status == ANSWERED and answer.sources[0]["chunk_id"] == "docs-support-troubleshooting_1"
