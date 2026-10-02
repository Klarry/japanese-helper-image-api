"""Day 25: the mini chat - history, task memory and RAG on every question.

The twelve things the assignment asks to prove, and the one it is really
about: after twelve messages of wandering, does the chat still know what the
conversation is for.

Everything runs on a real FAISS index built here with the local embedding
backend and a throwaway history file - no key, no network, nothing written
where the project keeps its own data. The model is stubbed, because what is
under test is the order the pieces run in, not Gemini's prose.
"""

import asyncio
import json
from pathlib import Path

import pytest

from app.schemas.agent import TaskStage
from app.services.agent_history_storage import AgentHistoryStorage
from app.services.agent_memory import LongTermMemory, WorkingMemory
from app.services.agent_memory_router import MemoryRouting
from app.services.agent_task_state import TaskState
from app.services.chat_memory import TaskMemory
from app.services.chat_prompt import chat_prompt
from app.services.chat_session import ChatSession
from app.services.chunking import STRUCTURAL, chunk_documents
from app.services.document_loader import Document
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import GeneratedText
from app.services.rag_enhanced import EnhancedRetriever
from app.services.rag_prompt import DONT_KNOW
from app.services.rag_response import ANSWERED, INSUFFICIENT
from app.services.rag_retriever import RAGRetriever
from app.services.rag_settings import RagSettings
from app.services.vector_index import save_index

REGISTRY = """# MCP registry

## Routing

Every registered server is asked what it offers, and the routing table is
built from the answers rather than written down anywhere. A tool name alone
is then enough to send a call to the server that offers it, and a server that
changes its tools changes the agent's list by itself the next time discovery
runs, with nothing to keep in step by hand.

## Failures

A server that cannot be reached takes only its own tools with it; the others
keep working, and the call comes back as a failed result rather than as an
exception. A tool that no server offers is reported differently from a server
that is down, because a chain that stops has to be able to say which of the
two happened rather than only that something went wrong.
"""

SCHEDULER = """# Scheduler

## Ticking

The digest scheduler ticks once a second, which is cheap because a tick only
asks which tasks are due and goes back to sleep when none of them are. The
shortest interval a task may be created with is ten seconds, and the longest
is a day; both limits are enforced by the schema, so a task that asks for one
second is refused before anything is written to disk rather than quietly
rounded up to something the scheduler can actually manage.

## Restarting

The next run is worked out from the last one rather than from the moment the
scheduler started, so a backend that was down for an hour does one run on the
way back up instead of sixty. Nothing is held in memory between ticks either:
the task file and the run history on disk are the whole state, which is why a
restart continues the count instead of starting it again from zero.
"""

KANJI = """# Kanji practice

## Stroke order

Stroke order is taught from the top left downwards, and the app shows it as
an animation rather than as a list of rules, because the order of the strokes
is easier to copy than to read. The practice screen keeps the same stroke
data for every character, so a kanji looked up in the dictionary and a kanji
met in an exercise are drawn from one source.

## Review queue

The review queue is rebuilt every morning from what was answered wrongly the
day before, and a character that has been answered correctly three times in a
row leaves the queue until the month is out. Nothing about the queue is stored
on the server: it is worked out again from the answer history each time.
"""


def documents() -> list[Document]:
    return [
        Document("docs", "docs/registry.md", "MCP registry", REGISTRY),
        Document("docs", "docs/scheduler.md", "Scheduler", SCHEDULER),
        Document("docs", "docs/kanji.md", "Kanji practice", KANJI),
    ]


@pytest.fixture
def index_dir(tmp_path) -> Path:
    service = EmbeddingService(backend="local")
    chunks = chunk_documents(documents(), STRUCTURAL)
    vectors = asyncio.run(service.embed_all([chunk.text for chunk in chunks]))
    save_index(
        tmp_path / "index" / "structural",
        vectors,
        [chunk.as_metadata() for chunk in chunks],
        STRUCTURAL,
        service.name,
    )

    return tmp_path / "index"


# The local hashing backend scores far below a real embedding model, so the
# thresholds here are set to the band this index produces. What is under test
# is the behaviour at a threshold, not the number.
ANSWERING = 0.15
QUESTION = "What happens when one MCP server cannot be reached?"
QUOTE = "A server that cannot be reached takes only its own tools with it"


def settings(**changes) -> RagSettings:
    base = {
        "retrieval_top_k": 6,
        "similarity_threshold": 0.0,
        "final_top_k": 3,
        "query_rewrite": False,
        "answer_threshold": ANSWERING,
    }
    base.update(changes)

    return RagSettings(**base)


class FakeRouter:
    """The memory router, without a model.

    Each call returns the next prepared layer, or nothing. Prepared rather
    than generated because what is under test is that the chat *asks* and
    stores what comes back - decided by the real router, which has its own
    tests.
    """

    def __init__(self, replies: list[WorkingMemory | None] | None = None) -> None:
        self.replies = list(replies or [])
        self.seen: list[str] = []

    async def route(self, message: str, working: WorkingMemory, long_term: LongTermMemory):
        self.seen.append(message)
        reply = self.replies.pop(0) if self.replies else None

        return MemoryRouting(working=reply, long_term=None, tokens_used=7)


def reply(answer: str, citations: list[dict] | None = None) -> str:
    return json.dumps({"answer": answer, "citations": citations or []})


def stub_model(monkeypatch, *texts: str) -> list[str]:
    """Answer with each text in turn, repeating the last, and record prompts."""
    from app.services import chat_session as module

    prompts: list[str] = []
    answers = list(texts) or [reply("An answer.")]

    async def fake(prompt, model=None, temperature=None):
        prompts.append(prompt)
        text = answers[min(len(prompts) - 1, len(answers) - 1)]

        return GeneratedText(text=text, input_tokens=100, output_tokens=20)

    monkeypatch.setattr(module, "generate_text_with_usage", fake)

    return prompts


def session_for(tmp_path, index_dir, router=None, config=None, **kwargs) -> ChatSession:
    active = config or settings()
    retriever = RAGRetriever(strategy=STRUCTURAL, index_dir=index_dir)

    return ChatSession(
        storage=AgentHistoryStorage(str(tmp_path / "chat_history.json")),
        retriever=retriever,
        settings=active,
        enhanced=EnhancedRetriever(retriever=retriever, settings=active),
        router=router or FakeRouter(),
        **kwargs,
    )


# --- 1. history persistence -------------------------------------------------


def test_both_turns_are_written_with_a_timestamp(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer.", [{"source": 1, "quote": QUOTE}]))
    chat = session_for(tmp_path, index_dir)

    asyncio.run(chat.ask(QUESTION))

    saved = json.loads((tmp_path / "chat_history.json").read_text(encoding="utf-8"))
    messages = saved["branches"]["main"]["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == QUESTION
    assert all(message["timestamp"] for message in messages)


def test_the_assistant_turn_keeps_the_sources_it_answered_with(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer.", [{"source": 1, "quote": QUOTE}]))
    chat = session_for(tmp_path, index_dir)

    turn = asyncio.run(chat.ask(QUESTION))

    stored = chat.messages[-1]["sources"]
    assert stored, "an answer's sources belong with the answer"
    assert stored[0]["chunk_id"] == turn.answer.sources[0].chunk_id
    assert set(stored[0]) == {"source", "file", "section", "chunk_id"}


def test_the_chat_writes_into_the_project_s_own_history_file(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer."))
    chat = session_for(tmp_path, index_dir)

    asyncio.run(chat.ask(QUESTION))

    saved = json.loads((tmp_path / "chat_history.json").read_text(encoding="utf-8"))
    assert "branches" in saved and "working_memory" in saved, "the Day 7 file, not a second one"


# --- 2. history restoration -------------------------------------------------


def test_a_restart_finds_the_conversation_where_it_was_left(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer."))
    first = session_for(tmp_path, index_dir)
    asyncio.run(first.ask(QUESTION))
    asyncio.run(first.ask("And the scheduler?"))

    restarted = session_for(tmp_path, index_dir)

    assert len(restarted.messages) == 4
    assert restarted.messages[0]["content"] == QUESTION


def test_reload_discards_what_is_in_hand_and_reads_the_file(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer."))
    chat = session_for(tmp_path, index_dir)
    asyncio.run(chat.ask(QUESTION))
    chat.messages.append({"role": "user", "content": "never saved"})

    chat.reload()

    assert all(message["content"] != "never saved" for message in chat.messages)


def test_clearing_ends_the_conversation_and_the_task_with_it(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer."))
    router = FakeRouter([WorkingMemory(goals=["ship the feature"], constraints=["N4"])])
    chat = session_for(tmp_path, index_dir, router=router)
    asyncio.run(chat.ask(QUESTION))

    chat.clear()

    assert chat.messages == []
    assert chat.task_memory().is_empty


# --- 3. task memory update --------------------------------------------------


def test_what_a_message_settles_goes_into_task_memory(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer."))
    router = FakeRouter([WorkingMemory(goals=["Japanese learning feature"], constraints=["N4"])])
    chat = session_for(tmp_path, index_dir, router=router)

    turn = asyncio.run(chat.ask("Our task is a Japanese learning feature, level N4."))

    memory = chat.task_memory()
    assert memory.goal == "Japanese learning feature"
    assert memory.constraints == ("N4",)
    assert turn.memory.changed
    assert "+ constraint: N4" in turn.memory.changes


def test_a_message_that_settles_nothing_leaves_the_memory_alone(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer."))
    router = FakeRouter([WorkingMemory(goals=["ship it"]), None])
    chat = session_for(tmp_path, index_dir, router=router)
    asyncio.run(chat.ask("Our task is to ship it."))

    turn = asyncio.run(chat.ask(QUESTION))

    assert not turn.memory.changed
    assert chat.task_memory().goal == "ship it"


def test_a_failed_router_does_not_fail_the_turn(monkeypatch, tmp_path, index_dir):
    class Broken:
        async def route(self, message, working, long_term):
            raise RuntimeError("no model today")

    stub_model(monkeypatch, reply("An answer."))
    chat = session_for(tmp_path, index_dir, router=Broken())

    turn = asyncio.run(chat.ask(QUESTION))

    assert turn.answer.answer == "An answer."
    assert not turn.memory.changed


def test_the_memory_is_updated_from_the_message_not_from_the_answer(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("I have decided the level is N1."))
    router = FakeRouter()
    chat = session_for(tmp_path, index_dir, router=router)

    asyncio.run(chat.ask("Our task is a learning feature."))

    assert router.seen == ["Our task is a learning feature."], "the router never sees the answer"


# --- 4-6. goal, constraints and decisions survive a long conversation -------


def test_the_goal_survives_a_dozen_messages_about_other_things(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer."))
    settled = WorkingMemory(
        goals=["Japanese learning feature"],
        constraints=["N4", "Russian explanations"],
        decisions=["use the existing JLPT API"],
        terms=["学習"],
    )
    router = FakeRouter([settled])
    chat = session_for(tmp_path, index_dir, router=router)
    asyncio.run(chat.ask("Our task is a Japanese learning feature."))

    for _ in range(12):
        asyncio.run(chat.ask(QUESTION))

    memory = chat.task_memory()
    assert memory.goal == "Japanese learning feature"
    assert memory.constraints == ("N4", "Russian explanations")
    assert memory.decisions == ("use the existing JLPT API",)
    assert memory.confirmed_terms == ("学習",)
    assert len(chat.messages) == 26


def test_a_later_goal_does_not_displace_the_first(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer."))
    router = FakeRouter([WorkingMemory(goals=["the first goal", "a later aside"])])
    chat = session_for(tmp_path, index_dir, router=router)

    asyncio.run(chat.ask("Our task is the first goal."))

    assert chat.task_memory().goal == "the first goal"


def test_the_task_memory_is_what_the_model_is_shown(monkeypatch, tmp_path, index_dir):
    prompts = stub_model(monkeypatch, reply("An answer."))
    router = FakeRouter([WorkingMemory(goals=["ship the feature"], constraints=["N4"], terms=["学習"])])
    chat = session_for(tmp_path, index_dir, router=router)
    asyncio.run(chat.ask("Our task is to ship the feature at N4, term 学習."))

    asyncio.run(chat.ask("And what about the scheduler?"))

    assert "TASK MEMORY" in prompts[-1]
    assert "Goal: ship the feature" in prompts[-1]
    assert "Constraint: N4" in prompts[-1]
    assert "Confirmed term: 学習" in prompts[-1]


# --- 7. RAG on every question -----------------------------------------------


def test_every_question_goes_through_retrieval(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer."))
    chat = session_for(tmp_path, index_dir)

    turns = [
        asyncio.run(chat.ask(QUESTION)),
        asyncio.run(chat.ask("What is our current goal?")),
        asyncio.run(chat.ask("Thanks, carry on.")),
    ]

    for turn in turns:
        assert turn.retrieval is not None, "retrieval runs whatever the question looks like"
        assert len(turn.retrieval.retrieval.chunks) > 0


def test_retrieval_runs_even_when_nothing_clears_the_bar(monkeypatch, tmp_path, index_dir):
    prompts = stub_model(monkeypatch, reply("From the task memory."))
    chat = session_for(tmp_path, index_dir, config=settings(answer_threshold=0.99))

    turn = asyncio.run(chat.ask("What is our current goal?"))

    assert turn.retrieval is not None
    assert turn.answer.rag_status == INSUFFICIENT
    assert "none were relevant enough" in prompts[0]


# --- 8-9. sources ------------------------------------------------------------


def test_an_answer_built_on_a_document_names_it(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("Only that server's tools go away.", [{"source": 1, "quote": QUOTE}]))
    chat = session_for(tmp_path, index_dir)

    turn = asyncio.run(chat.ask(QUESTION))

    assert turn.answer.rag_status == ANSWERED
    assert turn.answer.sources
    source = turn.answer.sources[0]
    assert source.file.endswith(".md")
    assert source.chunk_id
    assert turn.answered_from_documents


def test_every_source_is_a_chunk_that_was_really_retrieved(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer.", [{"source": 1, "quote": QUOTE}]))
    chat = session_for(tmp_path, index_dir)

    turn = asyncio.run(chat.ask(QUESTION))
    known = {chunk.chunk_id for chunk in turn.retrieval.chunks}

    for source in turn.answer.sources:
        assert source.chunk_id in known


def test_a_quote_that_is_not_in_its_chunk_leaves_no_source_behind(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("Invented.", [{"source": 1, "quote": "the registry phones each server"}]))
    chat = session_for(tmp_path, index_dir)

    turn = asyncio.run(chat.ask(QUESTION))

    assert turn.answer.citations == []
    assert turn.answer.sources == []
    assert turn.answer.rejected_citations


def test_an_answer_from_memory_alone_reports_no_sources_rather_than_hiding_them(
    monkeypatch, tmp_path, index_dir
):
    stub_model(monkeypatch, reply("Our goal is to ship the feature."))
    chat = session_for(tmp_path, index_dir, config=settings(answer_threshold=0.99))

    turn = asyncio.run(chat.ask("What is our current goal?"))

    assert turn.answer.sources == []
    assert turn.answer.rag_status == INSUFFICIENT
    assert isinstance(turn.answer.as_dict()["sources"], list)


# --- 10. a long conversation -------------------------------------------------


def test_fourteen_messages_keep_their_order_and_their_sources(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer.", [{"source": 1, "quote": QUOTE}]))
    chat = session_for(tmp_path, index_dir)

    for number in range(14):
        asyncio.run(chat.ask(f"Question number {number}: {QUESTION}"))

    messages = chat.messages
    answers = [message for message in messages if message["role"] == "assistant"]
    assert len(messages) == 28
    assert messages[0]["content"].startswith("Question number 0")
    assert messages[-2]["content"].startswith("Question number 13")
    # Every answer carries the key, whatever it found. Some of these turns
    # retrieve a chunk the quote is genuinely not in, and the validator
    # rightly drops the citation - an empty list is the honest report, and a
    # missing key would not be.
    assert all("sources" in message for message in answers)
    assert any(message["sources"] for message in answers)


def test_only_the_recent_window_is_sent_however_long_the_history(monkeypatch, tmp_path, index_dir):
    prompts = stub_model(monkeypatch, reply("An answer."))
    chat = session_for(tmp_path, index_dir, recent_messages=4)

    for number in range(10):
        asyncio.run(chat.ask(f"Question number {number}"))

    last = prompts[-1]
    assert "Question number 9" in last
    assert "Question number 0" not in last, "the whole history is kept, not sent"


# --- 11. context composition -------------------------------------------------


def test_the_prompt_carries_memory_history_documents_and_the_question(monkeypatch, tmp_path, index_dir):
    prompts = stub_model(monkeypatch, reply("An answer."))
    router = FakeRouter([WorkingMemory(goals=["ship the feature"])])
    chat = session_for(tmp_path, index_dir, router=router)
    asyncio.run(chat.ask("Our task is to ship the feature."))

    asyncio.run(chat.ask(QUESTION))

    prompt = prompts[-1]
    assert prompt.index("TASK MEMORY") < prompt.index("RECENT CONVERSATION")
    assert prompt.index("RECENT CONVERSATION") < prompt.index("RETRIEVED DOCUMENTS")
    assert prompt.index("RETRIEVED DOCUMENTS") < prompt.index("CURRENT USER QUESTION")
    assert QUESTION in prompt


def test_the_rules_forbid_inventing_and_name_the_refusal():
    prompt = chat_prompt("why?", [], task_memory="TASK MEMORY:\n- Goal: x")

    assert DONT_KNOW in prompt
    assert "not permission to remember one" in prompt


def test_a_first_question_is_not_padded_with_empty_headings(monkeypatch, tmp_path, index_dir):
    prompts = stub_model(monkeypatch, reply("An answer."))
    chat = session_for(tmp_path, index_dir)

    asyncio.run(chat.ask(QUESTION))

    # The rules mention task memory by name, so the block is what is checked.
    assert "TASK MEMORY (what this conversation" not in prompts[0]
    assert "RECENT CONVERSATION" not in prompts[0]


# --- 12. task memory after a restart ----------------------------------------


def test_the_task_memory_is_still_there_after_a_restart(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer."))
    settled = WorkingMemory(
        goals=["Japanese learning feature"],
        constraints=["N4", "Russian explanations"],
        decisions=["use the existing JLPT API"],
        terms=["学習"],
    )
    first = session_for(tmp_path, index_dir, router=FakeRouter([settled]))
    asyncio.run(first.ask("Our task is a Japanese learning feature."))

    restarted = session_for(tmp_path, index_dir)
    memory = restarted.task_memory()

    assert memory.goal == "Japanese learning feature"
    assert memory.constraints == ("N4", "Russian explanations")
    assert memory.decisions == ("use the existing JLPT API",)
    assert memory.confirmed_terms == ("学習",)


def test_the_restored_memory_is_what_the_next_answer_is_given(monkeypatch, tmp_path, index_dir):
    stub_model(monkeypatch, reply("An answer."))
    first = session_for(
        tmp_path, index_dir, router=FakeRouter([WorkingMemory(goals=["ship it"], constraints=["N4"])])
    )
    asyncio.run(first.ask("Our task is to ship it at N4."))

    prompts = stub_model(monkeypatch, reply("An answer."))
    restarted = session_for(tmp_path, index_dir)
    asyncio.run(restarted.ask("What is our current goal?"))

    assert "Goal: ship it" in prompts[0]
    assert "Constraint: N4" in prompts[0]


# --- the task memory view itself --------------------------------------------


def test_the_view_reads_both_stores_without_owning_either():
    working = WorkingMemory(goals=["g"], constraints=["c"], decisions=["d"], terms=["t"], requirements=["r"])
    memory = TaskMemory.of(working, TaskState(stage=TaskStage.EXECUTION))

    assert memory.as_dict() == {
        "goal": "g",
        "confirmed_terms": ["t"],
        "constraints": ["c"],
        "decisions": ["d"],
        "requirements": ["r"],
        "current_state": "execution",
    }


def test_the_changes_report_additions_and_the_stage():
    before = TaskMemory(goal="g", constraints=("a",))
    after = TaskMemory(goal="g", constraints=("a", "b"), decisions=("d",), current_state="planning")

    assert after.changes_from(before) == [
        "+ constraint: b",
        "+ decision: d",
        "stage: idle -> planning",
    ]


def test_an_empty_memory_renders_nothing_at_all():
    assert TaskMemory().as_section() == ""
    assert TaskMemory().is_empty


# --- the scenarios the evaluation runs --------------------------------------


def test_the_two_scenarios_are_long_and_ask_about_real_documents():
    from app.evaluate_chat import load_scenarios

    scenarios = load_scenarios()

    assert len(scenarios) == 2

    for scenario in scenarios:
        assert 10 <= len(scenario["messages"]) <= 15, scenario["name"]
        kinds = {message["expect"] for message in scenario["messages"]}
        assert {"goal", "constraint", "documents", "memory"} <= kinds, scenario["name"]
        assert scenario["expected_constraints"]


def test_the_closing_questions_are_about_the_task_not_the_documents():
    from app.evaluate_chat import FINAL_QUESTIONS

    assert "goal" in FINAL_QUESTIONS[0].lower()
    assert "constraints" in FINAL_QUESTIONS[1].lower()
    assert "decisions" in FINAL_QUESTIONS[1].lower()
