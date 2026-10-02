"""Day 25: the mini chat, assembled from what the project already has.

Nothing here is new machinery. The conversation is stored by the Day 7
history storage, in the same file and the same branch the agent uses. The
task's memory is the Day 11 working memory and the Day 13 task state, read
through one view. Retrieval is the Day 23 second stage, and the citations
are checked by the Day 24 validator. What this file adds is the order they
run in, and the two rules that make a chat out of them:

**Retrieval happens on every question.** Not when the question looks like a
documentation question - on every one. A chat where the model decides when to
look things up is a chat where "what did we decide about the API?" quietly
becomes a guess; and a question that finds nothing still reports that it
found nothing, which is information.

**The memory is updated from the message, not from the answer.** The router
reads what the person said and returns the layers it changed. An assistant
that could write its own constraints into the task's memory would be free to
agree with itself later, so it cannot: the only way into task memory is
something the person actually said.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.schemas.agent import ContextStrategy
from app.services.agent_context import build_context
from app.services.agent_history_storage import (
    AgentHistoryStorage,
    ConversationState,
    HistoryMessage,
)
from app.services.agent_memory import AgentLongTermMemoryStorage
from app.services.agent_memory_router import MemoryRouter
from app.services.chat_memory import MemoryUpdate, TaskMemory
from app.services.chat_prompt import chat_prompt
from app.services.gemini_service import generate_text_with_usage
from app.services.rag_citations import Source, validate
from app.services.rag_enhanced import EnhancedRetrieval, EnhancedRetriever
from app.services.rag_response import ANSWERED, INSUFFICIENT, CitedAnswer, confidence_of
from app.services.rag_retriever import RAGRetriever
from app.services.rag_settings import DEFAULT_SETTINGS, RagSettings

logger = logging.getLogger(__name__)

#: How many of the newest messages go into the prompt. The whole history is
#: kept on disk; only this much is sent, which is what the Day 8 strategies
#: exist for - and the sliding window is the one that needs no summariser.
RECENT_MESSAGES = 8


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ChatTurn:
    """One question and everything that went into answering it."""

    question: str
    answer: CitedAnswer
    retrieval: EnhancedRetrieval | None = None
    memory: MemoryUpdate | None = None
    history_length: int = 0
    seconds: float = 0.0
    tokens_used: int = 0
    prompt: str = ""
    sent_messages: list[HistoryMessage] = field(default_factory=list)

    @property
    def sources(self) -> list[Source]:
        return self.answer.sources

    @property
    def answered_from_documents(self) -> bool:
        return bool(self.answer.citations)

    def as_dict(self) -> dict:
        record = {
            "question": self.question,
            **self.answer.as_dict(),
            "history_length": self.history_length,
            "seconds": round(self.seconds, 3),
            "tokens_used": self.tokens_used,
        }

        if self.retrieval is not None:
            record["retrieval"] = self.retrieval.as_dict()

        if self.memory is not None:
            record["memory"] = self.memory.as_dict()

        return record


class ChatSession:
    """One conversation: history on disk, a task memory, and RAG per question."""

    def __init__(
        self,
        storage: AgentHistoryStorage | None = None,
        retriever: RAGRetriever | None = None,
        settings: RagSettings = DEFAULT_SETTINGS,
        enhanced: EnhancedRetriever | None = None,
        router: MemoryRouter | None = None,
        recent_messages: int = RECENT_MESSAGES,
        update_memory: bool = True,
        long_term: AgentLongTermMemoryStorage | None = None,
    ) -> None:
        self._storage = storage or AgentHistoryStorage()
        self._retriever = retriever or RAGRetriever()
        self._settings = settings
        self._enhanced = enhanced or EnhancedRetriever(retriever=self._retriever, settings=settings)
        self._router = router or MemoryRouter()
        # The router is shown long-term memory so it does not write a lasting
        # fact into the task's memory by mistake. The chat never writes to
        # that file - it belongs to the learner, not to this conversation.
        self._long_term = long_term or AgentLongTermMemoryStorage()
        self._recent = recent_messages
        self._update_memory = update_memory
        # Loaded once and written after every turn, so a restart picks the
        # conversation up where it was left - history, memory and stage.
        self._state: ConversationState = self._storage.load()

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def messages(self) -> list[HistoryMessage]:
        return self._state.current().messages

    def task_memory(self) -> TaskMemory:
        return TaskMemory.of(self._state.working_memory, self._state.task_state)

    def reload(self) -> "ChatSession":
        """Read the conversation back from disk, discarding what is in hand.
        What a restart does, without restarting."""
        self._state = self._storage.load()

        return self

    def clear(self) -> None:
        """End the conversation and the task with it. The task memory is
        scoped to the conversation by design (Day 11), so one cannot be kept
        without the other."""
        self._state = ConversationState()
        self._storage.save(self._state)

    async def ask(self, question: str) -> ChatTurn:
        """One turn: remember, retrieve, answer, record."""
        text = question.strip()

        if not text:
            raise ValueError("there is no question to ask")

        started = time.perf_counter()
        before = self.task_memory()
        update = await self._remember(text, before)
        found = await self._enhanced.retrieve(text, settings=self._settings)
        sent = self._recent_messages()
        answer = await self._answer(text, found, sent)
        cited = answer.cited
        self._record(text, cited)
        seconds = time.perf_counter() - started

        logger.info(
            "[Chat] turn done · status=%s · sources=%d · citations=%d · history=%d · memory %s",
            cited.rag_status,
            len(cited.sources),
            len(cited.citations),
            len(self.messages),
            "changed" if update.changed else "unchanged",
        )

        return ChatTurn(
            question=text,
            answer=cited,
            retrieval=found,
            memory=update,
            history_length=len(self.messages),
            seconds=seconds,
            tokens_used=answer.tokens_used + update.tokens_used,
            prompt=answer.prompt,
            sent_messages=sent,
        )

    async def _remember(self, text: str, before: TaskMemory) -> MemoryUpdate:
        """What this message settled, through the router the agent uses.

        A failed routing is not a failed turn: the question still gets
        answered, the memory simply does not grow this time.
        """
        if not self._update_memory:
            return MemoryUpdate(before=before, after=before)

        try:
            routing = await self._router.route(
                text, self._state.working_memory, self._long_term.load()
            )
        except Exception as error:  # noqa: BLE001 - the answer stands either way
            logger.warning("[Chat] memory routing failed (%s); memory left as it was", error)

            return MemoryUpdate(before=before, after=before)

        if routing.working is not None:
            self._state.working_memory = routing.working

        after = self.task_memory()

        return MemoryUpdate(
            before=before,
            after=after,
            changes=after.changes_from(before),
            tokens_used=routing.tokens_used,
        )

    def _recent_messages(self) -> list[HistoryMessage]:
        """The newest messages, through the project's own context builder.

        The sliding window rather than the whole history on purpose: the task
        memory already carries what matters from earlier, so sending every
        turn would be paying twice for the same facts.
        """
        return build_context(
            ContextStrategy.SLIDING_WINDOW,
            self._state.current(),
            recent_messages_kept=self._recent,
        ).messages

    async def _answer(self, text: str, found: EnhancedRetrieval, sent: list[HistoryMessage]):
        """Ask the model, then check every citation it offered."""
        relevant = (
            list(found.chunks)
            if found.best_relevance >= self._settings.answer_threshold
            else []
        )

        if not relevant:
            logger.info(
                "[Relevance] Best score: %.3f · Threshold: %.2f · Status: insufficient_context",
                found.best_relevance,
                self._settings.answer_threshold,
            )

        prompt = chat_prompt(
            text,
            relevant,
            task_memory=self.task_memory().as_section(),
            recent_history=_transcript(sent),
        )
        generated = await generate_text_with_usage(prompt)
        checked = validate(generated.text, relevant)
        cited = CitedAnswer(
            answer=checked.answer,
            # A turn that answered from task memory alone is still an answer -
            # it just has nothing to cite. The status says which happened.
            rag_status=ANSWERED if relevant else INSUFFICIENT,
            confidence=confidence_of(
                ANSWERED if relevant else INSUFFICIENT,
                found.best_relevance,
                self._settings.answer_threshold,
                checked.citations,
            ),
            sources=list(checked.sources),
            citations=list(checked.citations),
            rejected_citations=list(checked.rejected),
            best_relevance=found.best_relevance,
            answer_threshold=self._settings.answer_threshold,
        )

        return _Answered(
            cited=cited,
            prompt=prompt,
            tokens_used=(generated.input_tokens or 0) + (generated.output_tokens or 0),
        )

    def _record(self, question: str, cited: CitedAnswer) -> None:
        """Both turns into the conversation, and the conversation to disk.

        The assistant's sources are stored with its message rather than
        recomputed later: what an answer was built on is a fact about that
        answer, and the index it came from may be rebuilt tomorrow.
        """
        history = self._state.current()
        history.messages.append({"role": "user", "content": question, "timestamp": now()})
        history.messages.append(
            {
                "role": "assistant",
                "content": cited.answer,
                "timestamp": now(),
                "sources": [source.as_dict() for source in cited.sources],
            }
        )
        self._storage.save(self._state)


@dataclass
class _Answered:
    cited: CitedAnswer
    prompt: str
    tokens_used: int


def _transcript(messages: list[HistoryMessage]) -> str:
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)
