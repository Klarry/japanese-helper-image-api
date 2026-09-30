"""Days 22-23: one agent, three modes.

Day 22 put two modes behind one method so the answers could be compared:
the mode decided whether there was a retrieval step between the question and
the model, and nothing else changed. Day 23 adds a third for the same
reason - a second retrieval stage is only worth having if it can be measured
against the first one, on the same question, through the same agent.

    OFF       question -> model
    BASELINE  question -> FAISS top-k -> model
    ENHANCED  question -> rewrite -> FAISS -> filter -> rerank -> top-k -> model

What stays true in all three: the model is always asked the question the
person asked. The rewrite exists to find documents, never to replace the
question, and the answer comes back with what was actually retrieved -
nothing is reported that was not used.
"""

import logging
import time
from dataclasses import dataclass, field

from app.services.gemini_service import generate_text_with_usage
from app.services.rag_enhanced import EnhancedRetrieval, EnhancedRetriever
from app.services.rag_prompt import NO_CONTEXT_ANSWER, plain_prompt, rag_prompt
from app.services.rag_retriever import DEFAULT_TOP_K, RAGRetriever, Retrieval, RetrievedChunk
from app.services.rag_settings import DEFAULT_SETTINGS, RagSettings

logger = logging.getLogger(__name__)

OFF = "off"
BASELINE = "baseline"
ENHANCED = "enhanced"
MODES = (OFF, BASELINE, ENHANCED)


def mode_of(use_rag: bool | None, mode: str | None) -> str:
    """Which mode a call meant.

    ``mode`` wins when it is given. Otherwise the Day 22 flag decides, so
    every caller written before this existed keeps the behaviour it had.
    """
    if mode is not None:
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode!r} (expected one of {', '.join(MODES)})")

        return mode

    return BASELINE if use_rag else OFF


@dataclass
class RagAnswer:
    """One answer, and everything that went into it."""

    question: str
    answer: str
    rag_enabled: bool
    mode: str = BASELINE
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    retrieval_seconds: float = 0.0
    llm_seconds: float = 0.0
    tokens_used: int = 0
    top_k: int = 0
    embedding_model: str = ""
    prompt: str = ""
    #: Only for ENHANCED: what the second stage did. None in the other modes,
    #: because there was no second stage to report.
    enhanced: EnhancedRetrieval | None = None

    @property
    def total_seconds(self) -> float:
        return self.retrieval_seconds + self.llm_seconds

    @property
    def answered_from_documents(self) -> bool:
        return bool(self.retrieved_chunks)

    def as_dict(self, with_text: bool = False, with_prompt: bool = False) -> dict:
        record = {
            "question": self.question,
            "answer": self.answer,
            "mode": self.mode,
            "rag_enabled": self.rag_enabled,
            "retrieved_chunks": [chunk.as_dict(with_text) for chunk in self.retrieved_chunks],
            "sources": self.sources,
            "top_k": self.top_k,
            "embedding_model": self.embedding_model,
            "retrieval_seconds": round(self.retrieval_seconds, 3),
            "llm_seconds": round(self.llm_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "tokens_used": self.tokens_used,
        }

        if self.enhanced is not None:
            record["enhanced"] = self.enhanced.as_dict(with_text)

        if with_prompt:
            record["prompt"] = self.prompt

        return record


class RagAgent:
    """Answers a question about the project, in any of the three modes."""

    def __init__(
        self,
        retriever: RAGRetriever | None = None,
        top_k: int = DEFAULT_TOP_K,
        settings: RagSettings = DEFAULT_SETTINGS,
        enhanced: EnhancedRetriever | None = None,
    ) -> None:
        self._retriever = retriever or RAGRetriever()
        self._top_k = top_k
        self._settings = settings
        self._enhanced = enhanced or EnhancedRetriever(retriever=self._retriever, settings=settings)

    @property
    def retriever(self) -> RAGRetriever:
        return self._retriever

    @property
    def enhanced_retriever(self) -> EnhancedRetriever:
        return self._enhanced

    @property
    def settings(self) -> RagSettings:
        return self._settings

    async def ask(
        self,
        question: str,
        use_rag: bool = True,
        top_k: int | None = None,
        mode: str | None = None,
        settings: RagSettings | None = None,
    ) -> RagAnswer:
        """The same question, answered in whichever mode was asked for."""
        text = question.strip()

        if not text:
            raise ValueError("there is no question to answer")

        chosen = mode_of(use_rag, mode)

        if chosen == ENHANCED:
            return await self._ask_enhanced(text, settings)

        return await self._ask_plain(text, chosen, top_k)

    async def _ask_plain(self, text: str, mode: str, top_k: int | None) -> RagAnswer:
        """OFF and BASELINE: Day 22, unchanged."""
        retrieval: Retrieval | None = None
        prompt = plain_prompt(text)

        if mode == BASELINE:
            retrieval = await self._retriever.retrieve(text, top_k=top_k or self._top_k)
            prompt = rag_prompt(text, retrieval.chunks)

        logger.info("Asking the model (mode=%s): %r", mode, text[:60])
        generated, llm_seconds = await self._generate(prompt)

        return RagAnswer(
            question=text,
            answer=generated.text.strip(),
            rag_enabled=mode != OFF,
            mode=mode,
            retrieved_chunks=list(retrieval.chunks) if retrieval else [],
            sources=retrieval.sources if retrieval else [],
            retrieval_seconds=retrieval.seconds if retrieval else 0.0,
            llm_seconds=llm_seconds,
            tokens_used=(generated.input_tokens or 0) + (generated.output_tokens or 0),
            top_k=retrieval.top_k if retrieval else 0,
            embedding_model=retrieval.embedding_model if retrieval else "",
            prompt=prompt,
        )

    async def _ask_enhanced(self, text: str, settings: RagSettings | None) -> RagAnswer:
        """ENHANCED: the second stage, and the one case where the model is
        not asked at all."""
        found = await self._enhanced.retrieve(text, settings=settings)
        active = found.settings

        if found.nothing_relevant:
            # Nothing cleared the threshold. Sending the closest chunks
            # anyway is exactly how an answer gets invented, so the model is
            # not asked and the honest answer is returned as it stands.
            logger.info(
                "Nothing above %.2f for %r - answering without the model",
                active.similarity_threshold,
                text[:60],
            )

            return RagAnswer(
                question=text,
                answer=NO_CONTEXT_ANSWER,
                rag_enabled=True,
                mode=ENHANCED,
                retrieval_seconds=found.seconds,
                top_k=active.final_top_k,
                embedding_model=found.retrieval.embedding_model,
                prompt="",
                enhanced=found,
            )

        prompt = rag_prompt(text, found.chunks)
        logger.info("Asking the model (mode=%s): %r", ENHANCED, text[:60])
        generated, llm_seconds = await self._generate(prompt)

        return RagAnswer(
            question=text,
            answer=generated.text.strip(),
            rag_enabled=True,
            mode=ENHANCED,
            retrieved_chunks=found.chunks,
            sources=found.sources,
            retrieval_seconds=found.seconds,
            llm_seconds=llm_seconds,
            tokens_used=(generated.input_tokens or 0) + (generated.output_tokens or 0),
            top_k=active.final_top_k,
            embedding_model=found.retrieval.embedding_model,
            prompt=prompt,
            enhanced=found,
        )

    async def _generate(self, prompt: str):
        started = time.perf_counter()
        generated = await generate_text_with_usage(prompt)

        return generated, time.perf_counter() - started
