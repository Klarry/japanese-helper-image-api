"""Days 22-24: one agent, three modes, one shape of answer.

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

Day 24 adds the part that makes "with sources" mean something. ENHANCED now
asks the model for exact quotes, checks every one of them against the chunk
it claims to come from, and throws away the ones that do not hold. And
before any of that, it asks whether the evidence is good enough to answer
from at all: below the answering threshold the model is never called, and
the reply says so instead of being assembled from whatever was nearest.
"""

import logging
import time
from dataclasses import dataclass, field

from app.services import claim_support
from app.services.gemini_service import generate_text_with_usage
from app.services.rag_citations import Source, validate
from app.services.rag_enhanced import EnhancedRetrieval, EnhancedRetriever
from app.services.rag_prompt import cited_prompt, plain_prompt, rag_prompt
from app.services.rag_response import (
    ANSWERED,
    CitedAnswer,
    confidence_of,
    disabled,
    insufficient,
)
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
    #: Day 24: the same answer in the schema every caller can rely on -
    #: sources, citations, confidence and status. Always present.
    cited: CitedAnswer | None = None

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

        if self.cited is not None:
            record.update(self.cited.as_dict())

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
        check_support: bool = True,
    ) -> None:
        self._retriever = retriever or RAGRetriever()
        self._top_k = top_k
        self._settings = settings
        self._enhanced = enhanced or EnhancedRetriever(retriever=self._retriever, settings=settings)
        #: Whether the support check may use a model. Off, it still runs -
        #: on words rather than meaning - and says which it was.
        self._check_support = check_support

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
        answer = generated.text.strip()

        if mode == OFF:
            # No retrieval happened, so there is nothing to cite. The empty
            # lists are the honest report, not a missing feature.
            cited = disabled(answer)
        else:
            # BASELINE names the documents that went into the prompt, but
            # offers no quotes: nothing was asked for and nothing was
            # checked, so its confidence is never better than low. That is
            # the difference Day 24 is about, stated in the schema itself.
            cited = CitedAnswer(
                answer=answer,
                rag_status=ANSWERED,
                sources=[Source.of(chunk) for chunk in (retrieval.chunks if retrieval else ())],
            )

        return RagAnswer(
            question=text,
            answer=answer,
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
            cited=cited,
        )

    async def _ask_enhanced(self, text: str, settings: RagSettings | None) -> RagAnswer:
        """ENHANCED: the second stage, the relevance gate, and the citations.

        Three things decide the answer, in this order: whether anything
        survived the filter, whether what survived is relevant enough to
        answer from, and whether what the model then said can be traced back
        to it. The first two can stop the model being called at all.
        """
        found = await self._enhanced.retrieve(text, settings=settings)
        active = found.settings
        relevance = found.best_relevance
        threshold = active.answer_threshold

        if found.nothing_relevant or relevance < threshold:
            return self._refuse(text, found, relevance, threshold)

        logger.info(
            "[Relevance] Best score: %.3f · Threshold: %.2f · Status: sufficient_context",
            relevance,
            threshold,
        )
        prompt = cited_prompt(text, found.chunks)
        logger.info("Asking the model (mode=%s): %r", ENHANCED, text[:60])
        generated, llm_seconds = await self._generate(prompt)

        # Everything the model said about where its facts came from is
        # checked here, against the chunks that were actually retrieved.
        checked = validate(generated.text, found.chunks)
        support = await claim_support.check(
            checked.answer,
            [citation.quote for citation in checked.citations],
            use_model=self._check_support,
        )
        cited = CitedAnswer(
            answer=checked.answer,
            rag_status=ANSWERED,
            confidence=confidence_of(
                ANSWERED, relevance, threshold, checked.citations, support.verdict
            ),
            citation_support=support.verdict,
            sources=list(checked.sources),
            citations=list(checked.citations),
            rejected_citations=list(checked.rejected),
            best_relevance=relevance,
            answer_threshold=threshold,
            unsupported_claims=list(support.unsupported),
            support_checked_by=support.checked_by,
        )

        return RagAnswer(
            question=text,
            answer=cited.answer,
            rag_enabled=True,
            mode=ENHANCED,
            # Only the chunks a surviving citation points at are reported as
            # having been used; the rest were context the answer did not
            # lean on, and saying otherwise would overstate the evidence.
            retrieved_chunks=self._used(found.chunks, cited),
            sources=[f"{source.file} / {source.section}".rstrip(" /") for source in cited.sources],
            retrieval_seconds=found.seconds,
            llm_seconds=llm_seconds,
            tokens_used=(generated.input_tokens or 0) + (generated.output_tokens or 0),
            top_k=active.final_top_k,
            embedding_model=found.retrieval.embedding_model,
            prompt=prompt,
            enhanced=found,
            cited=cited,
        )

    def _refuse(
        self, text: str, found: EnhancedRetrieval, relevance: float, threshold: float
    ) -> RagAnswer:
        """The answer for a question the index cannot support.

        Either nothing cleared the similarity filter, or what did is not
        relevant enough to build on. Sending the closest chunks anyway is
        exactly how a confident paragraph gets assembled out of nothing, so
        the model is not asked and no tokens are spent.
        """
        logger.info(
            "[Relevance] Best score: %.3f · Threshold: %.2f · Status: insufficient_context",
            relevance,
            threshold,
        )
        logger.info("[Agent] Answering skipped · Reason: insufficient context")
        cited = insufficient(relevance, threshold)

        return RagAnswer(
            question=text,
            answer=cited.answer,
            rag_enabled=True,
            mode=ENHANCED,
            retrieval_seconds=found.seconds,
            top_k=found.settings.final_top_k,
            embedding_model=found.retrieval.embedding_model,
            prompt="",
            enhanced=found,
            cited=cited,
        )

    @staticmethod
    def _used(chunks, cited: CitedAnswer) -> list[RetrievedChunk]:
        """The chunks a citation actually points at, in the order they were
        ranked. Falls back to everything that was shown when the model cited
        nothing - an uncited answer still has to say what it was given."""
        wanted = {source.chunk_id for source in cited.sources}

        if not wanted:
            return list(chunks)

        return [chunk for chunk in chunks if chunk.chunk_id in wanted]

    async def _generate(self, prompt: str):
        started = time.perf_counter()
        generated = await generate_text_with_usage(prompt)

        return generated, time.perf_counter() - started
