"""Day 23: the second stage, end to end.

Day 22 was one step: embed the question, take the five nearest chunks, ask
the model. This is the same search with three things around it -

    question -> rewrite -> FAISS (10) -> filter (>= 0.70) -> rerank -> top 3

- and one rule that only shows up when the index has nothing to offer:
**if the filter keeps nothing, nothing is sent.** The model is told the
index has no answer instead of being handed the ten least-unlike chunks,
which is the difference between an honest "not in the documents" and a
confident paragraph assembled from whatever was nearest.

Nothing here replaces the Day 22 retriever: it is called, once, for the
first stage. What this module adds is what happens to its result afterwards,
and a record of it detailed enough to argue with - every chunk that was
dropped, with its score, and every chunk that survived, with the three
numbers that decided its place.
"""

import logging
import time
from dataclasses import dataclass, field

from app.services.query_rewriter import NONE, QueryRewriter, RewrittenQuery
from app.services.rag_filter import Filtered, RelevanceFilter
from app.services.rag_retriever import RAGRetriever, Retrieval
from app.services.rag_settings import DEFAULT_SETTINGS, RagSettings
from app.services.reranker import Reranked, RerankedChunk, Reranker

logger = logging.getLogger(__name__)


@dataclass
class EnhancedRetrieval:
    """One trip through the second stage, with the evidence."""

    query: RewrittenQuery
    retrieval: Retrieval
    filtered: Filtered
    reranked: Reranked
    final: tuple[RerankedChunk, ...]
    settings: RagSettings = field(default=DEFAULT_SETTINGS)
    seconds: float = 0.0

    @property
    def chunks(self) -> list:
        """The chunks that reach the prompt, as the rest of the code knows
        them - a plain RetrievedChunk, in the reranked order."""
        return [item.chunk for item in self.final]

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []

        for item in self.final:
            if item.reference not in seen:
                seen.append(item.reference)

        return seen

    @property
    def nothing_relevant(self) -> bool:
        """The index was searched and had nothing about this question."""
        return not self.final

    @property
    def best_relevance(self) -> float:
        """How relevant the best surviving chunk is, after reranking.

        The rerank score rather than the raw similarity, because this is the
        number the second stage actually produced - and because a chunk that
        is close in vector space but uses none of the question's words is
        exactly the one an answer should not be built on. Both numbers are
        reported, so a decision taken on this one can always be checked
        against the other.
        """
        return max((item.rerank_score for item in self.final), default=0.0)

    @property
    def best_similarity(self) -> float:
        """The best raw cosine similarity of everything retrieved, before
        anything was thrown away."""
        return max((chunk.score for chunk in self.retrieval.chunks), default=0.0)

    def as_dict(self, with_text: bool = False) -> dict:
        """What the assignment asks Enhanced RAG to report."""
        return {
            "original_query": self.query.original,
            "rewritten_query": self.query.query,
            "rewrite_used": self.query.used,
            "retrieval_top_k": self.settings.retrieval_top_k,
            "retrieved_count": len(self.retrieval.chunks),
            "filtered_count": len(self.filtered.kept),
            "final_count": len(self.final),
            "threshold": self.settings.similarity_threshold,
            "final_top_k": self.settings.final_top_k,
            "weights": {
                "similarity_weight": self.settings.similarity_weight,
                "keyword_weight": self.settings.keyword_weight,
                "section_weight": self.settings.section_weight,
            },
            "dropped_by_filter": [
                {
                    "chunk_id": chunk.chunk_id,
                    "reference": chunk.reference,
                    "similarity_score": round(chunk.score, 4),
                }
                for chunk in self.filtered.dropped
            ],
            "reordered": self.reranked.reordered,
            "best_relevance": round(self.best_relevance, 4),
            "best_similarity": round(self.best_similarity, 4),
            "answer_threshold": self.settings.answer_threshold,
            "latency": {
                "rewrite_seconds": round(self.query.seconds, 3),
                "retrieval_seconds": round(self.retrieval.seconds, 3),
                "rerank_seconds": round(self.reranked.seconds, 3),
                "total_seconds": round(self.seconds, 3),
            },
            "chunks": [item.as_dict(with_text) for item in self.final],
        }


class EnhancedRetriever:
    """Rewrite, retrieve, filter, rerank, keep the best few."""

    def __init__(
        self,
        retriever: RAGRetriever | None = None,
        settings: RagSettings = DEFAULT_SETTINGS,
        rewriter: QueryRewriter | None = None,
        relevance_filter: RelevanceFilter | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._retriever = retriever or RAGRetriever()
        self._settings = settings
        self._rewriter = rewriter or QueryRewriter(enabled=settings.query_rewrite)
        self._filter = relevance_filter or RelevanceFilter(settings.similarity_threshold)
        self._reranker = reranker or Reranker(
            similarity_weight=settings.similarity_weight,
            keyword_weight=settings.keyword_weight,
            section_weight=settings.section_weight,
        )

    @property
    def retriever(self) -> RAGRetriever:
        return self._retriever

    @property
    def settings(self) -> RagSettings:
        return self._settings

    async def retrieve(
        self, question: str, settings: RagSettings | None = None
    ) -> EnhancedRetrieval:
        """The whole second stage for one question."""
        text = question.strip()

        if not text:
            raise ValueError("there is no question to retrieve for")

        active = settings or self._settings
        started = time.perf_counter()

        rewritten = await self._rewriter.rewrite(text, enabled=active.query_rewrite)
        # The search gets the rewrite; the model, later, gets `text`.
        retrieval = await self._retriever.retrieve(
            rewritten.query, top_k=active.retrieval_top_k
        )
        filtered = self._filter.apply(retrieval.chunks, threshold=active.similarity_threshold)
        # Reranking reads the question's own words, not the rewrite's: the
        # rewrite may have added terms the person never used, and a chunk
        # should not be promoted for matching a guess.
        reranked = self._reranker.rerank(filtered.kept, text)
        final = reranked.top(active.final_top_k) if filtered.kept else ()
        seconds = time.perf_counter() - started

        # Day 24: the funnel, one labelled line per stage, so a run can be
        # read off the log without opening anything.
        logger.info("[Retrieval] Retrieved: %d", len(retrieval.chunks))
        logger.info(
            "[Filtering] Threshold: %.2f · Remaining: %d",
            active.similarity_threshold,
            len(filtered.kept),
        )
        logger.info("[Reranking] Final top-K: %d", len(final))

        return EnhancedRetrieval(
            query=rewritten,
            retrieval=retrieval,
            filtered=filtered,
            reranked=reranked,
            final=final,
            settings=active,
            seconds=seconds,
        )
