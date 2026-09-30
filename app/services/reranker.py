"""Day 23: putting the surviving chunks in a better order.

Similarity alone is a blunt sort. Two chunks can sit the same distance from
a question's vector while one of them says ``MIN_INTERVAL_SECONDS`` and the
other only talks about scheduling in general - and an embedding, which
reads meaning rather than letters, will not reliably prefer the one with the
identifier in it.

So the second stage scores each chunk again on something the first stage is
bad at: how much of the query's own vocabulary the chunk actually uses, in
its body and in its label. The two halves are combined with configurable
weights and nothing else - no second model, no training, no hidden feature.
That is on purpose. The point of this stage is that a person can read a
score and say why a chunk moved, which stops being true the moment the
reranker becomes something one has to trust rather than check.

    final_score = similarity_weight * similarity + keyword_weight * keyword_score
    keyword_score = (1 - section_weight) * words_in_text + section_weight * words_in_label

Every score stays between 0 and 1, so they can be compared across questions.
"""

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from app.services.rag_retriever import RetrievedChunk
from app.services.rag_settings import DEFAULT_SETTINGS

logger = logging.getLogger(__name__)

_WORDS = re.compile(r"[\w]+", re.UNICODE)
MIN_TERM_LENGTH = 3
# Words that appear in any question and prove nothing about a chunk.
_NOISE = frozenset(
    """
    the and or not for with from that this these those what which how why when where who
    does did are was were has have had can could should would will its it is be been
    и в на с по за из от до для о об при это эти тот как что где когда почему чем
    """.split()
)


@dataclass(frozen=True)
class RerankedChunk:
    """One chunk, with the three numbers that decided where it ended up."""

    chunk: RetrievedChunk
    similarity_score: float
    keyword_score: float
    text_score: float
    label_score: float
    rerank_score: float
    matched: tuple[str, ...]

    @property
    def reference(self) -> str:
        return self.chunk.reference

    @property
    def moved(self) -> bool:
        """Whether the reranker disagreed with the similarity order enough
        to change this chunk's score away from it."""
        return round(self.rerank_score, 4) != round(self.similarity_score, 4)

    def as_dict(self, with_text: bool = False) -> dict:
        record = self.chunk.as_dict(with_text)
        record.update(
            {
                "similarity_score": round(self.similarity_score, 4),
                "keyword_score": round(self.keyword_score, 4),
                "text_score": round(self.text_score, 4),
                "label_score": round(self.label_score, 4),
                "rerank_score": round(self.rerank_score, 4),
                "matched_terms": list(self.matched),
            }
        )

        return record


@dataclass(frozen=True)
class Reranked:
    """The chunks in their new order, and how long it took."""

    chunks: tuple[RerankedChunk, ...]
    seconds: float = 0.0

    @property
    def reordered(self) -> bool:
        """Whether the new order differs from the similarity order."""
        by_similarity = sorted(self.chunks, key=lambda item: item.similarity_score, reverse=True)

        return [item.chunk.chunk_id for item in by_similarity] != [
            item.chunk.chunk_id for item in self.chunks
        ]

    def top(self, count: int) -> tuple[RerankedChunk, ...]:
        if count < 1:
            raise ValueError("final_top_k must be at least 1")

        return self.chunks[:count]


def terms(query: str) -> tuple[str, ...]:
    """The words of a query worth matching on: long enough to mean
    something, not in every sentence, each counted once."""
    found = [word.lower() for word in _WORDS.findall(query)]
    picked = [
        word
        for word in found
        if word not in _NOISE and (len(word) >= MIN_TERM_LENGTH or any(c.isdigit() for c in word))
    ]

    return tuple(dict.fromkeys(picked))


def overlap(text: str, wanted: Sequence[str]) -> tuple[float, tuple[str, ...]]:
    """What share of the query's terms this text uses, and which ones.

    Substring matching, deliberately: ``retrieval`` should count as a match
    for ``retriever``, and an identifier written ``final_top_k`` should be
    found by the term ``top``.
    """
    if not wanted:
        return 0.0, ()

    lowered = text.lower()
    hit = tuple(word for word in wanted if word in lowered)

    return round(len(hit) / len(wanted), 4), hit


class Reranker:
    """Similarity, plus what similarity does not see."""

    def __init__(
        self,
        similarity_weight: float = DEFAULT_SETTINGS.similarity_weight,
        keyword_weight: float = DEFAULT_SETTINGS.keyword_weight,
        section_weight: float = DEFAULT_SETTINGS.section_weight,
    ) -> None:
        if similarity_weight < 0 or keyword_weight < 0:
            raise ValueError("the weights cannot be negative")

        total = similarity_weight + keyword_weight

        if total <= 0:
            raise ValueError("at least one weight has to be above zero")

        if not 0.0 <= section_weight <= 1.0:
            raise ValueError("section_weight is a share of the keyword half, between 0 and 1")

        self._similarity_weight = similarity_weight
        self._keyword_weight = keyword_weight
        self._section_weight = section_weight
        self._total = total

    @property
    def weights(self) -> dict:
        return {
            "similarity_weight": self._similarity_weight,
            "keyword_weight": self._keyword_weight,
            "section_weight": self._section_weight,
        }

    def score(self, chunk: RetrievedChunk, wanted: Sequence[str]) -> RerankedChunk:
        """One chunk's three numbers."""
        text_score, in_text = overlap(chunk.text, wanted)
        label_score, in_label = overlap(f"{chunk.file} {chunk.title} {chunk.section}", wanted)
        keyword_score = round(
            (1 - self._section_weight) * text_score + self._section_weight * label_score, 4
        )
        rerank = (
            self._similarity_weight * chunk.score + self._keyword_weight * keyword_score
        ) / self._total

        return RerankedChunk(
            chunk=chunk,
            similarity_score=chunk.score,
            keyword_score=keyword_score,
            text_score=text_score,
            label_score=label_score,
            rerank_score=round(rerank, 4),
            matched=tuple(dict.fromkeys(in_text + in_label)),
        )

    def rerank(self, chunks: Sequence[RetrievedChunk], query: str) -> Reranked:
        """The chunks, rescored and reordered, best first.

        Ties are broken by similarity and then by chunk id, so the same
        input always produces the same order - a reranker whose output moves
        between runs cannot be compared against anything.
        """
        started = time.perf_counter()
        wanted = terms(query)
        scored = [self.score(chunk, wanted) for chunk in chunks]
        scored.sort(
            key=lambda item: (-item.rerank_score, -item.similarity_score, item.chunk.chunk_id)
        )
        seconds = time.perf_counter() - started

        if scored:
            logger.info(
                "Reranked %d chunk(s) on %d term(s): %s",
                len(scored),
                len(wanted),
                ", ".join(f"{item.rerank_score:.3f} {item.reference}" for item in scored[:3]),
            )

        return Reranked(chunks=tuple(scored), seconds=seconds)
