"""Day 23: throwing away what the question was not about.

FAISS always returns something. Ask it for ten chunks and it gives ten, even
when the index holds nothing on the subject - the tenth is simply the least
unlike the question of everything there is. On Day 22 all five went into the
prompt, which is why the one question with no answer in the documents still
arrived at the model with five documents attached.

The filter is the smallest possible fix: a similarity a chunk has to reach
to count as being about the question at all. Below it the chunk is dropped
and its score is kept, so a run can say what it threw away and why rather
than only how much survived. When nothing survives, nothing survives - an
empty result is the honest answer for a question the index cannot answer,
and it is what keeps invented facts out of the prompt.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.services.rag_retriever import RetrievedChunk
from app.services.rag_settings import DEFAULT_SETTINGS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Filtered:
    """What the threshold kept, and what it did not."""

    kept: tuple[RetrievedChunk, ...]
    dropped: tuple[RetrievedChunk, ...]
    threshold: float

    @property
    def everything_was_dropped(self) -> bool:
        """True when the index had nothing relevant - not an error, and the
        one case where the model must be told so instead of being handed
        the closest chunks anyway."""
        return not self.kept and bool(self.dropped)

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "kept": len(self.kept),
            "dropped": len(self.dropped),
            "dropped_chunks": [
                {"chunk_id": chunk.chunk_id, "reference": chunk.reference, "score": round(chunk.score, 4)}
                for chunk in self.dropped
            ],
        }


class RelevanceFilter:
    """A similarity threshold, and nothing else."""

    def __init__(self, threshold: float = DEFAULT_SETTINGS.similarity_threshold) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("the threshold is a cosine similarity, between 0 and 1")

        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def apply(self, chunks: Sequence[RetrievedChunk], threshold: float | None = None) -> Filtered:
        """Split what was retrieved into what is relevant and what is not."""
        limit = self._threshold if threshold is None else threshold
        kept = tuple(chunk for chunk in chunks if chunk.score >= limit)
        dropped = tuple(chunk for chunk in chunks if chunk.score < limit)

        if dropped:
            logger.info(
                "Filter: %d of %d chunk(s) below %.2f dropped (%s)",
                len(dropped),
                len(chunks),
                limit,
                ", ".join(f"{chunk.score:.3f}" for chunk in dropped),
            )

        return Filtered(kept=kept, dropped=dropped, threshold=limit)
