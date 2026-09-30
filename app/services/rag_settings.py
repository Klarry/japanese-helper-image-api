"""Day 23: the numbers the second retrieval stage is steered by.

Every value the assignment says must be adjustable lives here and nowhere
else: how many chunks FAISS is asked for, the similarity a chunk has to
reach to survive, how many survive into the prompt, and how the reranker
weighs what it knows. Nothing downstream carries a default of its own, so
there is exactly one place to change any of them - and a caller that wants
different numbers for one question passes a different settings object
instead of reaching into the retriever.

Read from the environment directly rather than imported from
``app.core.config``, for the reason ``embedding_service`` does the same:
config requires GEMINI_API_KEY at import time, and the filter, the reranker
and their tests have to run without a key. The names and the defaults are
the ones config declares.
"""

import os
from dataclasses import dataclass, replace

# How many chunks FAISS is asked for, before anything is thrown away. More
# than the prompt will hold, on purpose: filtering can only remove.
RETRIEVAL_TOP_K = int(os.getenv("RAG_RETRIEVAL_TOP_K", "10"))
# The cosine similarity a chunk has to reach to be considered relevant at
# all. Below it, a chunk is not a worse answer - it is a different subject.
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.70"))
# How many chunks reach the model after filtering and reranking.
FINAL_TOP_K = int(os.getenv("RAG_FINAL_TOP_K", "3"))
# How the reranker weighs a chunk: how close the vectors are, against how
# much of the query's own vocabulary the chunk actually uses.
SIMILARITY_WEIGHT = float(os.getenv("RAG_SIMILARITY_WEIGHT", "0.7"))
KEYWORD_WEIGHT = float(os.getenv("RAG_KEYWORD_WEIGHT", "0.3"))
# Within the keyword half, how much of it is decided by the chunk's own
# label - its file and section - rather than by its body.
SECTION_WEIGHT = float(os.getenv("RAG_SECTION_WEIGHT", "0.4"))
# Whether the question is rewritten into a search query before retrieval.
QUERY_REWRITE_ENABLED = os.getenv("RAG_QUERY_REWRITE_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@dataclass(frozen=True)
class RagSettings:
    """One set of knobs for one run of the enhanced pipeline."""

    retrieval_top_k: int = RETRIEVAL_TOP_K
    similarity_threshold: float = SIMILARITY_THRESHOLD
    final_top_k: int = FINAL_TOP_K
    similarity_weight: float = SIMILARITY_WEIGHT
    keyword_weight: float = KEYWORD_WEIGHT
    section_weight: float = SECTION_WEIGHT
    query_rewrite: bool = QUERY_REWRITE_ENABLED

    def __post_init__(self) -> None:
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be at least 1")

        if self.final_top_k < 1:
            raise ValueError("final_top_k must be at least 1")

        if self.final_top_k > self.retrieval_top_k:
            raise ValueError(
                f"final_top_k ({self.final_top_k}) cannot be larger than retrieval_top_k "
                f"({self.retrieval_top_k}) - filtering only ever removes chunks"
            )

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold is a cosine similarity, between 0 and 1")

        if not 0.0 <= self.section_weight <= 1.0:
            raise ValueError("section_weight is a share of the keyword half, between 0 and 1")

        if self.similarity_weight < 0 or self.keyword_weight < 0:
            raise ValueError("the reranker weights cannot be negative")

        if self.similarity_weight + self.keyword_weight <= 0:
            raise ValueError("at least one reranker weight has to be above zero")

    def with_overrides(self, **changes) -> "RagSettings":
        """The same settings with some of them replaced - the way one call
        asks for a different threshold without changing anyone else's."""
        given = {name: value for name, value in changes.items() if value is not None}

        return replace(self, **given) if given else self

    def as_dict(self) -> dict:
        return {
            "retrieval_top_k": self.retrieval_top_k,
            "similarity_threshold": self.similarity_threshold,
            "final_top_k": self.final_top_k,
            "similarity_weight": self.similarity_weight,
            "keyword_weight": self.keyword_weight,
            "section_weight": self.section_weight,
            "query_rewrite": self.query_rewrite,
        }


DEFAULT_SETTINGS = RagSettings()
