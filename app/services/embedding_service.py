"""Day 21: turning chunks into vectors.

One service with one job - ``embed(text) -> vector`` - so nothing else in
the pipeline knows who produces the numbers. The real backend is the
project's existing Gemini integration: the same key, the same client, the
same error handling as every other call, through
``gemini_service.embed_texts`` and its batch endpoint.

There is a second backend, and it is honest about what it is: ``local``
hashes tokens into a fixed number of dimensions and normalises the result.
It is not a language model and cannot be mistaken for one - it exists so
that the tests, and anyone without a key, can run the whole pipeline end to
end offline. Which backend produced an index is written into the index's own
metadata and into the comparison report, so a run can always be traced back
to what embedded it.
"""

import hashlib
import logging
import math
import os
import re
from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

# Read from the environment rather than imported from app.core.config, for
# the same reason the digest storage does it: config requires GEMINI_API_KEY
# at import time, and the local backend exists precisely for runs that have
# no key. The names and the defaults are the ones config declares.
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

GEMINI = "gemini"
LOCAL = "local"
LOCAL_NAME = "local-hashing"
LOCAL_DIMENSION = 768

_TOKENS = re.compile(r"\w+", re.UNICODE)


class EmbeddingService:
    """Embeddings for a chunk, or for a few hundred of them."""

    def __init__(
        self,
        backend: str = GEMINI,
        model: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        dimension: int = LOCAL_DIMENSION,
    ) -> None:
        if backend not in (GEMINI, LOCAL):
            raise ValueError(f"unknown embedding backend: {backend}")

        self.backend = backend
        self.model = model
        self.batch_size = max(1, batch_size)
        self._dimension = dimension

    @property
    def name(self) -> str:
        """What to record as having made these vectors."""
        return self.model if self.backend == GEMINI else LOCAL_NAME

    async def embed(self, text: str) -> list[float]:
        """One text, one vector."""
        vectors = await self.embed_all([text])

        return vectors[0]

    async def embed_all(
        self,
        texts: Sequence[str],
        on_batch: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """Every text, in order, in batches.

        ``on_batch(done, total)`` is called after each batch, which is how
        the indexer reports progress without this service knowing about logs.
        """
        vectors: list[list[float]] = []

        for begin in range(0, len(texts), self.batch_size):
            batch = list(texts[begin : begin + self.batch_size])
            vectors.extend(await self._embed_batch(batch))

            if on_batch is not None:
                on_batch(len(vectors), len(texts))

        self._check(vectors)

        return vectors

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        if self.backend == LOCAL:
            return [self._hashed(text) for text in batch]

        from app.services.gemini_service import embed_texts

        return await embed_texts(batch, model=self.model)

    def _check(self, vectors: list[list[float]]) -> None:
        """Every vector has to have the same length, or the index cannot be
        built from them - better to say so here than to fail inside FAISS."""
        dimensions = {len(vector) for vector in vectors}

        if len(dimensions) > 1:
            raise ValueError(f"embeddings came back with different dimensions: {sorted(dimensions)}")

    def _hashed(self, text: str) -> list[float]:
        """A deterministic bag-of-words vector: each token lands in one
        dimension by its hash, and the result is normalised so that
        similarity is a dot product, exactly like a real embedding."""
        vector = [0.0] * self._dimension

        for token in _TOKENS.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign

        length = math.sqrt(sum(value * value for value in vector))

        if length == 0:
            # An empty or symbol-only chunk still needs a vector of the right
            # shape; a single fixed dimension keeps it far from everything else.
            vector[0] = 1.0
            return vector

        return [value / length for value in vector]
