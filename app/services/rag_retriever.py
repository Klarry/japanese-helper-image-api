"""Day 22: finding the chunks a question is about.

The index was built on Day 21 and nothing here changes it. This is the read
side: embed the question, ask FAISS for the nearest vectors, and hand back
the chunks those vectors stand for - with their file, their section and
their score, because an answer built on them has to be able to say where it
came from.

One detail decides whether any of this works: **the question has to be
embedded by whatever embedded the index**. A question in 768 dimensions
cannot be compared with vectors in 3072, and even at the same size, two
different models put the same sentence in different places. So the index is
asked what made it - ``metadata.json`` records the model - and the retriever
uses that, rather than whatever the environment happens to default to.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from app.services.chunking import FIXED, STRUCTURAL
from app.services.document_loader import PROJECT_ROOT
from app.services.embedding_service import GEMINI, LOCAL, LOCAL_NAME, EmbeddingService
from app.services.vector_index import LoadedIndex, VectorIndexError, load_index, search

logger = logging.getLogger(__name__)

INDEX_DIR = PROJECT_ROOT / "data" / "index"
DIRECTORIES = {FIXED: "fixed_size", STRUCTURAL: "structural"}
DEFAULT_STRATEGY = STRUCTURAL
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk the search returned, and how close it was."""

    chunk_id: str
    text: str
    source: str
    file: str
    title: str
    section: str
    strategy: str
    position: int
    score: float

    @property
    def reference(self) -> str:
        """How this chunk is named in an answer: "README.md / Architecture"."""
        return f"{self.file} / {self.section}" if self.section else self.file

    def as_dict(self, with_text: bool = True) -> dict:
        record = {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "file": self.file,
            "title": self.title,
            "section": self.section,
            "strategy": self.strategy,
            "position": self.position,
            "score": round(self.score, 4),
        }

        if with_text:
            record["text"] = self.text

        return record


@dataclass(frozen=True)
class Retrieval:
    """What one question found."""

    query: str
    chunks: tuple[RetrievedChunk, ...]
    top_k: int
    embedding_model: str
    strategy: str
    seconds: float

    @property
    def sources(self) -> list[str]:
        """The documents behind these chunks, in the order they were found,
        without repeats - and nothing that was not retrieved."""
        seen: list[str] = []

        for chunk in self.chunks:
            if chunk.reference not in seen:
                seen.append(chunk.reference)

        return seen

    @property
    def files(self) -> list[str]:
        seen: list[str] = []

        for chunk in self.chunks:
            if chunk.file not in seen:
                seen.append(chunk.file)

        return seen

    def as_dict(self, with_text: bool = False) -> dict:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "strategy": self.strategy,
            "embedding_model": self.embedding_model,
            "seconds": round(self.seconds, 3),
            "sources": self.sources,
            "chunks": [chunk.as_dict(with_text) for chunk in self.chunks],
        }


class RAGRetriever:
    """The index, queried by question."""

    def __init__(
        self,
        strategy: str = DEFAULT_STRATEGY,
        index_dir: Path = INDEX_DIR,
        embeddings: EmbeddingService | None = None,
    ) -> None:
        self._strategy = strategy
        self._directory = index_dir / DIRECTORIES.get(strategy, strategy)
        self._given_embeddings = embeddings
        self._index: LoadedIndex | None = None
        self._embeddings: EmbeddingService | None = embeddings

    @property
    def directory(self) -> Path:
        return self._directory

    def index(self) -> LoadedIndex:
        """The index, loaded once. Raises VectorIndexError when it is not
        there - which means nobody has run the indexer yet."""
        if self._index is None:
            self._index = load_index(self._directory)
            logger.info(
                "Index loaded: %s (%d chunks, %s, dim %d)",
                self._directory,
                len(self._index),
                self._index.embedding_model,
                self._index.dimension,
            )

        return self._index

    def embeddings(self) -> EmbeddingService:
        """The service that embedded this index, so a question lands in the
        same space as the chunks."""
        if self._embeddings is None:
            model = self.index().embedding_model
            self._embeddings = (
                EmbeddingService(backend=LOCAL)
                if model == LOCAL_NAME
                else EmbeddingService(backend=GEMINI, model=model)
            )

        return self._embeddings

    async def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> Retrieval:
        """The ``top_k`` chunks closest to ``question``, nearest first."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        text = question.strip()

        if not text:
            raise ValueError("there is no question to retrieve for")

        loaded = self.index()
        embeddings = self.embeddings()
        started = time.perf_counter()
        vector = await embeddings.embed(text)

        if len(vector) != loaded.dimension:
            raise VectorIndexError(
                f"the question was embedded in {len(vector)} dimensions by "
                f"'{embeddings.name}', but {self._directory} was built in "
                f"{loaded.dimension} by '{loaded.embedding_model}'. Rebuild the index, "
                "or query it with the model that built it."
            )

        found = search(loaded, vector, limit=top_k)
        seconds = time.perf_counter() - started

        chunks = tuple(
            RetrievedChunk(
                chunk_id=str(record.get("chunk_id", "")),
                text=str(record.get("text", "")),
                source=str(record.get("source", "")),
                file=str(record.get("file", "")),
                title=str(record.get("title", "")),
                section=str(record.get("section", "")),
                strategy=str(record.get("strategy", "")),
                position=int(record.get("position", -1)),
                score=float(record.get("score", 0.0)),
            )
            for record in found
        )
        logger.info(
            "Retrieved %d chunk(s) for %r in %.3fs: %s",
            len(chunks),
            text[:60],
            seconds,
            ", ".join(chunk.reference for chunk in chunks),
        )

        return Retrieval(
            query=text,
            chunks=chunks,
            top_k=top_k,
            embedding_model=embeddings.name,
            strategy=self._strategy,
            seconds=seconds,
        )
