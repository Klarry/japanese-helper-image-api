"""Day 21: the local vector index, and the metadata that goes with it.

FAISS holds the vectors and nothing else - it does not know what a chunk is.
So each index directory holds two files that only mean something together:

    data/index/<strategy>/index.faiss     vector 0, vector 1, vector 2 ...
    data/index/<strategy>/metadata.json   chunks[0], chunks[1], chunks[2] ...

The rule the whole thing rests on: **position is the join**. FAISS vector
number N describes ``metadata["chunks"][N]``. Nothing reorders either side
after they are written, the two are always written in one go, and loading
refuses a pair whose lengths disagree rather than answering with the wrong
chunk.

Vectors are L2-normalised and the index is inner-product, which makes the
score a cosine similarity - the usual thing to want from text embeddings,
and the reason no other index type is needed here.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"


class VectorIndexError(RuntimeError):
    """The index could not be built, written or read back."""


@dataclass(frozen=True)
class LoadedIndex:
    """An index read back from disk, with its metadata beside it."""

    index: Any
    metadata: list[dict]
    embedding_model: str
    dimension: int
    strategy: str

    def __len__(self) -> int:
        return len(self.metadata)


def _faiss():
    try:
        import faiss
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise VectorIndexError(
            "faiss is not installed. Install it with 'pip install faiss-cpu' "
            "(it is in requirements.txt)."
        ) from error

    return faiss


def _as_array(vectors: list[list[float]]):
    import numpy

    if not vectors:
        raise VectorIndexError("there is nothing to index: no vectors were produced")

    array = numpy.asarray(vectors, dtype="float32")

    if array.ndim != 2:
        raise VectorIndexError("vectors must all have the same length")

    return array


def build_index(vectors: list[list[float]]):
    """A flat inner-product index over normalised vectors, i.e. cosine."""
    faiss = _faiss()
    array = _as_array(vectors)
    # Normalise a copy: the caller's vectors are also written nowhere else,
    # but the index and any later query have to agree on the convention.
    faiss.normalize_L2(array)
    index = faiss.IndexFlatIP(array.shape[1])
    index.add(array)

    if index.ntotal != len(vectors):
        raise VectorIndexError(f"FAISS took {index.ntotal} of {len(vectors)} vectors")

    return index


def save_index(
    directory: Path,
    vectors: list[list[float]],
    metadata: list[dict],
    strategy: str,
    embedding_model: str,
) -> dict:
    """Write the vectors and their metadata together, in the same order.

    Returns what was written, so the caller can report it without reading
    the files back.
    """
    if len(vectors) != len(metadata):
        raise VectorIndexError(
            f"{len(vectors)} vectors and {len(metadata)} metadata records cannot be aligned"
        )

    faiss = _faiss()
    index = build_index(vectors)
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(directory / INDEX_FILE))

    document = {
        "strategy": strategy,
        "embedding_model": embedding_model,
        "dimension": index.d,
        "count": len(metadata),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # The list below is the index: chunks[N] belongs to FAISS vector N.
        "chunks": metadata,
    }
    (directory / METADATA_FILE).write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.debug("Index written: %s (%d vectors, dim %d)", directory, len(vectors), index.d)

    return {"count": len(metadata), "dimension": index.d, "path": str(directory)}


def load_index(directory: Path) -> LoadedIndex:
    """Read an index and its metadata back, refusing a mismatched pair."""
    faiss = _faiss()
    index_path = directory / INDEX_FILE
    metadata_path = directory / METADATA_FILE

    if not index_path.exists() or not metadata_path.exists():
        raise VectorIndexError(f"{directory} does not hold both {INDEX_FILE} and {METADATA_FILE}")

    index = faiss.read_index(str(index_path))
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks = document.get("chunks")

    if not isinstance(chunks, list):
        raise VectorIndexError(f"{metadata_path} has no 'chunks' list")

    if index.ntotal != len(chunks):
        raise VectorIndexError(
            f"{directory} is out of step: {index.ntotal} vectors, {len(chunks)} metadata records"
        )

    return LoadedIndex(
        index=index,
        metadata=chunks,
        embedding_model=str(document.get("embedding_model", "")),
        dimension=int(document.get("dimension", index.d)),
        strategy=str(document.get("strategy", "")),
    )


def search(loaded: LoadedIndex, vector: list[float], limit: int = 5) -> list[dict]:
    """The nearest chunks to a query vector, each with its score.

    Not needed to build an index - it is here because it is the one thing
    that proves the join works, and because retrieval is the next step.
    """
    import numpy

    faiss = _faiss()
    query = numpy.asarray([vector], dtype="float32")
    faiss.normalize_L2(query)
    scores, positions = loaded.index.search(query, min(limit, len(loaded)))

    return [
        {**loaded.metadata[position], "score": float(score)}
        for score, position in zip(scores[0], positions[0])
        if position >= 0
    ]
