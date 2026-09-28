"""Day 21: the indexing pipeline, end to end.

    documents -> extraction -> chunking -> embeddings -> FAISS -> metadata.json

Run it from the project root:

    python -m app.index_documents                      # both strategies
    python -m app.index_documents --strategy fixed
    python -m app.index_documents --strategy structural
    python -m app.index_documents --embeddings local   # no API key needed

Each strategy gets its own directory under data/index, and the two are
compared afterwards in data/index/comparison.json - from what the run
actually produced, never from anything written down in advance.
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.services.chunking import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    FIXED,
    MAX_STRUCTURAL_CHARS,
    MIN_STRUCTURAL_CHARS,
    STRUCTURAL,
    Chunk,
    chunk_documents,
)
from fastapi import HTTPException

from app.services.document_loader import CORPUS, PROJECT_ROOT, Document, load_documents
from app.services.embedding_service import EMBEDDING_MODEL, GEMINI, LOCAL, EmbeddingService
from app.services.vector_index import save_index

logger = logging.getLogger("indexer")

INDEX_DIR = PROJECT_ROOT / "data" / "index"
COMPARISON_FILE = "comparison.json"
DIRECTORIES = {FIXED: "fixed_size", STRUCTURAL: "structural"}


@dataclass
class StrategyReport:
    """What one strategy did, measured rather than assumed."""

    strategy: str
    documents: int
    total_characters: int
    chunks: int
    chunk_sizes: list[int] = field(default_factory=list)
    overlap: int = 0
    overlap_applies_to: str = ""
    chunk_size_limit: int = 0
    embedding_model: str = ""
    dimension: int = 0
    chunking_seconds: float = 0.0
    embedding_seconds: float = 0.0
    indexing_seconds: float = 0.0
    index_path: str = ""

    def as_dict(self) -> dict:
        sizes = self.chunk_sizes or [0]

        return {
            "strategy": self.strategy,
            "documents": self.documents,
            "total_characters": self.total_characters,
            "chunks": self.chunks,
            "average_chunk_size": round(sum(sizes) / len(sizes), 1),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "total_chunk_characters": sum(sizes),
            "overlap": self.overlap,
            "overlap_applies_to": self.overlap_applies_to,
            "chunk_size_limit": self.chunk_size_limit,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.dimension,
            "chunking_seconds": round(self.chunking_seconds, 3),
            "embedding_seconds": round(self.embedding_seconds, 3),
            "indexing_seconds": round(self.indexing_seconds, 3),
            "total_seconds": round(
                self.chunking_seconds + self.embedding_seconds + self.indexing_seconds, 3
            ),
            "index_path": self.index_path,
        }


async def index_strategy(
    documents: list[Document],
    strategy: str,
    embeddings: EmbeddingService,
    index_dir: Path,
    chunk_size: int,
    overlap: int,
) -> StrategyReport:
    """One strategy, from chunks to a written index."""
    logger.info("")
    logger.info("Strategy: %s", strategy)

    started = time.perf_counter()
    chunks = chunk_documents(documents, strategy, chunk_size=chunk_size, overlap=overlap)
    chunking_seconds = time.perf_counter() - started
    logger.info("Chunks created: %d", len(chunks))

    logger.info("Generating embeddings...")
    started = time.perf_counter()

    try:
        vectors = await embeddings.embed_all(
            [chunk.text for chunk in chunks],
            on_batch=lambda done, total: logger.info("  embedded %d/%d", done, total),
        )
    except HTTPException as error:
        # The likeliest cause is a model name this API does not serve, and
        # the likeliest fix is another one - so say both rather than only
        # the status code.
        logger.error("Embeddings failed (%s): %s", error.status_code, error.detail)
        raise SystemExit(
            f"Could not embed with '{embeddings.model}'. Try another model with "
            "--embedding-model (or GEMINI_EMBEDDING_MODEL), or run offline with "
            "--embeddings local."
        ) from error

    embedding_seconds = time.perf_counter() - started
    logger.info("Embeddings generated: %d", len(vectors))

    logger.info("Building FAISS index...")
    started = time.perf_counter()
    written = save_index(
        directory=index_dir / DIRECTORIES[strategy],
        vectors=vectors,
        metadata=[chunk.as_metadata() for chunk in chunks],
        strategy=strategy,
        embedding_model=embeddings.name,
    )
    indexing_seconds = time.perf_counter() - started
    logger.info("Index saved: %s", written["path"])

    return StrategyReport(
        strategy=strategy,
        documents=len(documents),
        total_characters=sum(document.size for document in documents),
        chunks=len(chunks),
        chunk_sizes=[chunk.chunk_size for chunk in chunks],
        overlap=overlap,
        overlap_applies_to=(
            "every chunk" if strategy == FIXED else "only blocks longer than the limit"
        ),
        chunk_size_limit=chunk_size if strategy == FIXED else MAX_STRUCTURAL_CHARS,
        embedding_model=embeddings.name,
        dimension=written["dimension"],
        chunking_seconds=chunking_seconds,
        embedding_seconds=embedding_seconds,
        indexing_seconds=indexing_seconds,
        index_path=written["path"],
    )


def comparison(reports: list[StrategyReport], documents: list[Document]) -> dict:
    """The two strategies side by side, plus what they were run over."""
    by_strategy = {report.strategy: report.as_dict() for report in reports}
    result = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "corpus": {
            "documents": len(documents),
            "total_characters": sum(document.size for document in documents),
            "by_source": _counted(documents),
            "roots": [root.path for root in CORPUS],
        },
        "settings": {
            "fixed_chunk_size": DEFAULT_CHUNK_SIZE,
            "fixed_overlap": DEFAULT_OVERLAP,
            "structural_max_chars": MAX_STRUCTURAL_CHARS,
            "structural_min_chars": MIN_STRUCTURAL_CHARS,
        },
        "strategies": by_strategy,
    }

    if len(by_strategy) == 2:
        fixed, structural = by_strategy[FIXED], by_strategy[STRUCTURAL]
        result["difference"] = {
            "chunks": structural["chunks"] - fixed["chunks"],
            "average_chunk_size": round(
                structural["average_chunk_size"] - fixed["average_chunk_size"], 1
            ),
            # Fixed-size repeats the overlap in the next chunk, so its chunks
            # add up to more than the corpus; structural repeats nothing
            # unless a block had to be windowed.
            "stored_characters_vs_corpus": {
                FIXED: round(fixed["total_chunk_characters"] / max(fixed["total_characters"], 1), 3),
                STRUCTURAL: round(
                    structural["total_chunk_characters"] / max(structural["total_characters"], 1), 3
                ),
            },
        }

    return result


def _counted(documents: list[Document]) -> dict:
    counts: dict[str, int] = {}

    for document in documents:
        counts[document.source] = counts.get(document.source, 0) + 1

    return dict(sorted(counts.items()))


def summarise(report: dict) -> None:
    """The comparison, as a person reads it at the end of a run."""
    logger.info("")
    logger.info("Comparison")

    for name, strategy in report["strategies"].items():
        logger.info(
            "  %-11s documents: %d  chunks: %d  avg: %.0f  min: %d  max: %d  %.1fs",
            name + ":",
            strategy["documents"],
            strategy["chunks"],
            strategy["average_chunk_size"],
            strategy["min_chunk_size"],
            strategy["max_chunk_size"],
            strategy["total_seconds"],
        )


async def run(options: argparse.Namespace) -> dict:
    logger.info("Loading documents...")
    documents = load_documents()
    logger.info("Documents loaded: %d (%d characters)", len(documents), sum(d.size for d in documents))

    if not documents:
        raise SystemExit("No documents were found - is data/documents empty?")

    embeddings = EmbeddingService(backend=options.embeddings, model=options.embedding_model)
    logger.info("Embeddings: %s", embeddings.name)

    wanted = {"fixed": [FIXED], "structural": [STRUCTURAL], "both": [FIXED, STRUCTURAL]}[options.strategy]
    reports = [
        await index_strategy(
            documents,
            strategy,
            embeddings,
            Path(options.index_dir),
            options.chunk_size,
            options.overlap,
        )
        for strategy in wanted
    ]

    report = comparison(reports, documents)
    path = Path(options.index_dir) / COMPARISON_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summarise(report)
    logger.info("")
    logger.info("Comparison written: %s", path)
    logger.info("Indexing completed.")

    return report


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.index_documents",
        description="Index the project's documents into a local FAISS index, one per chunking strategy.",
    )
    parser.add_argument("--strategy", choices=("fixed", "structural", "both"), default="both")
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    parser.add_argument(
        "--embeddings",
        choices=(GEMINI, LOCAL),
        default=GEMINI,
        help="gemini uses the project's Gemini integration; local hashes text offline, for runs without a key",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help="the embedding model to ask for; ignored by the local backend",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # faiss announces its own loading at INFO; the pipeline's log is about
    # the pipeline.
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)
    asyncio.run(run(parse(argv)))


if __name__ == "__main__":
    main()
