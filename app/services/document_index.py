"""Day 21: reading back what the indexer wrote, for anyone who only wants
to know whether an index exists and how big it is.

The indexing itself is a command you run (``python -m app.index_documents``).
This module is the other half of that: a small, read-only view of the report
it leaves behind, so the API - and through it the screen - can say "62
documents, 546 fixed chunks, 623 structural chunks, 768 dimensions" without
importing FAISS, loading a vector, or knowing what chunking is.

Nothing here builds anything. A missing report is not an error: it means
the index has not been built yet, which is a perfectly good answer.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.services.chunking import FIXED, STRUCTURAL
from app.services.document_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

COMPARISON_PATH = PROJECT_ROOT / "data" / "index" / "comparison.json"


@dataclass(frozen=True)
class IndexSummary:
    """What was indexed, as a few numbers."""

    found: bool = False
    documents: int = 0
    total_characters: int = 0
    fixed_chunks: int = 0
    structural_chunks: int = 0
    embedding_model: str = ""
    embedding_dimension: int = 0
    built_at: str = ""

    def as_dict(self) -> dict:
        return {
            "found": self.found,
            "documents": self.documents,
            "total_characters": self.total_characters,
            "fixed_chunks": self.fixed_chunks,
            "structural_chunks": self.structural_chunks,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "built_at": self.built_at,
        }


def read_summary(path: Path = COMPARISON_PATH) -> IndexSummary:
    """The comparison report as a handful of numbers, or found=False."""
    if not path.exists():
        return IndexSummary()

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Could not read the index report %s: %s", path, error)
        return IndexSummary()

    if not isinstance(report, dict):
        return IndexSummary()

    corpus = report.get("corpus") or {}
    strategies = report.get("strategies") or {}
    fixed = strategies.get(FIXED) or {}
    structural = strategies.get(STRUCTURAL) or {}
    built = fixed or structural

    return IndexSummary(
        found=bool(built),
        documents=int(corpus.get("documents") or 0),
        total_characters=int(corpus.get("total_characters") or 0),
        fixed_chunks=int(fixed.get("chunks") or 0),
        structural_chunks=int(structural.get("chunks") or 0),
        embedding_model=str(built.get("embedding_model") or ""),
        embedding_dimension=int(built.get("embedding_dimension") or 0),
        built_at=str(report.get("created_at") or ""),
    )
