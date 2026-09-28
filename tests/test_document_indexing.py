"""Day 21: the indexing pipeline, from a file on disk to a searchable index.

The nine things the assignment asks to prove: documents load, a PDF becomes
text, both chunking strategies do what they say, the metadata is complete,
embeddings come out, the FAISS index is built, vector N belongs to
metadata[N], and the comparison report is computed from the run.

The embeddings here are the local backend, so the suite needs no key and no
network; what it checks is the pipeline, not Gemini's numbers. The one thing
checked about the Gemini backend is the request it sends and what it does
with the answer, against a stubbed transport.
"""

import asyncio
import json
from argparse import Namespace
from pathlib import Path

import numpy
import pytest

from app.index_documents import comparison, run
from app.services.chunking import (
    FIXED,
    MAX_STRUCTURAL_CHARS,
    STRUCTURAL,
    Chunk,
    blocks_of,
    chunk_documents,
    fixed_size_chunks,
    structural_chunks,
)
from app.services.document_loader import (
    CorpusRoot,
    Document,
    extract_pdf_text,
    load_documents,
    load_file,
    title_of,
)
from app.services.embedding_service import LOCAL_NAME, EmbeddingService
from app.services.vector_index import VectorIndexError, load_index, save_index, search

# Sections are written at a realistic length on purpose: structural chunking
# joins anything too small to stand on its own, so a toy document with
# one-line sections would come back as a single chunk and prove nothing.
MARKDOWN = """# Project README

The Japanese Helper backend answers questions about Japanese words, keeps a
conversation, and reaches its tools over MCP. This file is the map: what runs
where, what each part is responsible for, and what to run to see it work.

## Architecture

The backend is FastAPI, one process, with the agent and its services inside
it. Requests arrive on /agent/chat, the agent assembles a prompt from the
conversation, the user profile and whatever its tools returned, and Gemini
writes the answer. Nothing about that assembly happens on the device: the
Android app sends a message and draws what comes back.

### MCP

Three servers, one per stage of the chain, each started as its own
subprocess over stdio. The registry asks every one of them what it offers and
builds the routing table from the answers, so a tool name alone is enough to
send a call to the right place. A server that cannot be reached takes only
its own tools with it.

## Deployment

systemd, one unit, on a small VPS. The unit reads its environment from .env,
starts uvicorn behind nginx, and restarts on failure. Deploying is a git pull
and a restart; the index and the conversation files live under data/ and are
not part of the repository.
"""

PYTHON = '''"""A module that does something."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CONSTANT = 3
DEFAULT_PATH = Path("data") / "storage.json"


class Storage:
    """Keeps things, on disk, in the dullest way that works.

    Reads are fail-safe: a missing or broken file reads as empty rather than
    raising, because that is what it means on the first run.
    """

    def __init__(self, path=DEFAULT_PATH):
        self._path = path

    def save(self, value):
        """Write, then move into place, so a reader never sees half a file."""
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value), encoding="utf-8")
        os.replace(temporary, self._path)

        return value

    def load(self):
        if not self._path.exists():
            return {}

        return json.loads(self._path.read_text(encoding="utf-8"))


def helper(argument):
    """A top-level function, here so the chunker has one to find.

    It does nothing interesting, but it is long enough to be its own block,
    which is the point of having it in the fixture at all.
    """
    logger.debug("helper(%r)", argument)

    return argument * 2
'''

KOTLIN = """package com.japanesehelper

import androidx.compose.runtime.Composable

/** What the screen shows. */
class AiAgentViewModel {
    fun send(message: String) {
        repository.chat(message)
    }

    fun clear() {
        history.clear()
    }
}
"""


def document(text: str, file: str, title: str = "Title", source: str = "test") -> Document:
    return Document(source=source, file=file, title=title, text=text)


def local() -> EmbeddingService:
    return EmbeddingService(backend="local")


# --- 1. document loading ----------------------------------------------------


def test_documents_are_loaded_with_their_source_file_title_and_text(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text(MARKDOWN, encoding="utf-8")
    (tmp_path / "docs" / "service.py").write_text(PYTHON, encoding="utf-8")
    (tmp_path / "docs" / "notes.xyz").write_text("not a supported suffix", encoding="utf-8")

    documents = load_documents((CorpusRoot("handbook", "docs"),), project_root=tmp_path)

    assert [d.file for d in documents] == ["docs/guide.md", "docs/service.py"]
    assert {d.source for d in documents} == {"handbook"}
    # the title is the document's own words, not the file name
    assert documents[0].title == "Project README"
    assert documents[1].title == "service.py - A module that does something."
    assert "FastAPI" in documents[0].text


def test_the_real_corpus_is_large_enough_to_be_worth_indexing():
    """The assignment asks for 20-30 pages; a page is ~2000 characters."""
    documents = load_documents()

    assert len(documents) >= 10
    assert sum(document.size for document in documents) > 30 * 2000
    assert {document.source for document in documents} >= {"project", "backend", "mcp"}


# --- 2. PDF text extraction -------------------------------------------------


def test_a_pdf_is_turned_into_text_before_anything_else(tmp_path):
    pdf = next(Path("data/documents").rglob("*.pdf"), None)

    if pdf is None:
        pytest.skip("no PDF in the corpus")

    text = extract_pdf_text(pdf)

    assert len(text) > 500
    assert "\n" in text
    # and it arrives as a document, with a title of its own
    loaded = load_file(pdf, source="documents")
    assert loaded.text == text
    assert loaded.title


def test_an_unreadable_pdf_is_reported_not_silently_empty(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_text("this is not a PDF at all", encoding="utf-8")

    with pytest.raises(Exception):
        extract_pdf_text(broken)


# --- 3. fixed-size chunking -------------------------------------------------


def test_fixed_size_chunks_have_the_size_and_the_overlap_asked_for():
    text = "".join(f"line {number}\n" for number in range(400))
    chunks = fixed_size_chunks(document(text, "long.txt"), chunk_size=1000, overlap=200)

    assert all(chunk.chunk_size <= 1000 for chunk in chunks)
    assert len(chunks) > 3
    # consecutive chunks share exactly the overlap
    assert chunks[0].text[-200:] == chunks[1].text[:200]
    assert all(chunk.strategy == FIXED for chunk in chunks)


def test_fixed_size_chunking_loses_nothing():
    text = "".join(f"paragraph {number} with some words in it.\n\n" for number in range(120))
    chunks = fixed_size_chunks(document(text, "long.txt"), chunk_size=500, overlap=100)

    rebuilt = chunks[0].text

    for chunk in chunks[1:]:
        rebuilt += chunk.text[100:]

    assert rebuilt == text


def test_overlap_must_be_smaller_than_the_chunk():
    with pytest.raises(ValueError):
        fixed_size_chunks(document("text", "a.txt"), chunk_size=100, overlap=100)


# --- 4. structural chunking -------------------------------------------------


def test_markdown_is_cut_at_its_headings_and_keeps_the_trail():
    chunks = structural_chunks(document(MARKDOWN, "README.md"))
    sections = [chunk.section for chunk in chunks]

    assert "Project README > Architecture" in sections
    assert "Project README > Architecture > MCP" in sections
    assert "Project README > Deployment" in sections
    assert all(chunk.strategy == STRUCTURAL for chunk in chunks)


def test_python_is_cut_at_classes_and_functions():
    sections = [block.section for block in blocks_of(document(PYTHON, "service.py"))]

    assert "class Storage" in sections
    assert "function helper" in sections
    assert "module header" in sections


def test_kotlin_is_cut_at_its_declarations():
    sections = [block.section for block in blocks_of(document(KOTLIN, "Screen.kt"))]

    assert "class AiAgentViewModel" in sections
    assert "fun send" in sections
    assert "fun clear" in sections


def test_a_block_too_big_to_embed_falls_back_to_windows():
    huge = "# One heading\n\n" + ("a paragraph that goes on and on. " * 400)
    chunks = structural_chunks(document(huge, "big.md"), chunk_size=1000, overlap=200)

    assert len(chunks) > 1
    assert all(chunk.chunk_size <= MAX_STRUCTURAL_CHARS for chunk in chunks)
    # the fallback keeps the section, so the pieces still read as that section
    assert {chunk.section for chunk in chunks} == {"One heading"}


def test_the_two_strategies_disagree_which_is_the_point():
    documents = [document(MARKDOWN, "README.md"), document(PYTHON, "service.py")]

    fixed = chunk_documents(documents, FIXED)
    structural = chunk_documents(documents, STRUCTURAL)

    assert [c.section for c in fixed] == [""] * len(fixed)
    assert any(chunk.section for chunk in structural)
    assert len(fixed) != len(structural)


# --- 5. metadata ------------------------------------------------------------


def test_every_chunk_carries_the_metadata_the_index_needs():
    chunks = chunk_documents([document(MARKDOWN, "docs/README.md", "Project README")], STRUCTURAL)
    record = chunks[1].as_metadata()

    assert set(record) == {
        "chunk_id",
        "source",
        "file",
        "title",
        "section",
        "strategy",
        "chunk_size",
        "position",
        "text",
    }
    assert record["chunk_id"] == "docs-README_1"
    assert record["file"] == "docs/README.md"
    assert record["title"] == "Project README"
    assert record["chunk_size"] == len(record["text"])
    assert record["position"] == 1


def test_chunk_ids_are_unique_across_the_corpus():
    documents = [document(MARKDOWN, "a/README.md"), document(MARKDOWN, "b/README.md")]

    for strategy in (FIXED, STRUCTURAL):
        ids = [chunk.chunk_id for chunk in chunk_documents(documents, strategy)]
        assert len(ids) == len(set(ids))


def test_positions_run_across_the_whole_run_in_order():
    documents = [document(MARKDOWN, "a.md"), document(PYTHON, "b.py")]
    chunks = chunk_documents(documents, STRUCTURAL)

    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))


# --- 6. embeddings ----------------------------------------------------------


def test_embeddings_come_back_one_per_text_and_all_the_same_size():
    service = local()

    vectors = asyncio.run(service.embed_all(["first text", "second text", "third"]))

    assert len(vectors) == 3
    assert {len(vector) for vector in vectors} == {768}
    assert service.name == LOCAL_NAME


def test_embedding_is_deterministic_and_carries_some_meaning():
    service = local()

    def similarity(left, right):
        first, second = asyncio.run(service.embed_all([left, right]))
        return sum(a * b for a, b in zip(first, second))

    assert asyncio.run(service.embed("same text")) == asyncio.run(service.embed("same text"))
    assert similarity("the digest scheduler", "the digest scheduler runs") > similarity(
        "the digest scheduler", "kanji stroke order"
    )


def test_batching_covers_every_text():
    service = EmbeddingService(backend="local", batch_size=4)
    seen: list[tuple[int, int]] = []

    vectors = asyncio.run(
        service.embed_all([f"text {number}" for number in range(10)], on_batch=lambda d, t: seen.append((d, t)))
    )

    assert len(vectors) == 10
    assert seen == [(4, 10), (8, 10), (10, 10)]


def test_the_gemini_backend_sends_one_batch_request_and_keeps_the_order(monkeypatch):
    """The real backend, without the network: what it asks for and what it
    does with the answer."""
    from app.services import gemini_service

    asked = {}

    async def fake_post(url, payload):
        asked["url"] = url
        asked["payload"] = payload
        return {"embeddings": [{"values": [float(index), 0.5]} for index in range(len(payload["requests"]))]}

    monkeypatch.setattr(gemini_service, "_post", fake_post)

    vectors = asyncio.run(gemini_service.embed_texts(["one", "two", "three"], model="test-embed"))

    assert "models/test-embed:batchEmbedContents" in asked["url"]
    assert [request["content"]["parts"][0]["text"] for request in asked["payload"]["requests"]] == [
        "one",
        "two",
        "three",
    ]
    assert vectors == [[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]]


def test_a_short_gemini_answer_is_an_error_not_a_silent_gap(monkeypatch):
    from fastapi import HTTPException

    from app.services import gemini_service

    async def fake_post(url, payload):
        return {"embeddings": [{"values": [1.0]}]}

    monkeypatch.setattr(gemini_service, "_post", fake_post)

    with pytest.raises(HTTPException):
        asyncio.run(gemini_service.embed_texts(["one", "two"]))


# --- 7-8. the index, and the join between vectors and metadata --------------


def chunks_for_index() -> list[Chunk]:
    return chunk_documents(
        [document(MARKDOWN, "README.md", "Project README"), document(PYTHON, "service.py")],
        STRUCTURAL,
    )


def test_the_index_is_written_with_its_metadata(tmp_path):
    chunks = chunks_for_index()
    vectors = asyncio.run(local().embed_all([chunk.text for chunk in chunks]))

    written = save_index(tmp_path / "structural", vectors, [c.as_metadata() for c in chunks], STRUCTURAL, LOCAL_NAME)

    assert written["count"] == len(chunks)
    assert (tmp_path / "structural" / "index.faiss").exists()
    document_json = json.loads((tmp_path / "structural" / "metadata.json").read_text(encoding="utf-8"))
    assert document_json["strategy"] == STRUCTURAL
    assert document_json["embedding_model"] == LOCAL_NAME
    assert document_json["dimension"] == 768
    assert len(document_json["chunks"]) == len(chunks)


def test_vector_n_belongs_to_metadata_n(tmp_path):
    """The rule the whole index rests on, checked against FAISS itself."""
    service = local()
    chunks = chunks_for_index()
    vectors = asyncio.run(service.embed_all([chunk.text for chunk in chunks]))
    save_index(tmp_path / "structural", vectors, [c.as_metadata() for c in chunks], STRUCTURAL, service.name)

    loaded = load_index(tmp_path / "structural")

    for position in range(len(loaded)):
        stored = loaded.index.reconstruct(position)
        expected = numpy.asarray(asyncio.run(service.embed(loaded.metadata[position]["text"])), dtype="float32")
        # written normalised; the local backend already normalises, so the
        # vector that comes back is the embedding of that chunk's own text
        assert numpy.allclose(stored, expected, atol=1e-5), f"vector {position} is not its metadata"

    assert [record["position"] for record in loaded.metadata] == list(range(len(loaded)))


def test_searching_finds_the_chunk_the_words_came_from(tmp_path):
    service = local()
    chunks = chunks_for_index()
    vectors = asyncio.run(service.embed_all([chunk.text for chunk in chunks]))
    save_index(tmp_path / "structural", vectors, [c.as_metadata() for c in chunks], STRUCTURAL, service.name)

    loaded = load_index(tmp_path / "structural")
    found = search(loaded, asyncio.run(service.embed("systemd, one unit")), limit=1)

    assert found[0]["section"].endswith("Deployment")
    # a real similarity, not a tie: the right chunk, and above nothing
    assert found[0]["score"] > 0.2


def test_an_index_and_metadata_that_disagree_are_refused(tmp_path):
    chunks = chunks_for_index()
    vectors = asyncio.run(local().embed_all([chunk.text for chunk in chunks]))

    with pytest.raises(VectorIndexError):
        save_index(tmp_path / "bad", vectors, [c.as_metadata() for c in chunks][:-1], STRUCTURAL, LOCAL_NAME)

    save_index(tmp_path / "bad", vectors, [c.as_metadata() for c in chunks], STRUCTURAL, LOCAL_NAME)
    path = tmp_path / "bad" / "metadata.json"
    broken = json.loads(path.read_text(encoding="utf-8"))
    broken["chunks"] = broken["chunks"][:-1]
    path.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(VectorIndexError):
        load_index(tmp_path / "bad")


# --- 9. the comparison report -----------------------------------------------


def test_the_report_is_computed_from_the_run_not_written_down(tmp_path):
    options = Namespace(
        strategy="both",
        index_dir=str(tmp_path),
        embeddings="local",
        chunk_size=1000,
        overlap=200,
    )

    report = asyncio.run(run(options))

    fixed = report["strategies"][FIXED]
    structural = report["strategies"][STRUCTURAL]
    assert fixed["documents"] == structural["documents"] == report["corpus"]["documents"]
    assert fixed["embedding_dimension"] == 768
    assert fixed["overlap"] == 200
    assert structural["overlap_applies_to"] == "only blocks longer than the limit"
    assert fixed["min_chunk_size"] <= fixed["average_chunk_size"] <= fixed["max_chunk_size"]
    assert report["difference"]["chunks"] == structural["chunks"] - fixed["chunks"]
    # both indexes are on disk, and the report is beside them
    assert (tmp_path / "fixed_size" / "index.faiss").exists()
    assert (tmp_path / "structural" / "index.faiss").exists()
    saved = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert saved["strategies"][FIXED]["chunks"] == fixed["chunks"]


def test_the_report_matches_what_the_chunker_really_produced():
    documents = [document(MARKDOWN, "README.md"), document(PYTHON, "service.py")]
    chunks = chunk_documents(documents, FIXED)

    from app.index_documents import StrategyReport

    report = StrategyReport(
        strategy=FIXED,
        documents=len(documents),
        total_characters=sum(d.size for d in documents),
        chunks=len(chunks),
        chunk_sizes=[chunk.chunk_size for chunk in chunks],
    ).as_dict()

    sizes = [chunk.chunk_size for chunk in chunks]
    assert report["chunks"] == len(sizes)
    assert report["average_chunk_size"] == round(sum(sizes) / len(sizes), 1)
    assert report["min_chunk_size"] == min(sizes)
    assert report["max_chunk_size"] == max(sizes)
