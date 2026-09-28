"""Day 21: reading the project's own documents, whatever they are written in.

One loader for every kind of source the corpus holds - Markdown, plain text,
Python, Kotlin, Java, JSON and PDF - because everything downstream
(chunking, embedding, the index) only wants text plus a few facts about
where it came from. A PDF is turned into text here, before anything is
chunked, so no later step has to know that PDFs exist.

The corpus is the project itself. Its code and its README are read from the
repository, where they are always current, rather than copied somewhere; the
files that do not live in this repository at all - the Android sources, the
overview PDFs - sit in data/documents. Nothing is written for the sake of
having something to index.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

SUPPORTED_SUFFIXES = (".md", ".txt", ".py", ".kt", ".java", ".json", ".pdf")

# Directories that are never worth indexing: build output, caches and the
# index this pipeline writes.
_SKIPPED_PARTS = frozenset({"__pycache__", ".git", ".venv", "build", "node_modules", "index"})


class DocumentError(RuntimeError):
    """A file that should have been readable was not."""


@dataclass(frozen=True)
class Document:
    """One source document, as everything downstream sees it."""

    source: str
    file: str
    title: str
    text: str

    @property
    def size(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class CorpusRoot:
    """Where a group of documents lives, and what to call that group."""

    source: str
    path: str
    suffixes: tuple[str, ...] = SUPPORTED_SUFFIXES


# What gets indexed. Real project material, in the four groups it naturally
# falls into; data/documents is last because it is the one anybody can add to.
CORPUS: tuple[CorpusRoot, ...] = (
    CorpusRoot("project", "README.md"),
    CorpusRoot("backend", "app", (".py",)),
    CorpusRoot("mcp", "mcp_servers", (".py",)),
    CorpusRoot("documents", "data/documents"),
)


def load_documents(
    roots: tuple[CorpusRoot, ...] = CORPUS,
    project_root: Path = PROJECT_ROOT,
) -> list[Document]:
    """Every readable document under the given roots, in a stable order.

    A file that cannot be read is logged and skipped rather than failing the
    run: one broken PDF should not cost the whole index.
    """
    documents: list[Document] = []

    for root in roots:
        location = project_root / root.path

        if not location.exists():
            logger.warning("Corpus root %s does not exist; skipping", root.path)
            continue

        for path in _files_in(location, root.suffixes):
            try:
                document = load_file(path, root.source, project_root)
            except DocumentError as error:
                logger.warning("Skipping %s: %s", path, error)
                continue

            if document.text.strip():
                documents.append(document)

    return documents


def _files_in(location: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if location.is_file():
        return [location] if location.suffix in suffixes else []

    return sorted(
        path
        for path in location.rglob("*")
        if path.is_file()
        and path.suffix in suffixes
        and not _SKIPPED_PARTS.intersection(path.parts)
    )


def load_file(path: Path, source: str = "documents", project_root: Path = PROJECT_ROOT) -> Document:
    """Read one file into a Document. Raises DocumentError when it cannot."""
    text = extract_text(path)

    try:
        name = str(path.resolve().relative_to(project_root))
    except ValueError:
        name = path.name

    return Document(source=source, file=name, title=title_of(path, text), text=text)


def extract_text(path: Path) -> str:
    """The file's text, whatever it is stored as."""
    if path.suffix == ".pdf":
        return extract_pdf_text(path)

    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DocumentError(f"could not be read as text: {error}") from error


def extract_pdf_text(path: Path) -> str:
    """A PDF turned into text, page by page, before anything is chunked."""
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise DocumentError("pypdf is not installed, so PDFs cannot be read") from error

    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as error:  # noqa: BLE001 - pypdf raises many types
        raise DocumentError(f"PDF could not be read: {error}") from error

    # One blank line between pages: the paragraph splitter treats a page
    # break like any other break rather than gluing two pages into one block.
    return "\n\n".join(page.strip() for page in pages if page.strip())


def pdf_title(path: Path) -> str:
    """The title a PDF declares about itself, if it declares one."""
    try:
        from pypdf import PdfReader

        title = (PdfReader(str(path)).metadata or {}).get("/Title")
    except Exception:  # noqa: BLE001 - a missing title is not an error
        return ""

    return str(title).strip() if title else ""


def title_of(path: Path, text: str) -> str:
    """What the document calls itself, or its file name when it says nothing.

    Markdown has a first heading, a Python module has a docstring, a JSON
    file may have a name or a title in it - all of them are better than a
    path, and all of them are the document's own words.
    """
    if path.suffix == ".pdf":
        declared = pdf_title(path)

        if declared:
            return declared

    if path.suffix == ".md":
        for line in text.splitlines():
            if line.startswith("# "):
                return line[2:].strip()

    if path.suffix == ".py":
        docstring = _first_docstring_line(text)

        if docstring:
            return f"{path.name} - {docstring}"

    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict):
            for key in ("title", "name"):
                if isinstance(data.get(key), str) and data[key].strip():
                    return data[key].strip()

    return path.name


def _first_docstring_line(text: str) -> str:
    stripped = text.lstrip()

    for quotes in ('"""', "'''"):
        if stripped.startswith(quotes):
            body = stripped[len(quotes):]
            first = body.split("\n", 1)[0].strip()

            return first.removesuffix(quotes).strip()

    return ""
