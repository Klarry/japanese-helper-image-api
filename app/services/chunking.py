"""Day 21: cutting documents into pieces small enough to embed.

Two strategies, deliberately independent, so they can be compared rather
than blended:

* **fixed-size** knows nothing about the document. It walks the text in
  windows of ``chunk_size`` characters that overlap by ``overlap``, so a
  sentence cut in half by one window is whole in the next. Every character
  of the document appears in at least one chunk.

* **structural** reads the document the way its author wrote it: Markdown by
  headings, Python by classes and functions, Kotlin and Java by their
  declarations, everything else by paragraphs. A block is kept whole even
  when it is small, because "### Configuration" and the paragraph under it
  belong together; a block too big to embed usefully falls back to
  fixed-size windows *inside that block*, which keeps its section name.

Both produce the same kind of chunk, with the same metadata, so the index
and everything after it cannot tell which strategy made a chunk except by
the ``strategy`` field.
"""

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from app.services.document_loader import Document

logger = logging.getLogger(__name__)

FIXED = "fixed-size"
STRUCTURAL = "structural"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
# A structural block longer than this is windowed inside itself. Two chunks
# of a section beat one chunk no embedding can represent.
MAX_STRUCTURAL_CHARS = 2000
# ...and one shorter than this is joined to the block after it. A heading on
# its own line, or an interface declaration without its methods, is not
# something anybody wants to find on its own.
MIN_STRUCTURAL_CHARS = 200

_MARKDOWN_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
# A Kotlin/Java declaration worth cutting at: a type at the left margin, or a
# function at any indentation (methods are indented, top-level ones are not).
_JVM_DECLARATION = re.compile(
    r"^(?P<indent>\s*)(?:@\w+[^\n]*\s+)*"
    r"(?:public |private |protected |internal |open |final |static |abstract |sealed |data |inner |suspend |override )*"
    r"(?P<kind>class|object|interface|enum class|fun|void|record)\s+(?P<name>[A-Za-z_][\w<>]*)"
)


@dataclass(frozen=True)
class Chunk:
    """One piece of one document, with everything needed to find it again."""

    chunk_id: str
    source: str
    file: str
    title: str
    section: str
    strategy: str
    text: str
    position: int

    @property
    def chunk_size(self) -> int:
        return len(self.text)

    def as_metadata(self) -> dict:
        """The record stored next to the vector, at the same position."""
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "file": self.file,
            "title": self.title,
            "section": self.section,
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "position": self.position,
            "text": self.text,
        }


def chunk_documents(
    documents: list[Document],
    strategy: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Every document cut by one strategy, numbered across the whole run."""
    chunker = fixed_size_chunks if strategy == FIXED else structural_chunks
    chunks: list[Chunk] = []

    for document in documents:
        chunks.extend(chunker(document, chunk_size=chunk_size, overlap=overlap, start=len(chunks)))

    return chunks


# --- strategy A: fixed size -------------------------------------------------


def fixed_size_chunks(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    start: int = 0,
) -> list[Chunk]:
    """Windows of ``chunk_size`` characters, each overlapping the last by
    ``overlap`` - so nothing falls between two chunks."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    pieces = _windows(document.text, chunk_size, overlap)

    return [
        _chunk(document, text, FIXED, section="", position=start + offset, index=offset)
        for offset, text in enumerate(pieces)
    ]


def _windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = chunk_size - overlap
    pieces = []

    for begin in range(0, max(len(text), 1), step):
        piece = text[begin : begin + chunk_size]

        if piece.strip():
            pieces.append(piece)

        if begin + chunk_size >= len(text):
            break

    return pieces


# --- strategy B: structural -------------------------------------------------


@dataclass(frozen=True)
class Block:
    """A piece of a document its author would recognise: a section, a class,
    a function, a paragraph."""

    section: str
    text: str


def structural_chunks(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    start: int = 0,
) -> list[Chunk]:
    """The document cut where it is already divided, with oversized blocks
    windowed inside themselves."""
    chunks: list[Chunk] = []

    for block in coalesce(blocks_of(document)):
        if not block.text.strip():
            continue

        if len(block.text) <= MAX_STRUCTURAL_CHARS:
            pieces = [block.text]
        else:
            # Fallback, and it says so: the section name is kept, so a long
            # class still reads as that class.
            pieces = _windows(block.text, chunk_size, overlap)
            logger.debug("%s: block %r windowed into %d", document.file, block.section, len(pieces))

        for piece in pieces:
            chunks.append(
                _chunk(
                    document,
                    piece,
                    STRUCTURAL,
                    section=block.section,
                    position=start + len(chunks),
                    index=len(chunks),
                )
            )

    return chunks


def coalesce(blocks: list[Block], minimum: int = MIN_STRUCTURAL_CHARS) -> list[Block]:
    """Join a block too small to stand alone to the one after it.

    Structure is not always the size of a thought: a heading, a one-line
    interface, an import block. Joining forward keeps the reading order and
    gives the merged piece the name of the first thing in it - which is the
    heading or the declaration the rest belongs to.
    """
    merged: list[Block] = []
    pending: Block | None = None

    for block in blocks:
        if pending is not None:
            block = Block(pending.section, pending.text + block.text)
            pending = None

        if len(block.text.strip()) < minimum:
            pending = block
            continue

        merged.append(block)

    if pending is not None:
        if merged:
            last = merged[-1]
            merged[-1] = Block(last.section, last.text + pending.text)
        else:
            merged.append(pending)

    return merged


def blocks_of(document: Document) -> list[Block]:
    """How this particular document is divided, by what it is."""
    suffix = PurePosixPath(document.file).suffix

    if suffix == ".md":
        return _markdown_blocks(document.text)

    if suffix == ".py":
        return _python_blocks(document.text)

    if suffix in (".kt", ".java"):
        return _jvm_blocks(document.text)

    return _paragraph_blocks(document.text)


def _markdown_blocks(text: str) -> list[Block]:
    """One block per heading, named by the trail that leads to it -
    "Deployment > Configuration" rather than just "Configuration"."""
    blocks: list[Block] = []
    trail: list[str] = []
    section = ""
    current: list[str] = []

    for line in text.splitlines(keepends=True):
        heading = _MARKDOWN_HEADING.match(line.rstrip("\n"))

        if heading is None:
            current.append(line)
            continue

        if current:
            blocks.append(Block(section, "".join(current)))
            current = []

        level = len(heading.group(1))
        trail = trail[: level - 1] + [heading.group(2)]
        section = " > ".join(trail)
        current.append(line)

    if current:
        blocks.append(Block(section, "".join(current)))

    return blocks


def _python_blocks(text: str) -> list[Block]:
    """The module header, then one block per top-level class or function,
    read with Python's own parser rather than a regular expression."""
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        logger.debug("Falling back to paragraphs, unparsable Python: %s", error)
        return _paragraph_blocks(text)

    lines = text.splitlines(keepends=True)
    tops = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    blocks: list[Block] = []
    cursor = 0

    for node in tops:
        begin = min((decorator.lineno for decorator in node.decorator_list), default=node.lineno) - 1
        end = (node.end_lineno or node.lineno)

        if begin > cursor:
            blocks.append(Block("module header", "".join(lines[cursor:begin])))

        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        blocks.append(Block(f"{kind} {node.name}", "".join(lines[begin:end])))
        cursor = end

    if cursor < len(lines):
        blocks.append(Block("module header" if not blocks else "module footer", "".join(lines[cursor:])))

    return blocks


def _jvm_blocks(text: str) -> list[Block]:
    """Kotlin and Java, cut at declarations: a type at the left margin, a
    function wherever it is written."""
    lines = text.splitlines(keepends=True)
    starts: list[tuple[int, str]] = []

    for number, line in enumerate(lines):
        found = _JVM_DECLARATION.match(line)

        if found is None:
            continue

        kind, name = found.group("kind"), found.group("name")

        if kind != "fun" and found.group("indent"):
            continue  # a nested type: it belongs to the block it is written in

        starts.append((number, f"{kind} {name}"))

    if not starts:
        return _paragraph_blocks(text)

    blocks: list[Block] = []

    if starts[0][0] > 0:
        blocks.append(Block("file header", "".join(lines[: starts[0][0]])))

    for index, (number, section) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(lines)
        blocks.append(Block(section, "".join(lines[number:end])))

    return blocks


def _paragraph_blocks(text: str, target: int = DEFAULT_CHUNK_SIZE) -> list[Block]:
    """Paragraphs, gathered until they are worth embedding. Used for plain
    text, JSON and PDFs, where a blank line is the only structure there is."""
    paragraphs = [piece for piece in re.split(r"\n\s*\n", text) if piece.strip()]
    blocks: list[Block] = []
    current: list[str] = []
    length = 0

    for paragraph in paragraphs:
        current.append(paragraph)
        length += len(paragraph)

        if length >= target:
            blocks.append(Block("paragraphs", "\n\n".join(current)))
            current, length = [], 0

    if current:
        blocks.append(Block("paragraphs", "\n\n".join(current)))

    return blocks


# --- shared -----------------------------------------------------------------


def _chunk(document: Document, text: str, strategy: str, section: str, position: int, index: int) -> Chunk:
    return Chunk(
        chunk_id=f"{file_slug(document.file)}_{index}",
        source=document.source,
        file=document.file,
        title=document.title,
        section=section,
        strategy=strategy,
        text=text,
        position=position,
    )


def file_slug(file: str) -> str:
    """"README.md" -> "README"; "app/services/digest.py" -> "app-services-digest".

    Unique per file, so chunk ids are unique across the corpus, and still
    readable in a listing.
    """
    path = PurePosixPath(file)

    return "-".join([*path.parent.parts, path.stem]).strip("-") or path.stem
