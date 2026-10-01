"""Day 24: making the model show its working, and checking it.

An answer with a source attached is only worth more than an answer without
one if the source is real. So nothing here trusts the model with the thing
that can be checked:

- **The model never writes a chunk id.** The prompt numbers the extracts
  [1], [2], [3]; the model cites a number, and the backend turns that number
  back into the chunk it stood for. A number that was not in the prompt is
  simply not a chunk, and there is nothing to invent.
- **The model never writes a quote from memory.** Whatever it returns as a
  quote has to appear, character for character, inside the chunk it cited.
  ``quote in chunk.text`` is the whole test, and a citation that fails it is
  dropped and logged rather than shown.

Whitespace is the one liberty taken: a model that reflows a line break into
a space has not invented anything, so both sides are compared with runs of
whitespace collapsed. Nothing else is normalised - not case, not
punctuation, not a word.
"""

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.services.rag_retriever import RetrievedChunk

logger = logging.getLogger(__name__)

#: A quote is meant to be evidence, not a copy of the document. Anything
#: longer than this is truncated to its first sentences before validation,
#: so an over-long quote costs its tail rather than the whole citation.
MAX_QUOTE_CHARS = 320
MAX_QUOTE_SENTENCES = 3
MIN_QUOTE_CHARS = 12

_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# [1], [2] ... as the prompt numbers them, with or without a leading "Source".
_MARKER = re.compile(r"\[(?:source\s*)?(\d{1,2})\]", re.IGNORECASE)


def flatten(text: str) -> str:
    """The comparable form of a piece of text: runs of whitespace collapsed,
    ends trimmed. Used on both sides of every quote check."""
    return _WHITESPACE.sub(" ", text).strip()


def shorten(quote: str, max_sentences: int = MAX_QUOTE_SENTENCES) -> str:
    """A quote cut to its first few sentences, and then to a hard length.

    Cutting happens before validation on purpose: the part that is kept is
    still an exact prefix of what the model returned, so it is still an
    exact fragment of the chunk if the whole was.
    """
    flat = flatten(quote)

    if not flat:
        return ""

    sentences = _SENTENCE_END.split(flat)

    if len(sentences) > max_sentences:
        flat = " ".join(sentences[:max_sentences])

    return flat[:MAX_QUOTE_CHARS].strip()


@dataclass(frozen=True)
class Source:
    """One document behind an answer, as the response schema names it."""

    source: str
    file: str
    section: str
    chunk_id: str

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "file": self.file,
            "section": self.section,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def of(cls, chunk: RetrievedChunk) -> "Source":
        return cls(
            source=chunk.source or "project",
            file=chunk.file,
            section=chunk.section,
            chunk_id=chunk.chunk_id,
        )


@dataclass(frozen=True)
class Citation:
    """One exact fragment of one retrieved chunk."""

    source: str
    section: str
    chunk_id: str
    quote: str

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "quote": self.quote,
        }


@dataclass(frozen=True)
class RejectedCitation:
    """One citation that did not survive validation, and why.

    Kept rather than discarded: a run that throws something away should be
    able to say what and on what grounds, both in the logs and in the
    evaluation file.
    """

    reason: str
    chunk_id: str
    quote: str

    def as_dict(self) -> dict:
        return {"reason": self.reason, "chunk_id": self.chunk_id, "quote": self.quote[:160]}


@dataclass(frozen=True)
class Validated:
    """What came out of one model reply."""

    answer: str
    citations: tuple[Citation, ...]
    rejected: tuple[RejectedCitation, ...]
    sources: tuple[Source, ...]

    @property
    def passed(self) -> bool:
        """Whether every citation the model offered survived."""
        return not self.rejected

    def as_dict(self) -> dict:
        return {
            "citations": [citation.as_dict() for citation in self.citations],
            "rejected_citations": [item.as_dict() for item in self.rejected],
            "sources": [source.as_dict() for source in self.sources],
        }


def parse_reply(reply: str) -> tuple[str, list[dict]]:
    """The answer and the raw citations out of whatever the model replied.

    The prompt asks for one JSON object. Models being what they are, it may
    arrive wrapped in a code fence or with a sentence in front of it, so the
    outermost braces are located rather than assumed. A reply that is not
    JSON at all is not an error: it becomes the answer, with no citations,
    and the validation below will report that there were none.
    """
    text = reply.strip()
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end <= start:
        return text, []

    try:
        document = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("[Citations] the reply was not valid JSON; keeping it as a plain answer")

        return text, []

    if not isinstance(document, dict):
        return text, []

    answer = str(document.get("answer", "")).strip() or text
    raw = document.get("citations")

    return answer, [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _numbered(chunks: Sequence[RetrievedChunk]) -> dict[str, RetrievedChunk]:
    """The map the prompt promised: "1" -> the first chunk, and so on."""
    return {str(number): chunk for number, chunk in enumerate(chunks, start=1)}


def _chunk_for(item: dict, chunks: Sequence[RetrievedChunk]) -> RetrievedChunk | None:
    """Which retrieved chunk a citation points at, if any.

    The number from the prompt decides. A chunk_id is accepted too - it is
    checked against the chunks that were actually retrieved, so it can
    confirm a citation but never conjure one.
    """
    numbered = _numbered(chunks)
    raw = item.get("source") if item.get("source") is not None else item.get("id")
    marker = _MARKER.search(str(raw or ""))
    key = marker.group(1) if marker else str(raw or "").strip()

    if key in numbered:
        return numbered[key]

    wanted = str(item.get("chunk_id", "")).strip()

    if wanted:
        for chunk in chunks:
            if chunk.chunk_id == wanted:
                return chunk

    return None


def validate(reply: str, chunks: Sequence[RetrievedChunk]) -> Validated:
    """The model's reply, with every citation checked against the chunks.

    Six things have to hold, and each failure is named: the citation points
    at a chunk that was retrieved, the chunk exists, the quote is long
    enough to be evidence, and the quote is an exact fragment of that
    chunk's text. Sources are built from the chunks the surviving citations
    point at - never from anything the model wrote - so a source that did
    not support the answer cannot appear.
    """
    answer, raw = parse_reply(reply)
    citations: list[Citation] = []
    rejected: list[RejectedCitation] = []
    used: list[RetrievedChunk] = []

    for item in raw:
        chunk = _chunk_for(item, chunks)
        quote = shorten(str(item.get("quote", "")))

        if chunk is None:
            rejected.append(
                RejectedCitation("unknown source - not among the retrieved chunks",
                                 str(item.get("chunk_id", "") or item.get("source", "")), quote)
            )
            continue

        if len(quote) < MIN_QUOTE_CHARS:
            rejected.append(RejectedCitation("quote too short to be evidence", chunk.chunk_id, quote))
            continue

        if flatten(quote) not in flatten(chunk.text):
            rejected.append(
                RejectedCitation("quote is not a fragment of the cited chunk", chunk.chunk_id, quote)
            )
            continue

        if any(existing.chunk_id == chunk.chunk_id and existing.quote == quote for existing in citations):
            continue

        citations.append(
            Citation(source=chunk.file, section=chunk.section, chunk_id=chunk.chunk_id, quote=quote)
        )

        if all(chunk.chunk_id != seen.chunk_id for seen in used):
            used.append(chunk)

    for item in rejected:
        logger.warning("[Citations] rejected (%s): chunk=%s quote=%r", item.reason, item.chunk_id, item.quote[:80])

    logger.info(
        "[Citations] Sources: %d · Citations: %d · Validation: %s",
        len(used),
        len(citations),
        "passed" if not rejected else f"{len(rejected)} rejected",
    )

    return Validated(
        answer=answer,
        citations=tuple(citations),
        rejected=tuple(rejected),
        sources=tuple(Source.of(chunk) for chunk in used),
    )
