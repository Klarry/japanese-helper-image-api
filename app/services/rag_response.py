"""Day 24: one shape for every answer.

Three modes, three very different amounts of evidence, one schema - because
a client that has to branch on the shape of a reply ends up guessing, and a
reply that changes shape when things go badly is exactly the reply nobody
checks.

    {
      "answer": "...",
      "sources": [{source, file, section, chunk_id}],
      "citations": [{source, section, chunk_id, quote}],
      "confidence": "high" | "medium" | "low",
      "rag_status": "answered" | "insufficient_context" | "disabled",
      "citation_support": "supported" | "unsupported" | "not_checked"
    }

``disabled`` is RAG_OFF: there was no retrieval, so there is nothing to
cite and the empty lists say so honestly. ``insufficient_context`` is the
one this day exists for: the index was searched, nothing cleared the bar,
and the model was never asked.

Confidence is derived, not guessed. It is a reading of the evidence that
actually survived - how relevant the best chunk was, whether any citation
held up, and whether the claims in the answer are backed by those
citations - and the rule is written out below rather than left to feel.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.services.rag_citations import Citation, Source

ANSWERED = "answered"
INSUFFICIENT = "insufficient_context"
DISABLED = "disabled"

HIGH = "high"
MEDIUM = "medium"
LOW = "low"

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
NOT_CHECKED = "not_checked"

#: What the model is told to say, and what the backend returns on its behalf
#: when there is nothing to reason over.
INSUFFICIENT_ANSWER = (
    "I don't know based on the indexed documents. "
    "Please clarify your question or provide more context."
)
# How far above the answering threshold a question has to score before the
# answer is called "high". Narrower than it looks: these are cosine-based
# scores, where a few hundredths is a real difference.
HIGH_MARGIN = 0.05


def confidence_of(
    status: str,
    best_relevance: float,
    threshold: float,
    citations: Sequence[Citation],
    support: str = NOT_CHECKED,
) -> str:
    """How much the evidence actually carries.

    ``low``     nothing was answered from documents, or nothing was cited,
                or the claims are not backed by what was cited.
    ``high``    comfortably above the bar, cited, and the citations support
                the answer.
    ``medium``  everything in between - cited and above the bar, but either
                only just, or the support check did not run.
    """
    if status != ANSWERED or not citations:
        return LOW

    if support == UNSUPPORTED:
        return LOW

    if best_relevance >= threshold + HIGH_MARGIN and support == SUPPORTED:
        return HIGH

    return MEDIUM


@dataclass
class CitedAnswer:
    """One answer in the shape every caller can rely on."""

    answer: str
    rag_status: str = ANSWERED
    confidence: str = LOW
    citation_support: str = NOT_CHECKED
    sources: list[Source] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    #: Everything that did not survive validation, kept for the record.
    rejected_citations: list = field(default_factory=list)
    best_relevance: float = 0.0
    answer_threshold: float = 0.0
    #: What the support check looked at, when it ran.
    unsupported_claims: list[str] = field(default_factory=list)
    support_checked_by: str = ""

    @property
    def answered(self) -> bool:
        return self.rag_status == ANSWERED

    @property
    def cited(self) -> bool:
        return bool(self.citations)

    def as_dict(self) -> dict:
        """The response schema, exactly as documented above."""
        return {
            "answer": self.answer,
            "sources": [source.as_dict() for source in self.sources],
            "citations": [citation.as_dict() for citation in self.citations],
            "confidence": self.confidence,
            "rag_status": self.rag_status,
            "citation_support": self.citation_support,
        }

    def as_record(self) -> dict:
        """The schema plus what was thrown away and why - for evaluation
        files and logs, never for the client."""
        record = self.as_dict()
        record.update(
            {
                "rejected_citations": [item.as_dict() for item in self.rejected_citations],
                "best_relevance": round(self.best_relevance, 4),
                "answer_threshold": self.answer_threshold,
                "unsupported_claims": self.unsupported_claims,
                "support_checked_by": self.support_checked_by,
            }
        )

        return record


def insufficient(best_relevance: float, threshold: float) -> CitedAnswer:
    """The answer for a question the index cannot support.

    No sources, no citations, low confidence - and the answer says what to
    do about it rather than only that something went wrong.
    """
    return CitedAnswer(
        answer=INSUFFICIENT_ANSWER,
        rag_status=INSUFFICIENT,
        confidence=LOW,
        citation_support=NOT_CHECKED,
        best_relevance=best_relevance,
        answer_threshold=threshold,
    )


def disabled(answer: str) -> CitedAnswer:
    """RAG_OFF in the same shape: an answer, and empty evidence."""
    return CitedAnswer(answer=answer, rag_status=DISABLED, confidence=LOW)
