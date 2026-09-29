"""Day 22: one agent, two modes.

``ask(question, use_rag=False)`` sends the question to the model on its own.
``ask(question, use_rag=True)`` embeds it, searches the local index, puts the
chunks it found in front of the question, and sends that. Same agent, same
model, same call - the mode decides whether there is a retrieval step in
between, which is what makes the two answers comparable.

The answer comes back with what was retrieved, because for a demonstration -
and for debugging - "where did that come from" is as interesting as the
answer itself. Nothing is reported that was not retrieved.
"""

import logging
import time
from dataclasses import dataclass, field

from app.services.gemini_service import generate_text_with_usage
from app.services.rag_prompt import plain_prompt, rag_prompt
from app.services.rag_retriever import DEFAULT_TOP_K, RAGRetriever, Retrieval, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RagAnswer:
    """One answer, and everything that went into it."""

    question: str
    answer: str
    rag_enabled: bool
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    retrieval_seconds: float = 0.0
    llm_seconds: float = 0.0
    tokens_used: int = 0
    top_k: int = 0
    embedding_model: str = ""
    prompt: str = ""

    @property
    def total_seconds(self) -> float:
        return self.retrieval_seconds + self.llm_seconds

    def as_dict(self, with_text: bool = False, with_prompt: bool = False) -> dict:
        record = {
            "question": self.question,
            "answer": self.answer,
            "rag_enabled": self.rag_enabled,
            "retrieved_chunks": [chunk.as_dict(with_text) for chunk in self.retrieved_chunks],
            "sources": self.sources,
            "top_k": self.top_k,
            "embedding_model": self.embedding_model,
            "retrieval_seconds": round(self.retrieval_seconds, 3),
            "llm_seconds": round(self.llm_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "tokens_used": self.tokens_used,
        }

        if with_prompt:
            record["prompt"] = self.prompt

        return record


class RagAgent:
    """Answers a question about the project, with or without the index."""

    def __init__(self, retriever: RAGRetriever | None = None, top_k: int = DEFAULT_TOP_K) -> None:
        self._retriever = retriever or RAGRetriever()
        self._top_k = top_k

    @property
    def retriever(self) -> RAGRetriever:
        return self._retriever

    async def ask(self, question: str, use_rag: bool = True, top_k: int | None = None) -> RagAnswer:
        """The same question, answered with or without retrieval."""
        text = question.strip()

        if not text:
            raise ValueError("there is no question to answer")

        retrieval: Retrieval | None = None
        prompt = plain_prompt(text)

        if use_rag:
            retrieval = await self._retriever.retrieve(text, top_k=top_k or self._top_k)
            prompt = rag_prompt(text, retrieval.chunks)

        logger.info("Asking the model (rag=%s): %r", use_rag, text[:60])
        started = time.perf_counter()
        generated = await generate_text_with_usage(prompt)
        llm_seconds = time.perf_counter() - started

        return RagAnswer(
            question=text,
            answer=generated.text.strip(),
            rag_enabled=use_rag,
            retrieved_chunks=list(retrieval.chunks) if retrieval else [],
            sources=retrieval.sources if retrieval else [],
            retrieval_seconds=retrieval.seconds if retrieval else 0.0,
            llm_seconds=llm_seconds,
            tokens_used=(generated.input_tokens or 0) + (generated.output_tokens or 0),
            top_k=retrieval.top_k if retrieval else 0,
            embedding_model=retrieval.embedding_model if retrieval else "",
            prompt=prompt,
        )
