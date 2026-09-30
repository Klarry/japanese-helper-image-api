"""Day 22: what the model is shown, in both modes.

Two prompts, deliberately as alike as possible, because the only honest way
to compare an answer with retrieval against an answer without it is to
change one thing. Same question, same framing, same request for sources -
the RAG prompt adds the retrieved context and the rule that comes with it:
answer from this, and if it is not here, say so.

Saying "I don't know" is a correct answer here, not a failure. A question
whose answer is genuinely not in the index has exactly one right response,
and the prompt asks for it in as many words so that the model does not fill
the gap from memory and call it a document.
"""

from collections.abc import Sequence

from app.services.rag_retriever import RetrievedChunk

UNAVAILABLE = "not available in the indexed documents"
# What Enhanced RAG answers when the filter keeps nothing (Day 23). No
# model call: there is no context to reason over, and asking anyway is
# how a confident paragraph gets built out of nothing.
NO_CONTEXT_ANSWER = f"The information is {UNAVAILABLE}."

_SHARED = (
    "You are answering questions about one software project: a FastAPI backend with an AI "
    "agent, MCP servers and an Android client."
)

_WITHOUT_CONTEXT = (
    f"{_SHARED} You have no access to its files - answer from what you know, and say plainly "
    "when you are not sure rather than inventing details."
)

_WITH_CONTEXT = (
    f"{_SHARED} Below are extracts from the project's own documents, found for this question in "
    "a local index of them.\n\n"
    "Rules:\n"
    "- Answer using the context below and nothing else. Do not add facts from general knowledge, "
    "however plausible they look.\n"
    f"- If the context does not answer the question, say clearly that the information is {UNAVAILABLE}. "
    "That is a correct answer, not a failure - do not guess to fill the gap.\n"
    "- Refer to where each statement came from, by file and section, for example: "
    "According to `mcp_servers/jlpt_vocab.py`, section `class LatestDigest`, ...\n"
    "- End with a 'Sources:' list of the extracts you actually used. Never list one you did not use, "
    "and never name a file that is not below.\n"
    "- Answer in the language of the question."
)


def context_block(chunks: Sequence[RetrievedChunk]) -> str:
    """The retrieved extracts, each labelled with where it came from."""
    return "\n\n".join(
        f"[Source: {chunk.file}]\n[Section: {chunk.section or '-'}]\n[Score: {chunk.score:.3f}]\n{chunk.text}"
        for chunk in chunks
    )


def rag_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Question, context, rules - in that order, so the question is read
    before the extracts and the rules are the last thing before answering."""
    if not chunks:
        return (
            f"Question:\n{question}\n\n"
            "Relevant context:\n(nothing was found in the indexed documents)\n\n"
            f"Instructions:\nSay that the information is {UNAVAILABLE}."
        )

    return (
        f"Question:\n{question}\n\n"
        f"Relevant context:\n\n{context_block(chunks)}\n\n"
        f"Instructions:\n{_WITH_CONTEXT}"
    )


def plain_prompt(question: str) -> str:
    """The same question with no context at all - the other half of the
    comparison."""
    return f"Question:\n{question}\n\nInstructions:\n{_WITHOUT_CONTEXT}"
