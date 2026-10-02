"""Day 25: what the chat puts in front of the model.

Four blocks, in the order the assignment names them, and the order matters.
Task memory first, because it holds whatever comes after it: a question asked
on turn twelve is still about the goal settled on turn one. Then the recent
conversation, so pronouns and "that" resolve. Then the documents retrieved
for *this* question. Then the question itself, and the rules, last - so the
rules are the final thing read before answering.

The one judgement this file makes is about questions the documents cannot
answer. A chat gets two kinds: questions about the project ("how is a chunk
id built?"), which only a document can settle, and questions about the
conversation ("what did we decide?"), which only the task memory can. Day 24
refuses the first kind without evidence, and it must keep refusing. But
refusing the second kind would be absurd - the answer is right there in the
memory block, and it was the person who put it there.

So the instructions split by where an answer may come from, not by whether
retrieval happened: facts about the project need a quoted document; what the
conversation itself settled needs the memory block; and the model is told,
in as many words, that it may not use one in place of the other.
"""

from collections.abc import Sequence

from app.services.rag_prompt import DONT_KNOW, numbered_context
from app.services.rag_retriever import RetrievedChunk

_ROLE = (
    "You are the assistant in an ongoing working conversation about one software "
    "project: a FastAPI backend with an AI agent, MCP servers and an Android client."
)

_SHARED_RULES = (
    "- Keep the task's goal. A question about one detail does not replace what this "
    "conversation is for.\n"
    "- Respect every constraint and decision in task memory, and keep using its "
    "confirmed terms rather than renaming things.\n"
    "- Answer in the language of the question.\n"
)

_WITH_DOCUMENTS = (
    "- For anything about the project itself, use ONLY the numbered extracts above. "
    "Do not add facts from general knowledge, however plausible.\n"
    "- Cite by NUMBER only. Never write a file name, a chunk id, or a number that is "
    "not above.\n"
    "- Each quote must be copied EXACTLY from the extract you cite - same words, same "
    "order, one to three short sentences.\n"
    "- For what this conversation has settled - the goal, constraints, decisions, "
    "terms - answer from TASK MEMORY, and cite nothing for it. Task memory is not a "
    "document and must never be quoted as one.\n"
    f'- If neither the extracts nor task memory answer the question, say exactly: "{DONT_KNOW}"\n'
)

_WITHOUT_DOCUMENTS = (
    "- No extract from the project's documents was relevant enough to this question, "
    "so you have none. You may still answer from TASK MEMORY and the conversation "
    "above - that is what they are for.\n"
    "- You may NOT state anything about the project's code, files or behaviour that "
    "is not in task memory or in the conversation. Not having a document is not "
    "permission to remember one.\n"
    f'- If task memory and the conversation do not answer it either, say exactly: "{DONT_KNOW}"\n'
    "- Return an empty citations list. There is nothing to cite.\n"
)

_REPLY_SHAPE = (
    "\nReply with one JSON object and nothing else:\n"
    "{\n"
    '  "answer": "your answer, in prose",\n'
    '  "citations": [{"source": 1, "quote": "exact words from extract 1"}]\n'
    "}"
)


def chat_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
    task_memory: str = "",
    recent_history: str = "",
) -> str:
    """The whole context for one turn, assembled."""
    blocks = [_ROLE, ""]

    if task_memory:
        blocks += [task_memory, ""]

    if recent_history:
        blocks += ["RECENT CONVERSATION (oldest first):", recent_history, ""]

    if chunks:
        blocks += [
            "RETRIEVED DOCUMENTS (found in the project's own index for this question):",
            "",
            numbered_context(chunks),
            "",
        ]
    else:
        blocks += ["RETRIEVED DOCUMENTS: none were relevant enough to this question.", ""]

    blocks += [
        "CURRENT USER QUESTION:",
        question,
        "",
        "Instructions:",
        _SHARED_RULES + (_WITH_DOCUMENTS if chunks else _WITHOUT_DOCUMENTS),
        _REPLY_SHAPE,
    ]

    return "\n".join(blocks)
