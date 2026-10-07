"""Day 23: asking the index a better question than the one that was asked.

A person asks "Как работает MCP в проекте?". That sentence is mostly
grammar: the words that carry meaning are one, and half the vector is spent
on "как", "работает" and "в". The chunks worth finding say "MCP server",
"routing table", "discovery", "orchestration" - so the search goes better if
it is given those words instead of the question's punctuation.

The one rule that matters here: **the rewrite is for the search only.** The
model is always shown the question the person actually asked, because a
rewritten query is a guess about what to look for, not a restatement of what
was wanted, and answering the guess instead of the question is how a RAG
system quietly starts answering something else.

The rewrite is a model call with a deliberately narrow brief. When there is
no model, or it returns something useless, a plain keyword fallback runs
instead: drop the stopwords, keep the rest. That is a worse query than the
model's, and better than nothing - and either way ``used`` says which one it
was, so a result is never mistaken for something it is not.
"""

import logging
import re
import time
from dataclasses import dataclass

from app.services.gemini_service import generate_text
from app.services.llm_provider import ProviderUnavailable

logger = logging.getLogger(__name__)

MODEL = "model"
FALLBACK = "keywords"
NONE = "off"

MAX_QUERY_CHARS = 240

_INSTRUCTIONS = (
    "You turn a question about a software project into a search query for a vector index "
    "over that project's own code and documentation.\n\n"
    "Rules:\n"
    "- Answer with the search query and nothing else: no explanation, no quotes, no label.\n"
    "- Keep the subject of the question exactly. Never widen it, narrow it or answer it.\n"
    "- Prefer the words the documents would use: identifiers, file names, technical terms.\n"
    "- Add the obvious near-synonyms of the subject, comma-separated. Six terms is plenty.\n"
    "- Keep the question's own language for ordinary words; keep identifiers as they are written.\n\n"
    "Question:\n{question}\n\nSearch query:"
)

_WORDS = re.compile(r"[\w/.\-]+", re.UNICODE)
# Words that carry no subject: they would only blur the query's vector.
_STOPWORDS = frozenset(
    """
    what which who whom whose when where why how does do did is are was were be been being
    the a an of to in on at by for with from into about over under and or not no if then than
    that this these those it its as can could should would may might must will shall
    project use uses used using get gets got make makes made work works working happen happens
    что какой какая какие какое кто когда где почему как зачем чем чему чего
    и в на с со по за из от до для о об при это этот эта эти тот та те же ли бы не ни
    делает делают работает работают используется используют есть быть был была было
    """.split()
)


@dataclass(frozen=True)
class RewrittenQuery:
    """What the search will actually be given, and where it came from."""

    original: str
    query: str
    used: str
    seconds: float = 0.0

    @property
    def rewritten(self) -> bool:
        return self.used != NONE and self.query != self.original

    def as_dict(self) -> dict:
        return {
            "original_query": self.original,
            "rewritten_query": self.query,
            "rewrite_used": self.used,
            "rewrite_seconds": round(self.seconds, 3),
        }


def keyword_query(question: str) -> str:
    """The fallback: the question with its grammar taken out.

    Not clever - it keeps the words a stopword list does not know - but it
    is deterministic, costs nothing and never invents a term that was not
    in the question.
    """
    words = [word for word in _WORDS.findall(question) if word.lower() not in _STOPWORDS]
    kept = list(dict.fromkeys(words))

    return " ".join(kept) if kept else question.strip()


def clean(text: str) -> str:
    """One line of query out of whatever the model replied with."""
    first = ""

    for line in text.strip().splitlines():
        stripped = line.strip().strip("`").strip()

        if stripped:
            first = stripped
            break

    # Models like to answer a request for X with "X: ...". Drop that.
    first = re.sub(r"^(search\s+query|query|запрос)\s*[:\-]\s*", "", first, flags=re.IGNORECASE)

    return first.strip().strip('"').strip()[:MAX_QUERY_CHARS].strip()


class QueryRewriter:
    """Question in, search query out."""

    def __init__(self, enabled: bool = True, provider=None) -> None:
        self._enabled = enabled
        #: Day 28. ``None`` keeps the Gemini call this class has made since
        #: Day 23, untouched; a provider sends the rewrite to whichever
        #: model is answering instead. The rewrite is a model call like any
        #: other, so leaving it in the cloud while the endpoint said
        #: "local" would have been a quiet lie.
        self._provider = provider

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def rewrite(self, question: str, enabled: bool | None = None) -> RewrittenQuery:
        """The query to search with. Falls back rather than failing: a
        rewrite that does not work is not a reason to answer nothing."""
        text = question.strip()

        if not text:
            raise ValueError("there is no question to rewrite")

        on = self._enabled if enabled is None else enabled

        if not on:
            return RewrittenQuery(original=text, query=text, used=NONE)

        started = time.perf_counter()

        try:
            instructions = _INSTRUCTIONS.format(question=text)

            if self._provider is None:
                generated = await generate_text(instructions, temperature=0.0)
            else:
                generated = (await self._provider.generate(instructions, temperature=0.0)).text

            query = clean(generated)
        except ProviderUnavailable:
            # Day 28. Everything else here is survivable - a rewrite that
            # does not work is not a reason to answer nothing. But a model
            # that cannot be reached at all will not write the answer
            # either, and degrading to keywords would turn "your local
            # model is off" into "I don't know based on the indexed
            # documents", which sends the person looking in the wrong place.
            raise
        except Exception as error:  # noqa: BLE001 - a failed rewrite must not stop the answer
            logger.warning("Query rewrite failed (%s); falling back to keywords", error)
            query = ""

        seconds = time.perf_counter() - started

        if not _usable(query, text):
            fallback = keyword_query(text)
            logger.info("Query rewrite: %r -> %r (keywords)", text[:60], fallback[:60])

            return RewrittenQuery(original=text, query=fallback, used=FALLBACK, seconds=seconds)

        logger.info("Query rewrite: %r -> %r (model)", text[:60], query[:60])

        return RewrittenQuery(original=text, query=query, used=MODEL, seconds=seconds)


def _usable(query: str, question: str) -> bool:
    """A rewrite is usable when it is a query and not an answer.

    Two things are refused: nothing at all, and a paragraph - a model that
    starts explaining has stopped writing a search query, and searching with
    its explanation would drag the vector somewhere the question never went.
    """
    if not query or len(query) < 3:
        return False

    return len(query) <= MAX_QUERY_CHARS and len(query) <= max(len(question) * 3, 120)
