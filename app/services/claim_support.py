"""Day 24: whether the answer actually says what the citations say.

A citation that validates proves one thing only - that the quote is really
in the document. It does not prove the answer follows from it. An answer can
cite a real sentence and then assert something the sentence never said, and
by every check in ``rag_citations`` that answer looks perfect.

So this is a second, smaller question, asked separately: **do the claims in
the answer stand on the quoted evidence?** Two ways of asking it, in order:

1. A short model call that is given the claims and the quotes and nothing
   else - no question, no instructions to be helpful - and answers which
   claims are not supported.
2. When there is no model, or it fails, a lexical check: a claim whose
   distinctive words barely appear in the evidence is marked unsupported.

The lexical check is crude and named for what it is. It is not a judge of
truth; it catches the obvious case where an answer wandered away from the
documents, and says so. Which of the two ran is recorded on every result,
so a verdict is never mistaken for more than it is.
"""

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.services.gemini_service import generate_text

logger = logging.getLogger(__name__)

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
NOT_CHECKED = "not_checked"

BY_MODEL = "model"
BY_WORDS = "lexical"

MAX_CLAIMS = 8
MIN_CLAIM_CHARS = 25
#: Share of a claim's distinctive words that has to appear in the evidence
#: before the lexical check will call it supported.
WORD_OVERLAP_FLOOR = 0.4

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORDS = re.compile(r"[\w/.\-]+", re.UNICODE)
_NOISE = frozenset(
    """
    the a an of to in and or is are was were be been it its this that these those for with from
    on at by as not no if then than can could should would may might must will shall there here
    according section file based provided context document documents
    и в на с по за из от до для о об при это эти тот как что где когда согласно разделе файле
    """.split()
)
# Lines that are structure rather than claim: headings, bullets of sources.
_SKIP = ("sources:", "citations:", "источники:", "цитаты:")

_INSTRUCTIONS = (
    "You are checking whether statements are supported by evidence. You are not answering "
    "anything and you are not being helpful - you are only comparing.\n\n"
    "Evidence (the only thing that counts as support):\n{evidence}\n\n"
    "Statements:\n{claims}\n\n"
    "For each numbered statement, decide whether the evidence above states it or directly "
    "implies it. A statement that is merely plausible, or true in general, is NOT supported "
    "unless the evidence says it.\n\n"
    'Reply with one JSON object and nothing else: {{"unsupported": [numbers]}}. '
    "An empty list means every statement is supported."
)


@dataclass(frozen=True)
class Support:
    """The verdict, and what it was reached from."""

    verdict: str
    checked_by: str
    claims: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return self.verdict == SUPPORTED

    def as_dict(self) -> dict:
        return {
            "citation_support": self.verdict,
            "support_checked_by": self.checked_by,
            "claims_checked": len(self.claims),
            "unsupported_claims": list(self.unsupported),
        }


def claims_in(answer: str, limit: int = MAX_CLAIMS) -> tuple[str, ...]:
    """The sentences of an answer that assert something.

    Deliberately simple: sentences long enough to carry a fact, minus the
    structural lines an answer ends with. Splitting prose into claims is its
    own research problem, and pretending otherwise would make the verdict
    look more precise than it is.
    """
    found: list[str] = []

    for raw in _SENTENCE.split(answer or ""):
        line = raw.strip().lstrip("*-•0123456789. ").strip()

        if len(line) < MIN_CLAIM_CHARS:
            continue

        if any(line.lower().startswith(prefix) for prefix in _SKIP):
            continue

        found.append(line)

        if len(found) >= limit:
            break

    return tuple(found)


def _terms(text: str) -> set[str]:
    return {
        word.lower()
        for word in _WORDS.findall(text)
        if word.lower() not in _NOISE and (len(word) > 3 or any(c.isdigit() for c in word))
    }


def lexically_unsupported(claims: Sequence[str], evidence: str) -> tuple[str, ...]:
    """Claims whose distinctive words are mostly absent from the evidence."""
    known = _terms(evidence)

    if not known:
        return tuple(claims)

    weak: list[str] = []

    for claim in claims:
        wanted = _terms(claim)

        if not wanted:
            continue

        if len(wanted & known) / len(wanted) < WORD_OVERLAP_FLOOR:
            weak.append(claim)

    return tuple(weak)


async def check(answer: str, quotes: Sequence[str], use_model: bool = True) -> Support:
    """Whether the answer's claims stand on the quoted evidence."""
    claims = claims_in(answer)
    evidence = "\n".join(f"- {quote}" for quote in quotes if quote)

    if not claims or not evidence:
        # Nothing to compare. Saying "supported" here would be the one
        # dishonest answer available.
        return Support(verdict=NOT_CHECKED, checked_by="", claims=claims)

    if use_model:
        verdict = await _ask_model(claims, evidence)

        if verdict is not None:
            unsupported = verdict
            result = Support(
                verdict=SUPPORTED if not unsupported else UNSUPPORTED,
                checked_by=BY_MODEL,
                claims=claims,
                unsupported=unsupported,
            )
            logger.info(
                "[Support] %s by %s · %d claim(s), %d unsupported",
                result.verdict,
                result.checked_by,
                len(claims),
                len(unsupported),
            )

            return result

    unsupported = lexically_unsupported(claims, evidence)
    result = Support(
        verdict=SUPPORTED if not unsupported else UNSUPPORTED,
        checked_by=BY_WORDS,
        claims=claims,
        unsupported=unsupported,
    )
    logger.info(
        "[Support] %s by %s · %d claim(s), %d unsupported",
        result.verdict,
        result.checked_by,
        len(claims),
        len(unsupported),
    )

    return result


async def _ask_model(claims: Sequence[str], evidence: str) -> tuple[str, ...] | None:
    """The claims the model says are unsupported, or None when it could not
    be asked. A failed check is never a failed answer."""
    numbered = "\n".join(f"{number}. {claim}" for number, claim in enumerate(claims, start=1))

    try:
        reply = await generate_text(
            _INSTRUCTIONS.format(evidence=evidence, claims=numbered), temperature=0.0
        )
    except Exception as error:  # noqa: BLE001 - the answer stands either way
        logger.warning("[Support] the check could not be run (%s); falling back to words", error)

        return None

    start = reply.find("{")
    end = reply.rfind("}")

    if start == -1 or end <= start:
        return None

    try:
        document = json.loads(reply[start : end + 1])
        numbers = document.get("unsupported") or []
    except (json.JSONDecodeError, AttributeError):
        return None

    picked: list[str] = []

    for value in numbers if isinstance(numbers, list) else []:
        try:
            index = int(value) - 1
        except (TypeError, ValueError):
            continue

        if 0 <= index < len(claims):
            picked.append(claims[index])

    return tuple(picked)
