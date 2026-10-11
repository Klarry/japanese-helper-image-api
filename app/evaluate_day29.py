"""Day 29: does tuning the local model make it better at this job?

    python -m app.evaluate_day29
    python -m app.evaluate_day29 --profiles optimized-params optimized
    python -m app.evaluate_day29 --variants qwen3:4b-instruct-2507-q4_K_M
    python -m app.evaluate_day29 --only 1 6 13 --repeats 2 --merge

Every profile answers the same fourteen questions from the same retrieved
context. Retrieval runs once per question, before any model is involved,
with the query rewrite off - so what changes between profiles is how the
model is asked (options, prompt) and nothing else. Each answer is scored
against the rubric in ``data/evaluation/day29_questions.json``.

Speed is measured twice. Over the question set (one warm run per question)
and in a benchmark on one fixed prompt: the model is unloaded, the first
request pays the load and is reported as cold, then ``--repeats`` warm
requests are timed and their median reported. Memory is read from Ollama
(``/api/ps``) and, where ``ps`` exists, sampled from the Ollama processes
while a request runs.

Nothing is estimated. A metric the machine does not report is written as
null, not filled in; a variant that is not installed is skipped and said
so - this script never downloads a model.
"""

import argparse
import asyncio
import json
import logging
import platform
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.services import ollama_profiles, ollama_service
from app.services.document_loader import PROJECT_ROOT
from app.services.llm_provider import OllamaProvider, ProviderUnavailable
from app.services.ollama_profiles import PROFILES, Profile
from app.services.rag_agent import ENHANCED, RagAgent
from app.services.rag_citations import flatten
from app.services.rag_enhanced import EnhancedRetrieval, EnhancedRetriever
from app.services.rag_prompt import cited_prompt_for
from app.services.rag_response import INSUFFICIENT
from app.services.rag_retriever import DEFAULT_STRATEGY, RAGRetriever
from app.services.rag_settings import DEFAULT_SETTINGS, RagSettings

logger = logging.getLogger("evaluate-day29")

QUESTIONS_FILE = PROJECT_ROOT / "data" / "evaluation" / "day29_questions.json"
RESULTS_FILE = PROJECT_ROOT / "data" / "evaluation" / "day29_results.json"
SUMMARY_FILE = PROJECT_ROOT / "data" / "evaluation" / "day29_summary.txt"
DEFAULT_PROFILES = list(PROFILES)
RULE = "=" * 78

#: A refusal, in the words models actually use. Applied to every answer:
#: on an unanswerable question it is the right reply, on an answerable one
#: it is a false refusal.
REFUSAL = re.compile(
    r"don'?t know|do not know|not (?:available|provided|mentioned|specified|described|stated|"
    r"covered|documented|included|present|found) in|(?:do|does) not (?:provide|contain|mention|"
    r"specify|describe|say|state|include|cover|document)|insufficient|not enough information|"
    r"no information|cannot (?:be )?(?:determined|answered)|unable to (?:determine|answer)",
    re.IGNORECASE,
)

#: What looks like a technical name: `backticked`, snake_case, CamelCase,
#: a file path or an endpoint. Used to find names an answer uses that the
#: context it was given does not contain - a heuristic, reported as one.
_IDENTIFIER = re.compile(
    r"`([^`\n]{2,80})`"
    r"|\b([\w./-]+\.(?:py|kt|json|md))\b"
    r"|(?<![\w])(/[a-z][\w-]*(?:/[\w{}-]+)+)"
    r"|\b([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+)\b"
    r"|\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b"
)


# --- questions and scoring ---------------------------------------------------


@dataclass(frozen=True)
class Fact:
    name: str
    patterns: tuple[re.Pattern, ...]

    def found_in(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.patterns)


@dataclass(frozen=True)
class Question:
    id: int
    category: str
    kind: str
    question: str
    expected: str
    expected_sources: tuple[str, ...] = ()
    rubric: tuple[Fact, ...] = ()
    accept: tuple[re.Pattern, ...] = ()
    forbidden: tuple[re.Pattern, ...] = ()

    @property
    def answerable(self) -> bool:
        return self.kind == "answer"


def _compiled(patterns) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in patterns or ())


def load_questions(path: Path = QUESTIONS_FILE) -> tuple[list[Question], int]:
    """The control set and the id of the benchmark question."""
    if not path.exists():
        raise SystemExit(f"{path} is missing - the Day 29 questions are part of the repository.")

    document = json.loads(path.read_text(encoding="utf-8"))
    questions = []

    for raw in document.get("questions", []):
        kind = raw.get("kind", "answer")

        if kind not in {"answer", "insufficient"}:
            raise SystemExit(f"question {raw.get('id')}: unknown kind {kind!r}")

        rubric = tuple(Fact(item["fact"], _compiled(item["any"])) for item in raw.get("rubric", []))

        if kind == "answer" and not rubric:
            raise SystemExit(f"question {raw['id']} is answerable but has no rubric")

        questions.append(
            Question(
                id=int(raw["id"]),
                category=str(raw.get("category", "")),
                kind=kind,
                question=str(raw["question"]),
                expected=str(raw.get("expected", "")),
                expected_sources=tuple(raw.get("expected_sources", [])),
                rubric=rubric,
                accept=_compiled(raw.get("accept")),
                forbidden=_compiled(raw.get("forbidden")),
            )
        )

    if not questions:
        raise SystemExit(f"{path} holds no questions")

    return questions, int(document.get("bench_question", questions[0].id))


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL.search(answer or ""))


def score(question: Question, answer: str, rag_status: str = "") -> dict:
    """How right one answer is, against the question's own rubric.

    Answerable: the share of rubric facts the answer states (0..1).
    Unanswerable: 1 when it refuses - or says plainly that the thing asked
    about does not exist - without stating anything the forbidden patterns
    catch; 0 otherwise. A refusal by the relevance gate counts as a refusal.
    """
    text = answer or ""
    refused = rag_status == INSUFFICIENT or is_refusal(text)

    if question.answerable:
        matched = [fact.name for fact in question.rubric if fact.found_in(text)]
        missing = [fact.name for fact in question.rubric if fact.name not in matched]
        value = len(matched) / len(question.rubric)

        return {
            "score": round(value, 3),
            "correct": value == 1.0,
            "matched": matched,
            "missing": missing,
            "refused": refused,
            "forbidden": [],
        }

    forbidden = [pattern.pattern for pattern in question.forbidden if pattern.search(text)]
    accepted = refused or any(pattern.search(text) for pattern in question.accept)
    right = accepted and not forbidden

    return {
        "score": 1.0 if right else 0.0,
        "correct": right,
        "matched": [],
        "missing": [],
        "refused": refused,
        "forbidden": forbidden,
    }


def identifiers(text: str) -> set[str]:
    found = set()

    for match in _IDENTIFIER.finditer(text or ""):
        value = next(group for group in match.groups() if group)
        found.add(value.strip().strip(".,;:()'\""))

    return {value for value in found if len(value) > 2}


def unsupported_identifiers(answer: str, context: str, question: str) -> list[str]:
    """Technical-looking names in the answer that appear neither in the
    context it was given nor in the question. Not proof of invention - a
    model may legitimately rephrase - but a list worth reading."""
    haystack = f"{context}\n{question}".lower()

    return sorted(name for name in identifiers(answer) if name.lower() not in haystack)


def format_problem(answer: str) -> bool:
    """Whether the reply's structure leaked into the answer text - the
    Day 28 failure: citations or a whole JSON object written as prose."""
    text = (answer or "").strip()

    return (
        text.startswith("{")
        or '"citations"' in text
        or '\\"' in text
        or bool(re.search(r'\{\s*"source"', text))
    )


# --- the pieces of one run ---------------------------------------------------


class FrozenRetrieval:
    """An EnhancedRetriever that hands back one retrieval it was given.

    How every profile gets the same context: the question was retrieved
    once, before any profile ran, and this stands in for the retriever so
    the agent's own pipeline - gate, prompt, validation, support check -
    runs unchanged on top of it.
    """

    def __init__(self, found: EnhancedRetrieval) -> None:
        self._found = found

    async def retrieve(self, question: str, settings: RagSettings | None = None) -> EnhancedRetrieval:
        return self._found


class RecordingProvider:
    """The Ollama provider, with every call it makes written down.

    The agent makes the answer call first and the support check second;
    the rewrite is off, so there is no third. Recording here rather than
    inside the agent keeps the agent the code that runs in production.
    """

    def __init__(self, inner: OllamaProvider) -> None:
        self._inner = inner
        self.name = inner.name
        self.model = inner.model
        self.profile = inner.profile
        self.calls: list[dict] = []

    @property
    def prompt_version(self) -> str:
        return self._inner.prompt_version

    async def generate(self, prompt: str, temperature: float | None = None):
        started = time.perf_counter()

        try:
            produced = await self._inner.generate(prompt, temperature=temperature)
        except ProviderUnavailable as error:
            self.calls.append(
                {"wall_ms": _ms(time.perf_counter() - started), "error": error.reason, "meta": None}
            )
            raise

        self.calls.append(
            {"wall_ms": _ms(time.perf_counter() - started), "error": "", "meta": produced.meta or {}}
        )

        return produced


class MemorySampler:
    """Peak resident memory of the Ollama processes while a request runs.

    Uses ``ps``, so it is real where ``ps`` is (macOS, Linux) and None
    elsewhere. On Apple silicon part of a model can live in GPU-mapped
    memory that RSS does not fully show; ``/api/ps`` is reported beside it
    for that reason, and neither is presented as the other.
    """

    def __init__(self, interval: float = 0.25) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mb: float | None = None

    @staticmethod
    def ollama_rss_mb() -> float | None:
        try:
            listing = subprocess.run(
                ["ps", "-axo", "rss=,command="], capture_output=True, text=True, timeout=3
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None

        total_kb = 0
        seen = False

        for line in listing.splitlines():
            parts = line.strip().split(None, 1)

            if len(parts) != 2 or not parts[0].isdigit():
                continue

            command = parts[1].lower()

            if "ollama" in command and "evaluate_day29" not in command and "ps -axo" not in command:
                total_kb += int(parts[0])
                seen = True

        return round(total_kb / 1024, 1) if seen else None

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self.ollama_rss_mb()

            if value is not None:
                self.peak_mb = value if self.peak_mb is None else max(self.peak_mb, value)

            self._stop.wait(self._interval)

    def __enter__(self) -> "MemorySampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=5)


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def _median(values) -> float | None:
    values = [value for value in values if value is not None]

    return round(statistics.median(values), 1) if values else None


def _mean(values) -> float | None:
    values = [value for value in values if value is not None]

    return round(sum(values) / len(values), 3) if values else None


# --- retrieval, once ---------------------------------------------------------


@dataclass
class Context:
    question: Question
    found: EnhancedRetrieval
    retrieval_ms: int
    passes_gate: bool
    text: str
    #: Which rubric facts are present in the context itself - the ceiling
    #: for any profile. A fact retrieval never found is not a model's fault.
    supported_facts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "question_id": self.question.id,
            "retrieved_source_ids": [chunk.chunk_id for chunk in self.found.chunks],
            "retrieved_files": sorted({chunk.file for chunk in self.found.chunks}),
            "best_relevance": round(self.found.best_relevance, 4),
            "passes_gate": self.passes_gate,
            "retrieval_ms": self.retrieval_ms,
            "facts_in_context": self.supported_facts,
            "facts_total": len(self.question.rubric),
        }


async def retrieve_all(questions, settings: RagSettings, retriever: RAGRetriever) -> dict[int, Context]:
    """Every question retrieved once, with no model in the loop."""
    enhanced = EnhancedRetriever(retriever=retriever, settings=settings, provider=None)
    contexts = {}

    for question in questions:
        started = time.perf_counter()
        found = await enhanced.retrieve(question.question, settings=settings)
        seconds = time.perf_counter() - started
        passes = not found.nothing_relevant and found.best_relevance >= settings.answer_threshold
        text = "\n\n".join(chunk.text for chunk in found.chunks)
        facts = [fact.name for fact in question.rubric if fact.found_in(text)]
        contexts[question.id] = Context(question, found, _ms(seconds), passes, text, facts)
        print(
            f"  Q{question.id:<2} {len(found.chunks)} chunk(s) · best {found.best_relevance:.3f} · "
            f"{'gate open' if passes else 'gate closed'}"
            + (f" · facts in context {len(facts)}/{len(question.rubric)}" if question.rubric else "")
        )

    return contexts


# --- quality: every question, one profile ------------------------------------


def recheck(answer) -> int:
    """The surviving citations, validated again from outside the pipeline."""
    if answer.enhanced is None or answer.cited is None:
        return 0

    by_id = {chunk.chunk_id: chunk.text for chunk in answer.enhanced.chunks}

    return sum(
        1
        for citation in answer.cited.citations
        if by_id.get(citation.chunk_id) and flatten(citation.quote) in flatten(by_id[citation.chunk_id])
    )


async def ask(profile: Profile, context: Context, settings: RagSettings, retriever) -> dict:
    question = context.question
    provider = RecordingProvider(OllamaProvider(model=profile.resolved_model, profile=profile))
    agent = RagAgent(
        retriever=retriever,
        settings=settings,
        enhanced=FrozenRetrieval(context.found),
        provider=provider,
        prompt_version=profile.prompt_version,
    )
    run = {
        "question_id": question.id,
        "category": question.category,
        "kind": question.kind,
        "question": question.question,
        "profile": profile.name,
        "model": profile.resolved_model,
        **{key: value for key, value in profile.options.as_dict().items()},
        "prompt_version": profile.prompt_version,
        "retrieved_source_ids": [chunk.chunk_id for chunk in context.found.chunks],
        "retrieval_ms": context.retrieval_ms,
        "error": "",
        "error_reason": "",
    }
    started = time.perf_counter()

    try:
        answer = await agent.ask(question.question, mode=ENHANCED, settings=settings)
    except ProviderUnavailable as error:
        run.update(
            answer="",
            rag_status="error",
            error=error.message.splitlines()[0],
            error_reason=error.reason,
            total_ms=_ms(time.perf_counter() - started),
            correctness=score(question, "", "error"),
            model_calls=len(provider.calls),
            generation_ms=provider.calls[0]["wall_ms"] if provider.calls else None,
        )

        return run

    total_ms = _ms(time.perf_counter() - started)
    cited = answer.cited
    calls = provider.calls
    first = (calls[0]["meta"] or {}) if calls else {}
    support = calls[1] if len(calls) > 1 else None
    text = answer.answer

    run.update(
        answer=text,
        rag_status=cited.rag_status if cited else "",
        confidence=cited.confidence if cited else "",
        cited_source_ids=[source.chunk_id for source in (cited.sources if cited else ())],
        citations=[citation.as_dict() for citation in (cited.citations if cited else ())],
        citations_total=len(cited.citations) if cited else 0,
        citations_valid=recheck(answer),
        rejected_citations=len(cited.rejected_citations) if cited else 0,
        citation_support=cited.citation_support if cited else "",
        correctness=score(question, text, cited.rag_status if cited else ""),
        format_problem=format_problem(text),
        unsupported_identifiers=unsupported_identifiers(text, context.text, question.question),
        model_calls=len(calls),
        generation_ms=_ms(answer.llm_seconds) if calls else 0,
        support_ms=support["wall_ms"] if support else None,
        support_error=support["error"] if support else "",
        total_ms=total_ms,
        prompt_tokens=first.get("prompt_tokens"),
        generated_tokens=first.get("response_tokens"),
        tokens_per_second=first.get("tokens_per_second"),
        load_ms=first.get("load_ms"),
        done_reason=first.get("done_reason", ""),
        thinking_chars=first.get("thinking_chars", 0),
    )

    return run


def _line(run: dict) -> str:
    if run["error"]:
        return f"    {run['profile']:<26} ERROR {run['error_reason']}: {run['error'][:70]}"

    grade = run["correctness"]
    verdict = f"{grade['score']:.2f}" if run["kind"] == "answer" else ("ok " if grade["correct"] else "NO ")

    return (
        f"    {run['profile']:<26} {verdict} · cit {run['citations_valid']}/{run['citations_total']}"
        f" · gen {run['generation_ms']:>6}ms · sup {run['support_ms'] or 0:>6}ms"
        f" · {run.get('generated_tokens') or 0:>5} tok · {run.get('tokens_per_second') or 0:>5} tok/s"
        f" · {run.get('done_reason') or '-'}"
        + (" · FORMAT" if run.get("format_problem") else "")
    )


# --- speed and memory: one prompt, cold then warm ----------------------------


async def _wait_unloaded(model: str, seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + seconds

    while time.monotonic() < deadline:
        loaded = await ollama_service.running_models()

        if not any(entry.get("name") == model or entry.get("model") == model for entry in loaded):
            return True

        await asyncio.sleep(0.5)

    return False


async def _timed(prompt: str, profile: Profile) -> dict:
    with MemorySampler() as sampler:
        started = time.perf_counter()

        try:
            if profile.options.is_default:
                answer = await ollama_service.generate(prompt, model=profile.resolved_model)
            else:
                answer = await ollama_service.generate(
                    prompt, model=profile.resolved_model, options=profile.options
                )
        except ollama_service.OllamaUnavailable as error:
            return {
                "wall_ms": _ms(time.perf_counter() - started),
                "error": error.reason,
                "peak_rss_mb": sampler.peak_mb,
            }

        wall = _ms(time.perf_counter() - started)

    return {"wall_ms": wall, "error": "", "peak_rss_mb": sampler.peak_mb, **answer.metrics()}


async def bench(profile: Profile, context: Context, repeats: int) -> dict:
    """Cold first, then warm: the same prompt ``1 + repeats`` times."""
    prompt = cited_prompt_for(profile.prompt_version, context.question.question, context.found.chunks)
    model = profile.resolved_model
    await ollama_service.unload_model(model)
    unloaded = await _wait_unloaded(model)
    cold = await _timed(prompt, profile)
    warm = [await _timed(prompt, profile) for _ in range(repeats)]
    loaded = await ollama_service.running_models()
    held = next(
        (entry for entry in loaded if entry.get("name") == model or entry.get("model") == model), {}
    )
    good = [run for run in warm if not run["error"]]

    return {
        "question_id": context.question.id,
        "model_was_unloaded": unloaded,
        "cold": cold,
        "warm": warm,
        "warm_median_wall_ms": _median([run["wall_ms"] for run in good]),
        "warm_median_eval_ms": _median([run.get("eval_ms") for run in good]),
        "warm_median_tokens": _median([run.get("response_tokens") for run in good]),
        "warm_median_tokens_per_second": _median([run.get("tokens_per_second") for run in good]),
        "warm_errors": len(warm) - len(good),
        "memory": {
            "ollama_ps_size_mb": round(held["size"] / 2**20, 1) if held.get("size") else None,
            "ollama_ps_size_vram_mb": round(held["size_vram"] / 2**20, 1)
            if held.get("size_vram")
            else None,
            "ollama_ps_context_length": held.get("context_length"),
            "peak_rss_mb": max(
                [run["peak_rss_mb"] for run in [cold, *warm] if run.get("peak_rss_mb")], default=None
            ),
        },
    }


# --- the summary -------------------------------------------------------------


def summarise(profile: Profile, runs: list[dict], benchmark: dict | None) -> dict:
    mine = [run for run in runs if run["profile"] == profile.name]
    answerable = [run for run in mine if run["kind"] == "answer"]
    unanswerable = [run for run in mine if run["kind"] == "insufficient"]
    ok = [run for run in mine if not run["error"]]
    generated = [run for run in ok if run.get("model_calls")]
    answered = [
        run for run in ok if run["kind"] == "answer" and not run["correctness"]["refused"]
    ]

    return {
        "profile": profile.name,
        **profile.as_dict(),
        "questions": len(mine),
        # Errors count as wrong: from the user's side a failed request is
        # not a better answer than a wrong one.
        "correctness_mean": _mean([run["correctness"]["score"] for run in answerable]),
        "fully_correct": sum(1 for run in answerable if run["correctness"]["correct"]),
        "answerable": len(answerable),
        "false_refusals": sum(1 for run in answerable if run["correctness"]["refused"]),
        "insufficient_handled": sum(1 for run in unanswerable if run["correctness"]["correct"]),
        "insufficient_total": len(unanswerable),
        "insufficient_by_gate": sum(1 for run in unanswerable if run.get("rag_status") == INSUFFICIENT),
        "insufficient_hallucinated": sum(1 for run in unanswerable if run["correctness"]["forbidden"]),
        "citations_total": sum(run.get("citations_total", 0) for run in ok),
        "citations_valid": sum(run.get("citations_valid", 0) for run in ok),
        "citation_validity": round(
            sum(run.get("citations_valid", 0) for run in ok)
            / max(1, sum(run.get("citations_total", 0) for run in ok)),
            4,
        ),
        "answers_with_citations": sum(1 for run in answered if run.get("citations_total")),
        "answers": len(answered),
        "format_problems": sum(1 for run in ok if run.get("format_problem")),
        "unsupported_identifiers": sum(len(run.get("unsupported_identifiers", [])) for run in ok),
        "median_generation_ms": _median([run["generation_ms"] for run in generated]),
        "median_support_ms": _median([run.get("support_ms") for run in generated]),
        "median_total_ms": _median([run["total_ms"] for run in generated]),
        "median_generated_tokens": _median([run.get("generated_tokens") for run in generated]),
        "median_tokens_per_second": _median([run.get("tokens_per_second") for run in generated]),
        "truncated": sum(
            1
            for run in mine
            if run.get("done_reason") == "length" or run.get("error_reason") == "truncated"
        ),
        "errors": len(mine) - len(ok),
        "error_reasons": sorted({run["error_reason"] for run in mine if run["error_reason"]}),
        "bench": benchmark,
    }


def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "n/a"

    if isinstance(value, float) and not value.is_integer():
        return f"{value:.2f}{suffix}" if value < 10 else f"{value:.0f}{suffix}"

    return f"{int(value)}{suffix}"


def table(summaries: list[dict]) -> str:
    """Rows are metrics, columns are profiles - fits a terminal, reads on film."""
    width = 19
    names = [summary["profile"] for summary in summaries]
    columns = [f"P{index}" for index in range(1, len(names) + 1)]
    rows = [
        ("correctness (mean 0-1)", lambda s: _fmt(s["correctness_mean"])),
        ("fully correct", lambda s: f"{s['fully_correct']}/{s['answerable']}"),
        ("false refusals", lambda s: str(s["false_refusals"])),
        ("insufficient handled", lambda s: f"{s['insufficient_handled']}/{s['insufficient_total']}"),
        ("  of them by the gate", lambda s: str(s["insufficient_by_gate"])),
        ("  invented an answer", lambda s: str(s["insufficient_hallucinated"])),
        ("citation validity", lambda s: f"{s['citations_valid']}/{s['citations_total']}"),
        ("answers with citations", lambda s: f"{s['answers_with_citations']}/{s['answers']}"),
        ("format problems", lambda s: str(s["format_problems"])),
        ("unknown identifiers", lambda s: str(s["unsupported_identifiers"])),
        ("median generation", lambda s: _fmt(s["median_generation_ms"], " ms")),
        ("median support check", lambda s: _fmt(s["median_support_ms"], " ms")),
        ("median total", lambda s: _fmt(s["median_total_ms"], " ms")),
        ("median tokens out", lambda s: _fmt(s["median_generated_tokens"])),
        ("median tokens/s", lambda s: _fmt(s["median_tokens_per_second"])),
        ("bench cold total", lambda s: _fmt(((s["bench"] or {}).get("cold") or {}).get("wall_ms"), " ms")),
        ("bench cold load", lambda s: _fmt(((s["bench"] or {}).get("cold") or {}).get("load_ms"), " ms")),
        ("bench warm median", lambda s: _fmt((s["bench"] or {}).get("warm_median_wall_ms"), " ms")),
        ("bench warm tokens/s", lambda s: _fmt((s["bench"] or {}).get("warm_median_tokens_per_second"))),
        ("memory (ollama ps)", lambda s: _fmt(((s["bench"] or {}).get("memory") or {}).get("ollama_ps_size_mb"), " MB")),
        ("memory (peak RSS)", lambda s: _fmt(((s["bench"] or {}).get("memory") or {}).get("peak_rss_mb"), " MB")),
        ("truncated by num_predict", lambda s: str(s["truncated"])),
        ("errors", lambda s: str(s["errors"])),
    ]
    lines = [RULE, "=== Day 29 · Summary ===", RULE]

    for column, summary in zip(columns, summaries):
        def given(name):
            value = summary[name]
            return "default" if value is None else value

        options = " ".join(f"{name}={given(name)}" for name in ("temperature", "num_predict", "num_ctx", "think"))
        lines.append(f"{column}  {summary['profile']}  ·  {summary['model']}  ·  {options}  ·  prompt {summary['prompt_version']}")

    lines.append("")
    lines.append(f"{'':<26}" + "".join(f"{column:>{width}}" for column in columns))

    for label, cell in rows:
        lines.append(f"{label:<26}" + "".join(f"{cell(summary):>{width}}" for summary in summaries))

    return "\n".join(lines)


# --- the run -----------------------------------------------------------------


def _full_tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


async def preflight(profiles: list[Profile]) -> tuple[dict, list[Profile], list[dict]]:
    """What is installed, what each model is, and which profiles can run."""
    health = await ollama_service.check_ollama_health()

    if not health.get("available"):
        raise SystemExit(health.get("error") or "Ollama is not available")

    installed = set(health.get("models", []))
    runnable, skipped, models = [], [], {}

    for profile in profiles:
        tag = profile.resolved_model

        if tag not in installed and _full_tag(tag) not in installed:
            skipped.append(
                {
                    "profile": profile.name,
                    "model": tag,
                    "reason": f"not installed - this script does not download; pull it with: ollama pull {tag}",
                }
            )
            continue

        runnable.append(profile)

        if tag not in models:
            models[tag] = await ollama_service.show_model(tag)

    sizes = {entry.get("name"): entry.get("size") for entry in await ollama_service.list_models()}

    for tag, info in models.items():
        size = sizes.get(tag) or sizes.get(_full_tag(tag))
        info["disk_size_mb"] = round(size / 2**20, 1) if size else None

    return (
        {"version": await ollama_service.server_version(), "installed": sorted(installed), "models": models},
        runnable,
        skipped,
    )


def machine() -> dict:
    total = None

    try:
        raw = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3).stdout
        total = round(int(raw.strip()) / 2**30, 1) if raw.strip().isdigit() else None
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "memory_gb": total,
    }


def selected_profiles(options) -> list[Profile]:
    profiles = [ollama_profiles.profile_for(name) for name in options.profiles]
    base = ollama_profiles.profile_for(options.variant_profile)
    profiles += [base.on_model(tag) for tag in options.variants]

    return profiles


async def run(options) -> int:
    questions, bench_id = load_questions(options.questions)

    if options.only:
        questions = [question for question in questions if question.id in set(options.only)]

    if not questions:
        raise SystemExit("no questions selected")

    settings = DEFAULT_SETTINGS.with_overrides(
        answer_threshold=options.answer_threshold, query_rewrite=False
    )
    print(f"\n{RULE}\n=== Day 29 · Local LLM optimisation ===\n{RULE}")
    ollama, profiles, skipped = await preflight(selected_profiles(options))
    print(f"Ollama {ollama['version'] or '?'} · {len(questions)} question(s) · threshold {settings.answer_threshold}")

    for tag, info in ollama["models"].items():
        details = info.get("details", {})
        print(
            f"  {tag}: {info.get('basename') or details.get('family', '?')} {info.get('finetune', '')} "
            f"{details.get('parameter_size', '')} {details.get('quantization_level', '')} · "
            f"{_fmt(info.get('disk_size_mb'), ' MB')} on disk · context {info.get('context_length')}"
        )

    for item in skipped:
        print(f"  SKIPPED {item['profile']}: {item['reason']}")

    if not profiles:
        raise SystemExit("nothing to run")

    print("\nRetrieval (once per question, no model, query rewrite off):")
    retriever = RAGRetriever(strategy=options.strategy)
    contexts = await retrieve_all(questions, settings, retriever)
    bench_context = contexts.get(bench_id) or next(iter(contexts.values()))
    started = time.perf_counter()
    runs: list[dict] = []
    benches: dict[str, dict] = {}
    measured_at: dict[str, str] = {}

    for profile in profiles:
        print(f"\n--- {profile.name} · {profile.resolved_model} · {profile.options.request_fields() or 'model defaults'} · prompt {profile.prompt_version}")
        measured_at[profile.name] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        if options.repeats > 0:
            benches[profile.name] = await bench(profile, bench_context, options.repeats)
            result = benches[profile.name]
            print(
                f"    bench Q{bench_context.question.id}: cold {result['cold'].get('wall_ms')}ms "
                f"(load {result['cold'].get('load_ms')}ms) · warm median {result['warm_median_wall_ms']}ms · "
                f"{result['warm_median_tokens_per_second']} tok/s · ps {result['memory']['ollama_ps_size_mb']} MB"
            )

        for question in questions:
            item = await ask(profile, contexts[question.id], settings, retriever)
            runs.append(item)
            print(f"  Q{question.id:<2}{_line(item)}")

    seconds = time.perf_counter() - started
    summaries = [summarise(profile, runs, benches.get(profile.name)) for profile in profiles]
    document = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": machine(),
        "ollama": ollama,
        "settings": {
            "strategy": options.strategy,
            "answer_threshold": settings.answer_threshold,
            "query_rewrite": settings.query_rewrite,
            "final_top_k": settings.final_top_k,
            "similarity_threshold": settings.similarity_threshold,
        },
        "questions_file": str(options.questions.relative_to(PROJECT_ROOT))
        if options.questions.is_relative_to(PROJECT_ROOT)
        else str(options.questions),
        "question_ids": [question.id for question in questions],
        "bench": {"question_id": bench_context.question.id, "warm_repeats": options.repeats},
        "evaluation_seconds": round(seconds, 1),
        "contexts": [contexts[question.id].as_dict() for question in questions],
        "profiles": {
            profile.name: {**profile.as_dict(), "measured_at": measured_at[profile.name]}
            for profile in profiles
        },
        "summary": {summary["profile"]: summary for summary in summaries},
        "results": runs,
        "skipped": skipped,
    }

    if options.merge and RESULTS_FILE.exists():
        document = merged(json.loads(RESULTS_FILE.read_text(encoding="utf-8")), document)
        summaries = list(document["summary"].values())

    text = table(summaries)
    print(f"\n{text}")
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_FILE.write_text(text + "\n", encoding="utf-8")
    print(f"\nWritten: {RESULTS_FILE}\nWritten: {SUMMARY_FILE}")

    return 0


def merged(previous: dict, current: dict) -> dict:
    """Earlier profiles kept, re-run profiles replaced - only when both runs
    asked the same questions under the same retrieval settings, so the
    columns of the table are still comparable. Each profile keeps its own
    ``measured_at``."""
    if (
        previous.get("question_ids") != current["question_ids"]
        or previous.get("settings") != current["settings"]
        or previous.get("contexts") is None
        or [c["retrieved_source_ids"] for c in previous["contexts"]]
        != [c["retrieved_source_ids"] for c in current["contexts"]]
    ):
        print("\n(--merge refused: the earlier run used different questions, settings or context)")

        return current

    fresh = set(current["profiles"])
    kept = [name for name in previous.get("profiles", {}) if name not in fresh]
    result = dict(current)
    result["profiles"] = {**{name: previous["profiles"][name] for name in kept}, **current["profiles"]}
    result["summary"] = {**{name: previous["summary"][name] for name in kept}, **current["summary"]}
    result["results"] = [run for run in previous["results"] if run["profile"] in kept] + current["results"]
    result["ollama"]["models"] = {**previous.get("ollama", {}).get("models", {}), **current["ollama"]["models"]}

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluate_day29",
        description="The Day 29 questions, answered from the same context by each local-model profile.",
    )
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=DEFAULT_PROFILES,
        choices=list(PROFILES),
        help=f"profiles to run (default: {' '.join(DEFAULT_PROFILES)})",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[],
        help="other installed model tags to run with --variant-profile (a quantisation or a smaller model)",
    )
    parser.add_argument("--variant-profile", default=ollama_profiles.OPTIMIZED_NAME, choices=list(PROFILES))
    parser.add_argument("--repeats", type=int, default=3, help="warm benchmark runs per profile (0 skips the benchmark)")
    parser.add_argument("--answer-threshold", type=float, default=0.35)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--only", nargs="*", type=int, default=[], help="run only these question ids")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--merge", action="store_true", help="keep earlier profiles' results in the file")
    parser.add_argument("--verbose", action="store_true", help="also show the pipeline's own log lines")

    parsed = parser.parse_args(argv)

    if parsed.repeats < 0:
        parser.error("--repeats cannot be negative")

    return parsed


def main() -> int:
    options = parse_args()
    logging.basicConfig(level=logging.INFO if options.verbose else logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return asyncio.run(run(options))


if __name__ == "__main__":
    raise SystemExit(main())
