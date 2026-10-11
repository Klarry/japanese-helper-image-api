"""Day 30: what stands between the network and the model on this machine.

Day 27 put the local model behind ``POST /local-chat`` for one phone on the
desk. Day 30 lets anything on the network reach it, and a model that takes
seconds per answer on a laptop has to be protected from the network the way
any shared resource is:

- **a rate limit** per client address - a sliding window, in memory;
- **an input limit** in characters, checked before anything else is done;
- **a context budget** in (estimated) tokens, so that the prompt and the
  answer it asks for fit the context window the model is actually given;
- **a concurrency cap** - at most N generations at once, and a clear "busy"
  for the rest instead of a queue that grows until every request times out;
- **a timeout** of its own, shorter than the HTTP client's.

Everything here is plain Python in one process. The limits are per process
and reset when it restarts: that is stated, not hidden, and it is enough for
a private service run as one uvicorn worker. There is no Redis and no new
dependency.

Nothing in this module talks to a model. It decides whether a request may
reach one, and says why not when it may not.
"""

import math
import os
import re
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Request

#: The project root, so a relative history directory means the same place
#: whichever directory uvicorn was started from.
_ROOT = Path(__file__).resolve().parents[2]

#: What a conversation id may look like. It becomes a file name, so nothing
#: else is allowed in - no dots, no slashes, nothing a path could be built of.
CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: Characters that a tokenizer turns into roughly one token each. This is a
#: Japanese-learning app: a budget that assumed English would be wrong by a
#: factor of three on exactly the text it is used for.
_DENSE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")


def _int(environ: Mapping[str, str], key: str, default: int, minimum: int = 0) -> int:
    raw = (environ.get(key) or "").strip()

    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{key}={raw!r} is not a whole number") from None

    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")

    return value


def _float(environ: Mapping[str, str], key: str, default: float) -> float:
    raw = (environ.get(key) or "").strip()

    if not raw:
        return default

    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{key}={raw!r} is not a number") from None

    if value <= 0:
        raise ValueError(f"{key} must be above zero")

    return value


@dataclass(frozen=True)
class LocalChatLimits:
    """Every number the service is protected by, in one place."""

    #: Chat requests one client address may make per window.
    rate_limit: int = 10
    rate_window_seconds: float = 60.0
    #: The message itself, in characters - checked before anything else.
    max_message_chars: int = 4000
    #: Generations running at once. More than this get "busy", not a queue.
    max_concurrent: int = 2
    #: How long one generation may take before the request gives up.
    timeout_seconds: float = 120.0
    #: The context window used when the active profile does not set one.
    default_num_ctx: int = 4096
    #: Tokens kept free for the answer when the profile sets no num_predict.
    output_reserve_tokens: int = 1024
    #: How many earlier messages of a conversation may go into the prompt.
    recent_messages: int = 8
    #: Where conversations are kept, one file each, by the Day 7 storage.
    history_dir: str = "data/local_chat"
    #: Proxy addresses whose X-Forwarded-For is believed. Empty: none is.
    trusted_proxies: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "LocalChatLimits":
        env = os.environ if environ is None else environ
        proxies = tuple(
            part.strip() for part in (env.get("LOCAL_CHAT_TRUSTED_PROXIES") or "").split(",") if part.strip()
        )

        return cls(
            rate_limit=_int(env, "LOCAL_CHAT_RATE_LIMIT", cls.rate_limit, minimum=1),
            rate_window_seconds=_float(env, "LOCAL_CHAT_RATE_WINDOW_SECONDS", cls.rate_window_seconds),
            max_message_chars=_int(env, "LOCAL_CHAT_MAX_MESSAGE_CHARS", cls.max_message_chars, minimum=1),
            max_concurrent=_int(env, "LOCAL_CHAT_MAX_CONCURRENT", cls.max_concurrent, minimum=1),
            timeout_seconds=_float(env, "LOCAL_CHAT_TIMEOUT_SECONDS", cls.timeout_seconds),
            default_num_ctx=_int(env, "LOCAL_CHAT_DEFAULT_NUM_CTX", cls.default_num_ctx, minimum=256),
            output_reserve_tokens=_int(env, "LOCAL_CHAT_OUTPUT_RESERVE_TOKENS", cls.output_reserve_tokens, minimum=1),
            recent_messages=_int(env, "LOCAL_CHAT_RECENT_MESSAGES", cls.recent_messages),
            history_dir=(env.get("LOCAL_CHAT_HISTORY_DIR") or cls.history_dir).strip(),
            trusted_proxies=proxies,
        )

    @property
    def history_path(self) -> Path:
        path = Path(self.history_dir)

        return path if path.is_absolute() else _ROOT / path

    def as_dict(self) -> dict:
        return {
            "rate_limit": self.rate_limit,
            "rate_window_seconds": self.rate_window_seconds,
            "max_message_chars": self.max_message_chars,
            "max_concurrent": self.max_concurrent,
            "timeout_seconds": self.timeout_seconds,
            "default_num_ctx": self.default_num_ctx,
            "output_reserve_tokens": self.output_reserve_tokens,
            "recent_messages": self.recent_messages,
        }


# --- who is asking -------------------------------------------------------------


def client_address(request: Request, trusted_proxies: Sequence[str] = ()) -> str:
    """The address a limit is counted against.

    The socket's peer, always - unless that peer is a proxy we were told to
    trust, and only then the first address in X-Forwarded-For. A header any
    client can write is not an identity: believing it by default would let
    every request claim to come from somewhere new and walk past the limit.
    """
    peer = request.client.host if request.client else "unknown"

    if peer in trusted_proxies:
        forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()

        if forwarded:
            return forwarded

    return peer


# --- how often ---------------------------------------------------------------------


@dataclass
class RateDecision:
    allowed: bool
    remaining: int
    retry_after: int = 0


class RateLimiter:
    """A sliding window per key: at most ``limit`` requests in any
    ``window`` seconds. Exact rather than bucketed, so a burst at the end of
    one minute and the start of the next is still one burst."""

    def __init__(self, limit: int, window: float, clock=time.monotonic) -> None:
        self._limit = limit
        self._window = window
        self._clock = clock
        self._seen: dict[str, deque] = {}

    def check(self, key: str) -> RateDecision:
        now = self._clock()
        seen = self._seen.setdefault(key, deque())

        while seen and now - seen[0] >= self._window:
            seen.popleft()

        if len(seen) >= self._limit:
            wait = self._window - (now - seen[0])

            return RateDecision(False, 0, max(1, math.ceil(wait)))

        seen.append(now)
        self._forget_idle(now)

        return RateDecision(True, self._limit - len(seen))

    def _forget_idle(self, now: float) -> None:
        # Keys whose newest request left the window are dropped, so the map
        # cannot grow with every address that ever asked once.
        if len(self._seen) < 1024:
            return

        for key in [key for key, seen in self._seen.items() if not seen or now - seen[-1] >= self._window]:
            del self._seen[key]


# --- how many at once -----------------------------------------------------------------


class ConcurrencyGate:
    """At most ``limit`` generations in flight; the rest are turned away.

    Not a semaphore that waits: a request that cannot start now is told so
    at once, which is what keeps a slow model from collecting a queue of
    requests that will all time out together. One event loop, no threads,
    so a counter is all the bookkeeping there is - and ``leave`` is always
    called from a ``finally``, so an exception or a timeout cannot leak a
    slot.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.active = 0

    @property
    def limit(self) -> int:
        return self._limit

    def try_enter(self) -> bool:
        if self.active >= self._limit:
            return False

        self.active += 1

        return True

    def leave(self) -> None:
        self.active = max(0, self.active - 1)


# --- how much ----------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """An upper-leaning estimate of how many tokens ``text`` is.

    Not a tokenizer - Ollama does not expose one - and not presented as
    one. Kana, kanji and hangul count one token each; everything else one
    token per three characters, which over-counts English (about four per
    token) on purpose. The real prompt size comes back from Ollama with
    every answer and is returned beside this estimate.
    """
    dense = len(_DENSE.findall(text))

    return dense + math.ceil((len(text) - dense) / 3)


class ContextTooLarge(Exception):
    def __init__(self, needed: int, budget: int) -> None:
        super().__init__(f"the message needs about {needed} tokens; the prompt budget is {budget}")
        self.needed = needed
        self.budget = budget


@dataclass
class ContextPlan:
    """The prompt that will be sent, and what it cost to fit it."""

    prompt: str
    estimated_tokens: int
    budget: int
    history_used: int = 0
    history_dropped: int = 0

    def as_dict(self) -> dict:
        return {
            "estimated_prompt_tokens": self.estimated_tokens,
            "prompt_budget_tokens": self.budget,
            "history_messages_used": self.history_used,
            "history_messages_dropped": self.history_dropped,
        }


def plan_prompt(
    message: str,
    history: Sequence[Mapping[str, str]],
    num_ctx: int,
    reserve: int,
    recent: int,
    render,
) -> ContextPlan:
    """Fit the conversation into the context window, newest first.

    The budget is ``num_ctx`` minus what is kept free for the answer. The
    current message always goes in - if it alone does not fit, the request
    is refused rather than shortened, because a question the model never
    fully read is not a question it answered. Earlier messages are added
    from the newest back while they fit, up to ``recent`` of them; the rest
    are dropped and counted, never silently.

    With no history the prompt is the message itself - exactly what Day 27
    sent.
    """
    budget = num_ctx - reserve

    if budget <= 0:
        raise ContextTooLarge(estimate_tokens(message), max(0, budget))

    alone = estimate_tokens(message)

    if alone > budget:
        raise ContextTooLarge(alone, budget)

    candidates = list(history)[-recent:] if recent else []
    dropped = len(history) - len(candidates)
    kept: list = []

    for entry in reversed(candidates):
        trial = [entry, *kept]
        size = estimate_tokens(_with_history(message, trial, render))

        if size > budget:
            dropped += len(candidates) - len(kept)
            break

        kept = trial

    prompt = _with_history(message, kept, render) if kept else message

    return ContextPlan(prompt, estimate_tokens(prompt), budget, len(kept), dropped)


def _with_history(message: str, history: Sequence, render) -> str:
    return (
        "Conversation so far:\n"
        f"{render(list(history))}\n\n"
        "Continue the conversation. Answer the last message.\n"
        f"Learner: {message}\nAssistant:"
    )


# --- all of it, for one process -------------------------------------------------------


@dataclass
class LocalChatGuard:
    limits: LocalChatLimits
    limiter: RateLimiter = field(init=False)
    gate: ConcurrencyGate = field(init=False)
    #: Conversations with a request in flight - a second one for the same
    #: conversation is refused rather than allowed to race the first to the file.
    busy_conversations: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.limiter = RateLimiter(self.limits.rate_limit, self.limits.rate_window_seconds)
        self.gate = ConcurrencyGate(self.limits.max_concurrent)

    @classmethod
    def from_env(cls) -> "LocalChatGuard":
        return cls(LocalChatLimits.from_env())

    def status(self) -> dict:
        return {**self.limits.as_dict(), "active_generations": self.gate.active}
