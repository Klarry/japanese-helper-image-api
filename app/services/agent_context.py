"""What each strategy puts in front of the model.

One place decides this for every strategy, so the prompt a request sends,
the context GET /agent/context reports, and the messages counted as "sent"
can never drift apart - they all come from the same ContextWindow.

None of the summary-free strategies touch ConversationHistory.summary: they
work from the messages a branch already stores, plus (for Sticky Facts) the
key-value memory kept alongside them.
"""

from dataclasses import dataclass

from app.core.config import AGENT_RECENT_MESSAGES_KEPT
from app.schemas.agent import ContextStrategy
from app.services.agent_history_storage import ConversationHistory, HistoryMessage, format_transcript
from app.services.agent_memory import MemoryLayers


@dataclass(frozen=True)
class ContextWindow:
    """Exactly what one request sends: the prompt sections, and the pieces
    they were built from so the caller can report them without guessing."""

    sections: list[str]
    messages: list[HistoryMessage]
    summary_tokens: int | None
    facts: dict[str, str]

    @property
    def text(self) -> str:
        return "\n\n".join(self.sections)


def _transcript_section(caption: str, messages: list[HistoryMessage]) -> list[str]:
    if not messages:
        return []

    return [f"{caption}\n{format_transcript(messages)}"]


def _facts_section(facts: dict[str, str]) -> list[str]:
    if not facts:
        return []

    listed = "\n".join(f"- {key}: {value}" for key, value in facts.items())
    return [f"Known facts about the learner and this conversation:\n{listed}"]


def build_context(
    strategy: ContextStrategy,
    history: ConversationHistory,
    recent_messages_kept: int = AGENT_RECENT_MESSAGES_KEPT,
    memory: MemoryLayers | None = None,
) -> ContextWindow:
    """Build the context for one request, without changing the history.

    * FULL and BRANCHING send the branch's whole conversation. They differ
      only in where that conversation comes from - BRANCHING is the strategy
      used when the point is which branch is being talked on, and every
      branch keeps its own messages.
    * SUMMARY sends the running summary plus the messages still stored
      verbatim - unchanged from how compression already worked.
    * SLIDING_WINDOW sends only the newest ``recent_messages_kept``
      messages. Older ones stay on disk (so the same dialogue can still be
      replayed under another strategy) but never reach the model.
    * STICKY_FACTS sends the key-value facts plus that same window, and no
      summary at all.
    * LAYERED_MEMORY sends the three memory layers, each as its own labelled
      section: long-term, then working, then short-term. Nothing is merged -
      the model is told which layer each line came from, which is what lets
      one layer be cleared without disturbing the others.
    """
    window = history.messages[-recent_messages_kept:] if recent_messages_kept > 0 else []

    if strategy is ContextStrategy.LAYERED_MEMORY:
        layers = memory or MemoryLayers()

        return ContextWindow(
            sections=layers.as_sections(),
            messages=layers.short_term.messages,
            summary_tokens=0,
            facts={},
        )

    if strategy is ContextStrategy.SUMMARY:
        sections = []

        if history.summary:
            sections.append(f"Summary of the earlier part of the conversation:\n{history.summary}")

        sections += _transcript_section("Conversation so far:", history.messages)

        return ContextWindow(
            sections=sections,
            messages=history.messages,
            summary_tokens=history.summary_tokens if history.summary else 0,
            facts={},
        )

    if strategy is ContextStrategy.SLIDING_WINDOW:
        return ContextWindow(
            sections=_transcript_section("Most recent messages:", window),
            messages=window,
            summary_tokens=0,
            facts={},
        )

    if strategy is ContextStrategy.STICKY_FACTS:
        return ContextWindow(
            sections=_facts_section(history.facts) + _transcript_section("Most recent messages:", window),
            messages=window,
            summary_tokens=0,
            facts=history.facts,
        )

    return ContextWindow(
        sections=_transcript_section("Conversation so far:", history.messages),
        messages=history.messages,
        summary_tokens=0,
        facts={},
    )
