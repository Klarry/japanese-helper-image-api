"""Day 19: running search -> summarize -> save_to_file as one chain.

The learner writes one sentence - "look 学習 up, summarise it and save it" -
and three tools run, each on what the one before it returned. Nobody picks
the tools by hand and nobody copies data between them.

Who decides what. The planner (Day 17) still decides *whether* tools are
needed and names them; it cannot do more than that, because the arguments of
the second stage are the output of the first and no one knows those before
the first has run. So the plan is read here as a destination rather than as
three calls: the furthest stage it names is how far the chain goes, and this
module supplies every argument in between. A plan naming only ``search`` is
not a pipeline at all - it is a lookup, and it takes the ordinary path.

Stopping. A stage that fails ends the run there: the stages after it are not
attempted, nothing is saved, and the prompt block says which stage failed
and why, so the answer tells the learner instead of pretending a file exists.
"""

import logging
from collections.abc import Sequence
from typing import Any, NamedTuple

from app.services.agent_tool_planner import PlannedCall
from app.services.agent_tools import McpToolbox, call_line
from app.services.mcp_client import McpToolCallResult
from app.services.pipeline_tools import STAGES

logger = logging.getLogger(__name__)

SEARCH, SUMMARIZE, SAVE = STAGES

_CAPTION = (
    "PIPELINE RESULTS (ran just now, automatically, through MCP): "
    "search -> summarize -> save_to_file. Each stage was given the previous stage's result "
    "unchanged; nothing was looked up twice and nothing was invented in between. Base the "
    "answer on this data, and tell the learner what was found, what the summary says and - "
    "if it was saved - the file name it was saved under."
)

_STOPPED = (
    "The chain stopped at {stage}: {error}. The stages after it did not run. Tell the learner "
    "that the chain did not finish and at which step, and do not claim anything was saved."
)


class PipelineRequest(NamedTuple):
    """What the plan asked for: a query, and how far down the chain to go."""

    query: str
    stages: tuple[str, ...]


class PipelineRun(NamedTuple):
    """What actually happened, stage by stage, in order."""

    query: str
    requested: tuple[str, ...]
    results: tuple[McpToolCallResult, ...]

    @property
    def completed(self) -> bool:
        """Every requested stage ran and none of them failed."""
        return len(self.results) == len(self.requested) and all(result.ok for result in self.results)

    @property
    def failed_at(self) -> McpToolCallResult | None:
        return next((result for result in self.results if not result.ok), None)

    @property
    def saved_as(self) -> str:
        """The file name, when the chain got as far as saving one."""
        for result in self.results:
            if result.name == SAVE and result.ok and isinstance(result.data, dict):
                return str(result.data.get("file_name") or "")

        return ""


def pipeline_request(calls: Sequence[PlannedCall]) -> PipelineRequest | None:
    """Read a tool plan as a pipeline request, or None when it is not one.

    The furthest stage the plan names decides the length of the chain, since
    asking for something saved implies the steps that produce it. ``search``
    on its own is a lookup, not a chain, and is left to the ordinary path.
    """
    named = [call.tool for call in calls if call.tool in STAGES]

    if not named:
        return None

    furthest = max(STAGES.index(tool) for tool in named)

    if furthest == 0:
        return None

    query = _query_in(calls)

    if not query:
        logger.warning("A pipeline was planned without anything to search for; ignoring it")
        return None

    return PipelineRequest(query, STAGES[: furthest + 1])


def _query_in(calls: Sequence[PlannedCall]) -> str:
    """The word to start from: what the planner put in the search call, or
    anything else it called a query."""
    for call in calls:
        if call.tool == SEARCH:
            candidate = str(call.arguments.get("query", "")).strip()

            if candidate:
                return candidate

    for call in calls:
        if call.tool in STAGES:
            candidate = str(call.arguments.get("query", "")).strip()

            if candidate:
                return candidate

    return ""


class PipelineRunner:
    """Runs the stages in order, feeding each one the last one's result."""

    def __init__(self, toolbox: McpToolbox) -> None:
        self._toolbox = toolbox

    async def run(self, request: PipelineRequest) -> PipelineRun:
        """Never raises: a stage that could not run comes back as a failed
        result, which ends the run the same way a tool error does."""
        results: list[McpToolCallResult] = []
        findings: dict[str, Any] | None = None
        summary: dict[str, Any] | None = None

        for stage in request.stages:
            arguments = self._arguments(stage, request.query, findings, summary)
            result = await self._toolbox.call(stage, arguments)
            results.append(result)

            logger.info(
                "Pipeline %s/%s: %s -> %s",
                len(results),
                len(request.stages),
                stage,
                "ok" if result.ok else f"failed - {result.error}",
            )

            if not result.ok:
                break

            data = result.data if isinstance(result.data, dict) else {}

            if stage == SEARCH:
                findings = data
            elif stage == SUMMARIZE:
                summary = data

        return PipelineRun(request.query, request.stages, tuple(results))

    @staticmethod
    def _arguments(
        stage: str,
        query: str,
        findings: dict[str, Any] | None,
        summary: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """The one place data crosses between stages."""
        if stage == SEARCH:
            return {"query": query}

        if stage == SUMMARIZE:
            return {"findings": findings or {}}

        return {"summary": summary or {}, "findings": findings or {}}


def pipeline_section(run: PipelineRun) -> str:
    """The chain as the model reads it: every stage, its arguments and what
    it returned - and, when it stopped early, where and why."""
    lines = [_CAPTION, f"Requested chain: {' -> '.join(run.requested)} for '{run.query}'."]
    lines.extend(call_line(result) for result in run.results)

    stopped = run.failed_at

    if stopped is not None:
        lines.append(_STOPPED.format(stage=stopped.name, error=stopped.error))
    elif run.saved_as:
        lines.append(f"Saved as {run.saved_as}.")

    return "\n".join(lines)
