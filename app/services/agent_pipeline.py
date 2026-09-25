"""Days 19-20: the orchestrator - one request, three servers, in order.

The learner writes one sentence - "look 学習 up, summarise it and save it" -
and three tools run, each on what the one before it returned. Since Day 20
each of them lives on a server of its own (japanese-data, processing,
storage), so the orchestrator does two things at once: it decides the order,
and it decides where each call goes. The second part it does not decide by
hand - it asks the registry, which knows because every server was asked what
it offers.

Who decides what. The planner (Day 17) still decides *whether* tools are
needed and names them; it cannot do more than that, because the arguments of
the second stage are the output of the first and no one knows those before
the first has run. So the plan is read here as a destination rather than as
three calls: the furthest stage it names is how far the chain goes, and this
module supplies every argument in between. A plan naming only ``search`` is
not a pipeline at all - it is a lookup, and it takes the ordinary path.

Stopping. Five things end a run, and each one says so in the log and in the
prompt: a server that cannot be reached, a tool no server offers, a tool
that ran and failed, a server that went quiet (timeout), and a stage that
answered with something the next stage cannot use. In every case the stages
after it are not attempted and nothing is saved - the point of checking the
output of a stage is that the chain must not continue on data it cannot
trust.
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
    "PIPELINE RESULTS (ran just now, automatically, through MCP, across several servers): "
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
    #: The server each result came from, by the same index.
    servers: tuple[str, ...] = ()

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

    def server_of(self, index: int) -> str:
        return self.servers[index] if index < len(self.servers) else ""


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
        logger.warning("[Orchestrator] A pipeline was planned without anything to search for")
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


def usable(stage: str, data: Any) -> str:
    """Whether the next stage can be given this stage's output.

    Returns "" when it can, and what is wrong with it when it cannot. A tool
    that answers ``ok`` with something unusable is the one failure the MCP
    layer cannot catch - the call worked, the data did not - so the chain
    checks it before it hands it on.
    """
    if not isinstance(data, dict):
        return f"{stage} returned {type(data).__name__}, not an object"

    if stage == SEARCH:
        if not isinstance(data.get("matches"), list):
            return "search returned no 'matches' list"

        if not str(data.get("query", "")).strip():
            return "search returned no 'query'"

    if stage == SUMMARIZE and not str(data.get("summary", "")).strip():
        return "summarize returned an empty 'summary'"

    return ""


class PipelineRunner:
    """Runs the stages in order, each on the server that offers it, feeding
    each one the last one's result."""

    def __init__(self, toolbox: McpToolbox) -> None:
        self._toolbox = toolbox

    async def run(self, request: PipelineRequest) -> PipelineRun:
        """Never raises: a stage that could not run, or answered with
        something unusable, comes back as a failed result, which ends the run
        the same way a tool error does."""
        logger.info("[Orchestrator] User request received: %r", request.query)

        results: list[McpToolCallResult] = []
        servers: list[str] = []
        findings: dict[str, Any] | None = None
        summary: dict[str, Any] | None = None

        for stage in request.stages:
            logger.info("[Orchestrator] Selected tool: %s", stage)
            server = await self._server_for(stage)
            logger.info("[Orchestrator] Server: %s", server)

            result = await self._toolbox.call(stage, self._arguments(stage, request.query, findings, summary))
            results.append(result)
            servers.append(server)

            if not result.ok:
                logger.warning("[Orchestrator] %s failed on %s: %s", stage, server, result.error)
                break

            problem = usable(stage, result.data)

            if problem:
                logger.warning("[Orchestrator] %s returned an unusable result: %s", stage, problem)
                results[-1] = McpToolCallResult(
                    name=result.name,
                    arguments=result.arguments,
                    ok=False,
                    data=result.data,
                    error=f"invalid output from {stage}: {problem}",
                )
                break

            logger.info("[Orchestrator] %s completed", stage)

            if stage == SEARCH:
                findings = result.data
            elif stage == SUMMARIZE:
                summary = result.data

        run = PipelineRun(request.query, request.stages, tuple(results), tuple(servers))

        if run.completed:
            logger.info("[Orchestrator] Pipeline completed%s", f": saved as {run.saved_as}" if run.saved_as else "")
        else:
            logger.warning(
                "[Orchestrator] Pipeline stopped at %s",
                run.failed_at.name if run.failed_at else "an unknown stage",
            )

        return run

    async def _server_for(self, stage: str) -> str:
        """Which server this stage is routed to, for the log and the report.
        An unknown tool is not raised here - the call itself reports it, in
        one place, the same way an unreachable server is reported."""
        table = await self._toolbox.routing_table()

        return table.get(stage, "unknown")

    @staticmethod
    def _arguments(
        stage: str,
        query: str,
        findings: dict[str, Any] | None,
        summary: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """The one place data crosses between stages - and between servers."""
        if stage == SEARCH:
            return {"query": query}

        if stage == SUMMARIZE:
            return {"findings": findings or {}}

        return {"summary": summary or {}, "findings": findings or {}}


def pipeline_section(run: PipelineRun) -> str:
    """The chain as the model reads it: every stage, the server it ran on,
    its arguments and what it returned - and, when it stopped early, where
    and why."""
    lines = [_CAPTION, f"Requested chain: {' -> '.join(run.requested)} for '{run.query}'."]

    for index, result in enumerate(run.results):
        server = run.server_of(index)
        lines.append(f"{call_line(result)}   [server: {server}]" if server else call_line(result))

    stopped = run.failed_at

    if stopped is not None:
        lines.append(_STOPPED.format(stage=stopped.name, error=stopped.error))
    elif run.saved_as:
        lines.append(f"Saved as {run.saved_as}.")

    return "\n".join(lines)
