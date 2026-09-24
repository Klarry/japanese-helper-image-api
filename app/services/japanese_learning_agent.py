"""Day 6-8: a small, dedicated LLM agent for the Japanese-learning app.

Every other module in this package is a bare async function that calls
gemini_service directly. JapaneseLearningAgent is different on purpose: the
task asks for a real agent entity that owns the whole request -> context ->
prompt -> Gemini call -> history update -> response path for open-ended
learner questions, so the routes never do more than call one of its methods.
"""

import logging
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timezone

from fastapi import HTTPException

from app.core.config import (
    AGENT_COMPRESSION_ENABLED,
    AGENT_MCP_TOOLS_ENABLED,
    AGENT_RECENT_MESSAGES_KEPT,
    AGENT_TASK_TRACKING_ENABLED,
)
from app.schemas.agent import (
    AgentBranchResponse,
    AgentChatResponse,
    AgentCheckpointResponse,
    AgentCompressionStatus,
    AgentDigestResponse,
    AgentContextResponse,
    AgentHistoryMessage,
    AgentInvariant,
    AgentInvariantRequest,
    AgentInvariantsResponse,
    AgentLongTermMemory,
    AgentLongTermMemoryRequest,
    AgentMemoryResponse,
    AgentShortTermMemory,
    AgentStrategyResponse,
    AgentTaskStateResponse,
    AgentTaskTransitionError,
    AgentTokenUsage,
    AgentToolCall,
    AgentUserProfile,
    AgentUserProfileRequest,
    AgentWorkingMemory,
    AgentWorkingMemoryRequest,
    ContextStrategy,
    InvariantCategory,
    MemoryLayer,
)
from app.services.agent_context import ContextWindow, build_context
from app.services.agent_facts_extractor import FactsExtractor
from app.services.agent_history_compressor import HistoryCompressor
from app.services.agent_history_storage import (
    AgentHistoryStorage,
    Checkpoint,
    ConversationHistory,
    ConversationState,
    HistoryMessage,
    storage,
)
from app.services.agent_memory import (
    AgentLongTermMemoryStorage,
    LongTermMemory,
    MemoryLayers,
    ShortTermMemory,
    WorkingMemory,
    long_term_storage,
)
from app.services.agent_invariants import (
    AgentInvariantsStorage,
    Invariant,
    invariants_storage,
)
from app.services.agent_invariants import as_section as invariants_section
from app.services.agent_memory_router import MemoryRouter
from app.services.agent_task_state import (
    TaskStage,
    TaskState,
    TransitionRefusal,
    allowed_next,
    next_requirement,
)
from app.services.agent_task_tracker import TaskTracker
from app.services.agent_pipeline import PipelineRunner, pipeline_request, pipeline_section
from app.services.agent_tool_planner import ToolPlanner
from app.services.agent_tools import McpToolbox, results_section, unavailable_section
from app.services.digest import DigestStore, DigestTaskStorage, build_digest
from app.services.mcp_client import McpConnectionError, McpToolCallResult
from app.services.agent_user_profile import (
    AgentUserProfileStorage,
    UserProfile,
    user_profile_storage,
)
from app.services.agent_usage_log import AgentUsageLog, UsageRecord, usage_log
from app.services.gemini_service import count_tokens, generate_text_with_usage

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "You are a Japanese language learning assistant. Help the learner with "
    "whatever they asked about - a kanji, a word, grammar, or an example "
    "sentence - with a clear, focused answer aimed at a JLPT learner. "
    "Respond in Russian, unless the learner's message explicitly asks for "
    "a different language. Earlier turns of the conversation are included "
    "below for context - use them to resolve references like \"it\" or "
    "\"that\" back to whatever was being discussed."
)


class JapaneseLearningAgent:
    """Owns the whole conversation, not just a single turn.

    `run` loads the persisted conversation, builds a Gemini prompt from
    whichever strategy is active, asks Gemini, and - only once Gemini
    actually answers - appends both the learner's message and the answer to
    the current branch and saves it. A failed Gemini call raises straight
    through and nothing is persisted: that keeps the file free of orphaned
    user turns with no reply, which would otherwise misrepresent the
    conversation to the next prompt.

    The strategies differ only in what they put in front of the model
    (see agent_context.build_context, which is the single place that decides
    it); the conversation itself is stored the same way for all of them:

    * ``full`` - the whole conversation, every turn. The baseline.
    * ``summary`` - the running summary plus the messages still stored
      verbatim (HistoryCompressor).
    * ``sliding_window`` - only the newest few messages. Older ones stay on
      disk, so the same dialogue can still be replayed under another
      strategy, but they never reach the model.
    * ``sticky_facts`` - a key-value memory of goals, constraints,
      preferences and decisions, plus that same window. No summary.
    * ``branching`` - the current branch's whole conversation. Checkpoints
      and branches are available under every strategy; this one exists for
      when the point of the experiment is which branch is being talked on.
    * ``layered_memory`` - the three memory layers, each sent as its own
      labelled section (see agent_memory). This is the only strategy that
      routes what the learner says into working or long-term memory; under
      every other one those layers are left untouched, which is what keeps
      the earlier days' behaviour exactly as it was.

    The user profile (see agent_user_profile) is not one of these and is not
    a strategy: it says how the learner wants to be answered, so it goes
    into every request under every strategy, and the learner never has to
    repeat it. It sits between the context and the new message, which keeps
    it out of the history token count - it is not history, it is a setting.

    The task state (see agent_task_state) is the third thing that is not a
    strategy: it says where the work has got to - planning, execution,
    validation, done - and travels with the conversation so a task survives
    the app being closed. It is sent last, right before the new message, so
    "carry on from this step" is the final thing the model reads. Where a
    task stands is proposed by TaskTracker or asked for through the API, and
    in both cases it is accepted only if the state machine allows that move -
    the edge has to exist and the stage's own work has to be finished (a plan
    approved before execution, a validation passed before done). A refused
    move leaves the task where it was and records why, which is what the next
    answer explains.

    MCP tools (see agent_tools) are the fifth, and the only one that reaches
    outside the backend. Before answering, ToolPlanner shows the model the
    tools the MCP server listed and lets it decide whether the message needs
    a lookup; the calls it asks for go through the MCP client to the server,
    and what comes back is sent as a TOOL RESULTS block right before the
    message. A server that is down or an API that fails is reported to the
    model as a failed lookup - the answer is never blocked on it.

    The invariants (see agent_invariants) are the fourth: the rules of the
    project that must never be broken. They are not memory - no conversation
    writes them and clearing one cannot forget them - and they are sent on
    every request, ahead of the profile and the task state, together with
    the protocol for what to do when a request conflicts with one.

    Token usage: Gemini's generateContent/Interactions response reports one
    combined prompt-token count for everything sent (instructions + context
    + the new message), not a breakdown. To report context tokens and
    current-request tokens separately, the context portion of the prompt is
    measured on its own via Gemini's :countTokens endpoint (a real count,
    not an estimate), and current-request tokens is the remainder of the
    real total. Because it measures what was actually sent, it is also the
    number the strategy comparison turns on. Measuring is best-effort: if it
    fails, the chat answer is still returned - only the usage numbers that
    depend on it come back as null.
    """

    def __init__(
        self,
        history_storage: AgentHistoryStorage = storage,
        usage_log: AgentUsageLog = usage_log,
        compressor: HistoryCompressor | None = None,
        facts_extractor: FactsExtractor | None = None,
        compression_enabled: bool = AGENT_COMPRESSION_ENABLED,
        recent_messages_kept: int = AGENT_RECENT_MESSAGES_KEPT,
        long_term_memory: AgentLongTermMemoryStorage = long_term_storage,
        memory_router: MemoryRouter | None = None,
        profile_storage: AgentUserProfileStorage = user_profile_storage,
        task_tracker: TaskTracker | None = None,
        task_tracking_enabled: bool = AGENT_TASK_TRACKING_ENABLED,
        invariants: AgentInvariantsStorage = invariants_storage,
        toolbox: McpToolbox | None = None,
        tool_planner: ToolPlanner | None = None,
        tools_enabled: bool = AGENT_MCP_TOOLS_ENABLED,
    ) -> None:
        self._history = history_storage
        self._usage_log = usage_log
        self._compressor = compressor or HistoryCompressor()
        self._facts = facts_extractor or FactsExtractor()
        self._compression_enabled = compression_enabled
        self._recent_kept = recent_messages_kept
        self._long_term = long_term_memory
        self._memory_router = memory_router or MemoryRouter()
        self._profile = profile_storage
        self._task_tracker = task_tracker or TaskTracker()
        self._task_tracking_enabled = task_tracking_enabled
        self._invariants = invariants
        self._toolbox = toolbox or McpToolbox()
        self._tool_planner = tool_planner or ToolPlanner()
        self._tools_enabled = tools_enabled
        # The chain runs through the same toolbox as any single call: one
        # server, one client, three ordinary tools (Day 19).
        self._pipeline = PipelineRunner(self._toolbox)

    async def run(
        self,
        message: str,
        compression_enabled: bool | None = None,
        strategy: ContextStrategy | None = None,
    ) -> AgentChatResponse:
        """Answer one message on the current branch with the active strategy."""
        state = self._history.load()
        active = self._resolve_strategy(state, strategy, compression_enabled)
        history = state.current()
        long_term = self._long_term.load()
        window = build_context(active, history, self._recent_kept, self._memory_layers(state, long_term))
        history_prefix = self._build_history_prefix(window)
        # Before the prompt, because what a tool returns is part of it: the
        # model decides whether this message needs a lookup, and the answer
        # is written with the result in front of it.
        tool_section, tool_calls, tool_tokens = await self._use_tools(message, history.messages)
        # Read fresh on every request rather than cached at startup: the
        # profile is meant to be changed between messages and take effect on
        # the next one.
        prompt = self._build_prompt(
            message,
            history_prefix,
            invariants_section(self._invariants.load()),
            self._profile.load().as_section(),
            state.task_state.as_section(),
            tool_section,
        )

        # Let a Gemini failure here (including "context window exceeded")
        # propagate as-is - gemini_service already turns a non-200 response
        # into an HTTPException, so FastAPI reports it as a real error
        # response instead of a crash, and nothing below runs, so nothing
        # is persisted.
        generated = await generate_text_with_usage(prompt)

        history_tokens = await self._count_history_tokens(history_prefix)
        usage = self._build_usage(generated.input_tokens, generated.output_tokens, history_tokens)
        # What this request actually sent - captured before the new turn is
        # appended and before any summarising or fact-keeping rearranges it,
        # so the status and the token counts describe the same request.
        sent_context = AgentCompressionStatus(
            enabled=active is ContextStrategy.SUMMARY,
            summary_tokens=window.summary_tokens,
            messages_sent=len(window.messages),
        )
        summary_used = active is ContextStrategy.SUMMARY and bool(history.summary)

        history.messages.append({"role": "user", "content": message})
        history.messages.append({"role": "assistant", "content": generated.text})
        summarization_tokens = await self._compress_if_due(history, active)
        facts_tokens = await self._refresh_facts(history, message, active)
        memory_tokens = await self._route_memory(state, long_term, message, active)
        task_tokens = await self._track_task(state, message)
        self._history.save(state)

        self._record_usage(
            usage,
            active,
            sent_context,
            summary_used,
            summarization_tokens,
            facts_tokens,
            memory_tokens,
            task_tokens,
            tool_tokens,
        )

        return AgentChatResponse(
            response=generated.text,
            usage=usage,
            compression=sent_context,
            strategy=active.value,
            tool_calls=tool_calls,
        )

    # --- conversation ------------------------------------------------------

    def get_history(self) -> ConversationHistory:
        """The current branch's stored conversation."""
        return self._history.load().current()

    def clear_history(self) -> None:
        """Empty the conversation - every branch and checkpoint - while
        keeping the chosen strategy, which is a setting, not a message.

        This ends the conversation, so it takes the two layers scoped to one
        with it: short-term memory and the working memory of the task. It
        deliberately does not touch long-term memory, which is stored in its
        own file precisely so that "clear this dialogue" cannot mean "forget
        the learner".
        """
        state = self._history.load()
        self._history.save(ConversationState(strategy=state.strategy))

    def get_usage(self) -> list[UsageRecord]:
        return self._usage_log.load()

    # --- strategy and context ---------------------------------------------

    def set_strategy(self, strategy: ContextStrategy) -> AgentStrategyResponse:
        """Choose the strategy for the conversation. Persisted with it, so it
        survives a restart like everything else here."""
        state = self._history.load()
        state.strategy = strategy.value
        self._history.save(state)

        return AgentStrategyResponse(strategy=strategy.value)

    def get_context(self) -> AgentContextResponse:
        """What the next request would put in front of the model, built by
        the same code that builds the real prompt."""
        state = self._history.load()
        active = self._resolve_strategy(state, None, None)
        window = build_context(
            active, state.current(), self._recent_kept, self._memory_layers(state, self._long_term.load())
        )

        return AgentContextResponse(
            strategy=active.value,
            branch=state.current_branch,
            branches=sorted(state.branches),
            checkpoints=sorted(state.checkpoints),
            facts=window.facts,
            messages=[
                AgentHistoryMessage(role=entry["role"], content=entry["content"])
                for entry in window.messages
            ],
            context=self._build_history_prefix(window),
        )

    # --- memory layers -----------------------------------------------------

    def get_memory(self) -> AgentMemoryResponse:
        """All three layers as they stand, each read from where it is kept:
        short-term from the branch's messages, working memory from the
        conversation state, long-term from its own file."""
        state = self._history.load()

        return self._memory_response(state, self._long_term.load())

    def update_short_term(self, messages: list[AgentHistoryMessage]) -> AgentMemoryResponse:
        """Replace the current conversation. Writing the dialogue directly is
        what makes short-term memory updatable like the other two layers;
        ordinary turns still arrive through /agent/chat."""
        state = self._history.load()
        history = state.current()
        history.messages = [
            HistoryMessage(role=entry.role, content=entry.content) for entry in messages
        ]
        history.summary = ""
        history.summary_tokens = None
        self._history.save(state)

        return self._memory_response(state, self._long_term.load())

    def update_working_memory(self, request: AgentWorkingMemoryRequest) -> AgentMemoryResponse:
        """Update the task layer. A field left out of the request stays as it
        was - so a client can add constraints without restating the goals."""
        state = self._history.load()
        memory = state.working_memory
        state.working_memory = WorkingMemory(
            goals=memory.goals if request.goals is None else list(request.goals),
            requirements=memory.requirements if request.requirements is None else list(request.requirements),
            constraints=memory.constraints if request.constraints is None else list(request.constraints),
            decisions=memory.decisions if request.decisions is None else list(request.decisions),
        )
        self._history.save(state)

        return self._memory_response(state, self._long_term.load())

    def update_long_term_memory(self, request: AgentLongTermMemoryRequest) -> AgentMemoryResponse:
        """Update the learner layer, with the same leave-out-to-keep rule."""
        memory = self._long_term.load()
        updated = LongTermMemory(
            profile=memory.profile if request.profile is None else dict(request.profile),
            preferences=memory.preferences if request.preferences is None else list(request.preferences),
            decisions=memory.decisions if request.decisions is None else list(request.decisions),
            knowledge=memory.knowledge if request.knowledge is None else list(request.knowledge),
        )
        self._long_term.save(updated)

        return self._memory_response(self._history.load(), updated)

    def clear_memory(self, layer: MemoryLayer) -> AgentMemoryResponse:
        """Empty one layer and leave the other two alone.

        That the layers are stored apart is what makes this a three-line
        method: clearing long-term memory rewrites one file, clearing the
        other two rewrites part of another, and neither can reach into the
        layer it is not clearing.
        """
        state = self._history.load()

        if layer is MemoryLayer.LONG_TERM:
            self._long_term.clear()
            return self._memory_response(state, self._long_term.load())

        if layer is MemoryLayer.WORKING:
            state.working_memory = WorkingMemory()
        else:
            history = state.current()
            history.messages = []
            history.summary = ""
            history.summary_tokens = None

        self._history.save(state)

        return self._memory_response(state, self._long_term.load())

    # --- user profile ------------------------------------------------------

    def get_profile(self) -> AgentUserProfile:
        """The learner's answering preferences as they stand."""
        return self._profile_response(self._profile.load())

    def update_profile(self, request: AgentUserProfileRequest) -> AgentUserProfile:
        """Update the profile. A field left out of the request keeps its
        current value, so a client can change the level without restating
        the style."""
        profile = self._profile.load()
        updated = UserProfile(
            preferred_language=self._setting(profile.preferred_language, request.preferred_language),
            japanese_level=self._setting(profile.japanese_level, request.japanese_level),
            explanation_style=self._setting(profile.explanation_style, request.explanation_style),
            answer_format=self._setting(profile.answer_format, request.answer_format),
            translation_language=self._setting(profile.translation_language, request.translation_language),
            preferences=profile.preferences if request.preferences is None else list(request.preferences),
        )
        self._profile.save(updated)

        return self._profile_response(self._profile.load())

    def clear_profile(self) -> AgentUserProfile:
        """Unset every setting. Touches no memory layer and no message."""
        self._profile.clear()

        return self._profile_response(self._profile.load())

    # --- invariants --------------------------------------------------------

    def get_invariants(self) -> AgentInvariantsResponse:
        """Every rule the agent is bound by."""
        return self._invariants_response(self._invariants.load())

    def add_invariant(self, request: AgentInvariantRequest) -> AgentInvariantsResponse:
        """Add a rule. The id is generated here so a client never has to
        invent one; PUT is the way to choose it."""
        invariants = self._invariants.load()
        invariants.append(
            Invariant(
                id=self._next_invariant_id(invariants, request.category),
                category=request.category,
                rule=request.rule,
            )
        )
        self._invariants.save(invariants)

        return self._invariants_response(invariants)

    def set_invariant(self, invariant_id: str, request: AgentInvariantRequest) -> AgentInvariantsResponse:
        """Add or change the rule with this id, keeping its place in the
        list so the order the rules are read in does not shuffle."""
        identifier = self._require_name(invariant_id)
        updated = Invariant(id=identifier, category=request.category, rule=request.rule)
        invariants = self._invariants.load()
        existing = next((index for index, item in enumerate(invariants) if item.id == identifier), None)

        if existing is None:
            invariants.append(updated)
        else:
            invariants[existing] = updated

        self._invariants.save(invariants)

        return self._invariants_response(invariants)

    def delete_invariant(self, invariant_id: str) -> AgentInvariantsResponse:
        """Remove one rule. Deleting a rule that is not there is an error,
        not a silent success - a rule going missing is worth knowing about."""
        identifier = self._require_name(invariant_id)
        invariants = self._invariants.load()
        remaining = [item for item in invariants if item.id != identifier]

        if len(remaining) == len(invariants):
            raise HTTPException(status_code=404, detail=f"Invariant '{identifier}' does not exist")

        self._invariants.save(remaining)

        return self._invariants_response(remaining)

    # --- task state --------------------------------------------------------

    def get_task_state(self) -> AgentTaskStateResponse:
        """Where the task in progress has got to, and which moves are legal
        from here."""
        return self._task_response(self._history.load().task_state)

    def clear_task_state(self) -> AgentTaskStateResponse:
        """End the task. The conversation, the memory layers and the profile
        are all left alone - only the task is forgotten, which is what makes
        starting a new one from a clean stage possible without throwing away
        everything else."""
        state = self._history.load()
        state.task_state = TaskState()
        self._history.save(state)

        return self._task_response(state.task_state)

    def request_task_transition(
        self,
        stage: TaskStage,
        current_step: str = "",
        expected_action: str = "",
    ) -> AgentTaskStateResponse:
        """Move the task to a named stage, if it is allowed to go there.

        The same check the agent's own proposals go through, exposed so a
        client can drive the task deliberately. A refused move raises 409 with
        the whole refusal - where the task is, where it may go next, and what
        is missing - and leaves the stage, the step and the expected action
        exactly as they were. What is written in that case is only the reason:
        the screen keeps showing it, and the agent explains it on the next
        message rather than silently ignoring what was asked.
        """
        state = self._history.load()
        outcome = state.task_state.with_update(stage, current_step, expected_action)
        state.task_state = outcome.state
        self._history.save(state)

        if outcome.refusal is not None:
            raise HTTPException(status_code=409, detail=outcome.refusal.as_json())

        return self._task_response(outcome.state)

    def approve_task_plan(self, plan: str) -> AgentTaskStateResponse:
        """Record the plan the task will be executed by.

        Only while the task is planning: the plan is what planning produces,
        and a plan approved from anywhere else would be a way round the very
        condition it exists to satisfy.
        """
        state = self._history.load()

        if state.task_state.stage is not TaskStage.PLANNING:
            raise HTTPException(
                status_code=409,
                detail=self._stage_required(state.task_state, TaskStage.PLANNING, "a plan is approved"),
            )

        state.task_state = state.task_state.with_plan(plan)
        self._history.save(state)

        return self._task_response(state.task_state)

    def record_task_validation(self, passed: bool, notes: str = "") -> AgentTaskStateResponse:
        """Record how validation went. Only while the task is validating.

        A failed check is recorded as well as a passing one: "checked, not
        right yet" is not "not checked", and neither of them opens the way to
        done.
        """
        state = self._history.load()

        if state.task_state.stage is not TaskStage.VALIDATION:
            raise HTTPException(
                status_code=409,
                detail=self._stage_required(
                    state.task_state, TaskStage.VALIDATION, "validation is recorded"
                ),
            )

        state.task_state = state.task_state.with_validation(passed, notes)
        self._history.save(state)

        return self._task_response(state.task_state)

    @staticmethod
    def _stage_required(task_state: TaskState, stage: TaskStage, action: str) -> dict[str, object]:
        """A refusal in the same shape as a refused transition, for the two
        records that only one stage may make."""
        return {
            "message": (
                f"The task is in '{task_state.stage.value}': it has to be in "
                f"'{stage.value}' before {action}."
            ),
            "current_stage": task_state.stage.value,
            "requested_stage": None,
            "required_next": allowed_next(task_state.stage),
            "unmet_condition": f"the task is not in '{stage.value}'",
        }

    # --- periodic digest ---------------------------------------------------

    @staticmethod
    def get_digest() -> AgentDigestResponse:
        """The digest the periodic task has been building.

        Read straight from the store rather than through MCP: this is a
        readout the screen refreshes after every message, and starting a
        server subprocess for it would cost a second each time. It is the
        same aggregate the tool returns - one implementation, in
        app.services.digest - so the screen and the agent cannot disagree.
        """
        task = DigestTaskStorage().latest()

        if task is None:
            return AgentDigestResponse(
                found=False,
                summary="No periodic digest has been created yet.",
            )

        return AgentDigestResponse(found=True, **build_digest(task, DigestStore().state_of(task.id)))

    # --- checkpoints and branches -----------------------------------------

    def create_checkpoint(self, name: str | None = None) -> AgentCheckpointResponse:
        """Capture the current branch as it stands, so branches can be forked
        from this exact point later."""
        state = self._history.load()
        history = state.current()
        checkpoint_name = self._require_name(name) if name is not None else self._next_checkpoint_name(state)

        if checkpoint_name in state.checkpoints:
            raise HTTPException(status_code=409, detail=f"Checkpoint '{checkpoint_name}' already exists")

        state.checkpoints[checkpoint_name] = Checkpoint(
            branch=state.current_branch,
            history=deepcopy(history),
        )
        self._history.save(state)

        return AgentCheckpointResponse(
            name=checkpoint_name,
            branch=state.current_branch,
            messages=len(history.messages),
        )

    def create_branch(self, name: str, checkpoint: str) -> AgentBranchResponse:
        """Fork a branch from a checkpoint.

        The branch starts as a copy of what the checkpoint captured, so two
        branches forked from the same one start identical and then never
        touch each other again. Creating a branch deliberately does not
        switch to it - that is what makes forking a second one from the same
        checkpoint straightforward.
        """
        state = self._history.load()
        branch_name = self._require_name(name)

        if branch_name in state.branches:
            raise HTTPException(status_code=409, detail=f"Branch '{branch_name}' already exists")

        saved = state.checkpoints.get(self._require_name(checkpoint))

        if saved is None:
            raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint}' does not exist")

        state.branches[branch_name] = deepcopy(saved.history)
        self._history.save(state)

        return self._branch_response(state)

    def switch_branch(self, name: str) -> AgentBranchResponse:
        """Talk on another branch. Its own messages and facts come back with
        it; nothing from the branch being left behind comes along."""
        state = self._history.load()
        branch_name = self._require_name(name)

        if branch_name not in state.branches:
            raise HTTPException(status_code=404, detail=f"Branch '{branch_name}' does not exist")

        state.current_branch = branch_name
        self._history.save(state)

        return self._branch_response(state)

    # --- internals ---------------------------------------------------------

    def _resolve_strategy(
        self,
        state: ConversationState,
        requested: ContextStrategy | None,
        compression_enabled: bool | None,
    ) -> ContextStrategy:
        """Which strategy answers this request.

        A strategy named in the request wins, because steering a single
        request is what makes an experiment possible. Otherwise the one
        chosen through PUT /agent/strategy and persisted with the
        conversation. Only when there is neither does the older compression
        flag decide, which is what keeps clients that know nothing about
        strategies working exactly as they did.
        """
        if requested is not None:
            return requested

        if state.strategy:
            try:
                return ContextStrategy(state.strategy)
            except ValueError:
                logger.warning("Stored agent strategy %r is unknown; ignoring it", state.strategy)

        use_compression = self._compression_enabled if compression_enabled is None else compression_enabled

        return ContextStrategy.SUMMARY if use_compression else ContextStrategy.FULL

    async def _compress_if_due(self, history: ConversationHistory, strategy: ContextStrategy) -> int:
        """Fold aged-out messages into the summary once enough have piled up.

        Best-effort, like counting tokens: the learner's turn has already
        succeeded and is about to be saved, so a failed summarisation is
        logged and left for the next turn to retry rather than turned into
        an error for an answer that was perfectly fine. Nothing is dropped
        in the meantime - the messages simply stay verbatim a while longer.
        """
        if strategy is not ContextStrategy.SUMMARY or not self._compressor.needs_compression(history.messages):
            return 0

        try:
            result = await self._compressor.compress(history.summary, history.messages)
        except Exception as error:  # noqa: BLE001 - see docstring
            logger.warning("Could not compress agent history: %s", error)
            return 0

        history.summary = result.summary
        history.messages = result.messages
        history.summary_tokens = result.summary_tokens

        return result.tokens_used

    async def _refresh_facts(
        self,
        history: ConversationHistory,
        message: str,
        strategy: ContextStrategy,
    ) -> int:
        """Sticky Facts keeps its key-value memory current after every
        learner message. Best-effort for the same reason as summarising: a
        failed update leaves the previous facts in place and is retried on
        the next message rather than failing an answer that already worked.
        """
        if strategy is not ContextStrategy.STICKY_FACTS:
            return 0

        try:
            update = await self._facts.update(history.facts, message)
        except Exception as error:  # noqa: BLE001 - see docstring
            logger.warning("Could not update agent facts: %s", error)
            return 0

        history.facts = update.facts

        return update.tokens_used

    async def _route_memory(
        self,
        state: ConversationState,
        long_term: LongTermMemory,
        message: str,
        strategy: ContextStrategy,
    ) -> int:
        """Put what the learner just said into the layer it belongs to.

        Only the layered-memory strategy does this: under the other
        strategies the memory layers are read but never written, so nothing
        that worked before starts behaving differently.

        The router names the layers that change and gives each one back in
        full; a layer it does not name is left exactly as it was, so an
        ordinary question writes nothing. Best-effort, like summarising and
        fact-keeping: the answer is already the learner's, and a failed
        routing keeps the previous memory and is retried on the next
        message rather than failing a good answer.
        """
        if strategy is not ContextStrategy.LAYERED_MEMORY:
            return 0

        try:
            routing = await self._memory_router.route(message, state.working_memory, long_term)
        except Exception as error:  # noqa: BLE001 - see docstring
            logger.warning("Could not route agent memory: %s", error)
            return 0

        if routing.working is not None:
            state.working_memory = routing.working

        if routing.long_term is not None:
            # Saved on its own, to its own file: long-term memory outlives
            # the conversation being saved below, and outlives it being
            # cleared.
            self._long_term.save(routing.long_term)
            long_term.profile = routing.long_term.profile
            long_term.preferences = routing.long_term.preferences
            long_term.decisions = routing.long_term.decisions
            long_term.knowledge = routing.long_term.knowledge

        return routing.tokens_used

    async def _use_tools(
        self,
        message: str,
        history: list[dict[str, str]],
    ) -> tuple[str, list[AgentToolCall], int]:
        """Let the model decide whether this message needs an MCP tool, run
        the calls it asks for, and return the TOOL RESULTS block, the calls
        as the response reports them, and what deciding cost.

        Never raises. A server that cannot be listed is told to the model as
        "the dictionary could not be checked"; a plan that fails means no
        tools; a call that fails is reported as a failed lookup. Every one
        of them still ends in an answer.
        """
        if not self._tools_enabled:
            return "", [], 0

        try:
            tools = await self._toolbox.tools()
        except McpConnectionError as error:
            logger.warning("MCP tools unavailable: %s", error)
            return unavailable_section(), [], 0

        try:
            plan = await self._tool_planner.plan(message, tools, history[-2:])
        except Exception as error:  # noqa: BLE001 - best-effort, see docstring
            logger.warning("Could not plan tool calls: %s", error)
            return "", [], 0

        # A plan that asks for a summary or for something saved is not three
        # separate calls but one chain, and only the chain can fill in the
        # arguments the planner could not know (Day 19).
        request = pipeline_request(plan.calls)

        if request is not None:
            run = await self._pipeline.run(request)
            logger.info(
                "Agent ran the MCP pipeline %s for '%s': %s",
                " -> ".join(run.requested),
                run.query,
                f"saved as {run.saved_as}" if run.completed else "stopped early",
            )

            return pipeline_section(run), self._reported(run.results), plan.tokens_used

        results = [await self._toolbox.call(call.tool, call.arguments) for call in plan.calls]

        for result in results:
            logger.info(
                "Agent used MCP tool %s(%s): %s",
                result.name,
                result.arguments,
                "ok" if result.ok else f"failed - {result.error}",
            )

        return results_section(results), self._reported(results), plan.tokens_used

    @staticmethod
    def _reported(results: Sequence[McpToolCallResult]) -> list[AgentToolCall]:
        """The calls as the response carries them back to the screen."""
        return [
            AgentToolCall(
                tool=result.name,
                arguments=result.arguments,
                ok=result.ok,
                result=result.data,
                error=result.error,
            )
            for result in results
        ]

    async def _track_task(self, state: ConversationState, message: str) -> int:
        """Move the task on, if the learner's message calls for it.

        The tracker only proposes; TaskState.with_update accepts the move
        only when the state machine allows it, so a proposal to skip a stage
        changes nothing but the recorded reason - which the next answer
        explains to the learner. Best-effort, like every other extra call: a
        failed update leaves the task exactly where it was and is retried on
        the next message rather than failing an answer that already worked.
        """
        if not self._task_tracking_enabled:
            return 0

        try:
            update = await self._task_tracker.track(message, state.task_state)
        except Exception as error:  # noqa: BLE001 - see docstring
            logger.warning("Could not track the task state: %s", error)
            return 0

        state.task_state = state.task_state.with_update(
            update.stage,
            update.current_step,
            update.expected_action,
            plan=update.plan,
            validation_passed=update.validation_passed,
        ).state

        return update.tokens_used

    @staticmethod
    def _invariants_response(invariants: list[Invariant]) -> AgentInvariantsResponse:
        return AgentInvariantsResponse(
            invariants=[
                AgentInvariant(id=item.id, category=item.category.value, rule=item.rule)
                for item in invariants
            ]
        )

    @staticmethod
    def _next_invariant_id(invariants: list[Invariant], category: InvariantCategory) -> str:
        """A readable id nobody has to think about: the category and the
        first number not already taken."""
        taken = {item.id for item in invariants}
        index = 1

        while f"{category.value}-{index}" in taken:
            index += 1

        return f"{category.value}-{index}"

    @staticmethod
    def _task_response(task_state: TaskState) -> AgentTaskStateResponse:
        return AgentTaskStateResponse(
            task_stage=task_state.stage.value,
            current_step=task_state.current_step,
            expected_action=task_state.expected_action,
            allowed_next=allowed_next(task_state.stage),
            plan=task_state.plan,
            validation_passed=task_state.validation_passed,
            validation_note=task_state.validation_note,
            next_requirement=next_requirement(task_state),
            blocked=JapaneseLearningAgent._blocked_response(task_state.blocked),
        )

    @staticmethod
    def _blocked_response(refusal: TransitionRefusal | None) -> AgentTaskTransitionError | None:
        return AgentTaskTransitionError(**refusal.as_json()) if refusal is not None else None

    @staticmethod
    def _memory_layers(state: ConversationState, long_term: LongTermMemory) -> MemoryLayers:
        """The three layers assembled from the three places they are kept."""
        return MemoryLayers(
            short_term=ShortTermMemory(messages=state.current().messages),
            working=state.working_memory,
            long_term=long_term,
        )

    @staticmethod
    def _memory_response(state: ConversationState, long_term: LongTermMemory) -> AgentMemoryResponse:
        working = state.working_memory

        return AgentMemoryResponse(
            short_term=AgentShortTermMemory(
                messages=[
                    AgentHistoryMessage(role=entry["role"], content=entry["content"])
                    for entry in state.current().messages
                ]
            ),
            working=AgentWorkingMemory(
                goals=working.goals,
                requirements=working.requirements,
                constraints=working.constraints,
                decisions=working.decisions,
            ),
            long_term=AgentLongTermMemory(
                profile=long_term.profile,
                preferences=long_term.preferences,
                decisions=long_term.decisions,
                knowledge=long_term.knowledge,
            ),
        )

    def _record_usage(
        self,
        usage: AgentTokenUsage,
        strategy: ContextStrategy,
        sent_context: AgentCompressionStatus,
        summary_used: bool,
        summarization_tokens: int,
        facts_tokens: int,
        memory_tokens: int = 0,
        task_tokens: int = 0,
        tool_tokens: int = 0,
    ) -> None:
        """Persist what this request cost and which strategy produced it, so
        the strategies can be compared afterwards. Instrumentation only - a
        failure here must never sink an answer the learner already has."""
        record: UsageRecord = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy.value,
            "compression_enabled": sent_context.enabled,
            "messages_sent": sent_context.messages_sent,
            "summary_used": summary_used,
            "current_request_tokens": usage.current_request_tokens,
            "history_tokens": usage.history_tokens,
            "response_tokens": usage.response_tokens,
            "total_tokens": usage.total_tokens,
            "summarization_tokens": summarization_tokens,
            "facts_tokens": facts_tokens,
            "memory_tokens": memory_tokens,
            "task_tokens": task_tokens,
            "tool_tokens": tool_tokens,
        }

        try:
            self._usage_log.append(record)
        except Exception as error:  # noqa: BLE001 - see docstring
            logger.warning("Could not record agent token usage: %s", error)

    @staticmethod
    def _branch_response(state: ConversationState) -> AgentBranchResponse:
        return AgentBranchResponse(branch=state.current_branch, branches=sorted(state.branches))

    @staticmethod
    def _next_checkpoint_name(state: ConversationState) -> str:
        index = len(state.checkpoints) + 1

        while f"cp-{index}" in state.checkpoints:
            index += 1

        return f"cp-{index}"

    @staticmethod
    def _require_name(name: str) -> str:
        cleaned = name.strip()

        if not cleaned:
            raise HTTPException(status_code=422, detail="Name must not be empty")

        return cleaned

    @staticmethod
    async def _count_history_tokens(history_prefix: str) -> int | None:
        if not history_prefix:
            return 0

        try:
            return await count_tokens(history_prefix)
        except Exception as error:  # noqa: BLE001 - a usage-tracking hiccup must not sink a successful answer
            logger.warning("Could not count history tokens: %s", error)
            return None

    @staticmethod
    def _build_usage(
        prompt_tokens: int | None,
        response_tokens: int | None,
        history_tokens: int | None,
    ) -> AgentTokenUsage:
        current_request_tokens = (
            max(prompt_tokens - history_tokens, 0)
            if prompt_tokens is not None and history_tokens is not None
            else None
        )
        total_tokens = (
            prompt_tokens + response_tokens
            if prompt_tokens is not None and response_tokens is not None
            else None
        )

        return AgentTokenUsage(
            current_request_tokens=current_request_tokens,
            history_tokens=history_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _setting(current: str, requested: str | None) -> str:
        return current if requested is None else requested.strip()

    @staticmethod
    def _profile_response(profile: UserProfile) -> AgentUserProfile:
        return AgentUserProfile(
            preferred_language=profile.preferred_language,
            japanese_level=profile.japanese_level,
            explanation_style=profile.explanation_style,
            answer_format=profile.answer_format,
            translation_language=profile.translation_language,
            preferences=profile.preferences,
        )

    @staticmethod
    def _build_history_prefix(window: ContextWindow) -> str:
        """Everything that comes before the new message in the prompt - the
        exact text token-counted as "history". Empty when the strategy has
        nothing to add, so the very first turn reports zero context tokens.
        """
        if not window.sections:
            return ""

        return "\n\n".join([_INSTRUCTIONS, *window.sections])

    @staticmethod
    def _build_prompt(
        message: str,
        history_prefix: str,
        invariants: str = "",
        profile_section: str = "",
        task_section: str = "",
        tool_section: str = "",
    ) -> str:
        """Instructions, then what is remembered, then what may never be
        broken, then who the learner is, then where the task stands, then
        what they just asked.

        The last three sit after the context and before the message on
        purpose: they are what the question lands on, and they stay out of
        the text counted as history - history is what the conversation
        produced, while these are rules, a setting and a position. They are
        ordered from the hardest to the softest: the invariants bind
        whatever is asked, the profile is a default the learner's own
        message can override, and the task state says where to carry on
        from, so it ends up next to the question. What the tools looked up
        for this very message comes last of all, directly above it: it is the
        freshest and most specific thing the answer rests on.
        """
        opening = history_prefix or _INSTRUCTIONS
        request_label = "Learner's new request" if history_prefix else "Learner's request"
        sections = [opening, invariants, profile_section, task_section, tool_section]

        return "\n\n".join([*(section for section in sections if section), f"{request_label}: {message}"])


agent = JapaneseLearningAgent()
