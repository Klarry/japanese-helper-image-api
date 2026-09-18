from fastapi import APIRouter

from app.schemas.agent import (
    AgentBranchRequest,
    AgentBranchResponse,
    AgentBranchSwitchRequest,
    AgentChatRequest,
    AgentChatResponse,
    AgentCheckpointRequest,
    AgentCheckpointResponse,
    AgentContextResponse,
    AgentHistoryResponse,
    AgentInvariantRequest,
    AgentInvariantsResponse,
    AgentLongTermMemoryRequest,
    AgentMemoryResponse,
    AgentShortTermMemoryRequest,
    AgentStrategyRequest,
    AgentStrategyResponse,
    AgentTaskPlanRequest,
    AgentTaskStateResponse,
    AgentTaskTransitionRequest,
    AgentTaskValidationRequest,
    AgentUsageResponse,
    AgentUserProfile,
    AgentUserProfileRequest,
    AgentWorkingMemoryRequest,
    MemoryLayer,
)
from app.services.japanese_learning_agent import agent

router = APIRouter()


@router.post("/agent/chat")
async def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    return await agent.run(request.message, request.compression_enabled, request.strategy)


@router.get("/agent/history")
async def agent_history() -> AgentHistoryResponse:
    history = agent.get_history()
    return AgentHistoryResponse(messages=history.messages, summary=history.summary)


@router.delete("/agent/history")
async def clear_agent_history() -> AgentHistoryResponse:
    agent.clear_history()
    return AgentHistoryResponse(messages=[])


@router.put("/agent/strategy")
async def set_agent_strategy(request: AgentStrategyRequest) -> AgentStrategyResponse:
    """Choose how much of the conversation the next requests will send."""
    return agent.set_strategy(request.strategy)


@router.get("/agent/context")
async def agent_context() -> AgentContextResponse:
    """The context the next request would send, plus the branch and
    checkpoints it would send it from."""
    return agent.get_context()


@router.post("/agent/checkpoint")
async def create_agent_checkpoint(request: AgentCheckpointRequest | None = None) -> AgentCheckpointResponse:
    """Capture the current branch so branches can be forked from this point."""
    return agent.create_checkpoint(request.name if request else None)


@router.post("/agent/branch")
async def create_agent_branch(request: AgentBranchRequest) -> AgentBranchResponse:
    """Fork a branch from a checkpoint. Does not switch to it."""
    return agent.create_branch(request.name, request.checkpoint)


@router.put("/agent/branch")
async def switch_agent_branch(request: AgentBranchSwitchRequest) -> AgentBranchResponse:
    """Talk on another branch from now on."""
    return agent.switch_branch(request.name)


@router.get("/agent/memory")
async def agent_memory() -> AgentMemoryResponse:
    """All three memory layers, each under its own key: the current
    conversation, the current task, and what is remembered about the
    learner across conversations."""
    return agent.get_memory()


@router.put("/agent/memory/short_term")
async def update_short_term_memory(request: AgentShortTermMemoryRequest) -> AgentMemoryResponse:
    """Replace the current conversation."""
    return agent.update_short_term(request.messages)


@router.put("/agent/memory/working")
async def update_working_memory(request: AgentWorkingMemoryRequest) -> AgentMemoryResponse:
    """Update the current task's goals, requirements, constraints and
    decisions. Fields left out keep their current value."""
    return agent.update_working_memory(request)


@router.put("/agent/memory/long_term")
async def update_long_term_memory(request: AgentLongTermMemoryRequest) -> AgentMemoryResponse:
    """Update what is remembered about the learner between conversations."""
    return agent.update_long_term_memory(request)


@router.delete("/agent/memory/{layer}")
async def clear_memory_layer(layer: MemoryLayer) -> AgentMemoryResponse:
    """Empty one layer - short_term, working or long_term - and leave the
    other two exactly as they are."""
    return agent.clear_memory(layer)


@router.get("/agent/profile")
async def agent_profile() -> AgentUserProfile:
    """How the learner wants to be answered. Read on every chat request, so
    the preferences never have to be repeated in a message."""
    return agent.get_profile()


@router.put("/agent/profile")
async def update_agent_profile(request: AgentUserProfileRequest) -> AgentUserProfile:
    """Update the profile. Fields left out keep their current value."""
    return agent.update_profile(request)


@router.delete("/agent/profile")
async def clear_agent_profile() -> AgentUserProfile:
    """Unset every setting. Leaves all three memory layers untouched."""
    return agent.clear_profile()


@router.get("/agent/invariants")
async def agent_invariants() -> AgentInvariantsResponse:
    """Every rule the agent is bound by. Read on every chat request, so a
    rule added here applies to the next message."""
    return agent.get_invariants()


@router.post("/agent/invariants")
async def add_agent_invariant(request: AgentInvariantRequest) -> AgentInvariantsResponse:
    """Add a rule. The id is generated from the category."""
    return agent.add_invariant(request)


@router.put("/agent/invariants/{invariant_id}")
async def set_agent_invariant(
    invariant_id: str,
    request: AgentInvariantRequest,
) -> AgentInvariantsResponse:
    """Add or change the rule with this id, keeping its place in the list."""
    return agent.set_invariant(invariant_id, request)


@router.delete("/agent/invariants/{invariant_id}")
async def delete_agent_invariant(invariant_id: str) -> AgentInvariantsResponse:
    """Remove one rule. Unknown ids are reported rather than ignored."""
    return agent.delete_invariant(invariant_id)


@router.get("/agent/task")
async def agent_task_state() -> AgentTaskStateResponse:
    """Where the task in progress has got to, and which stages it may move
    to from here."""
    return agent.get_task_state()


@router.delete("/agent/task")
async def clear_agent_task_state() -> AgentTaskStateResponse:
    """End the task. The conversation, the memory layers and the profile are
    left untouched."""
    return agent.clear_task_state()


@router.post("/agent/task/transition")
async def request_agent_task_transition(
    request: AgentTaskTransitionRequest,
) -> AgentTaskStateResponse:
    """Move the task to a named stage.

    409 when the move is not allowed, with the current stage, the stage that
    may come next and the condition that is not met yet. The task itself does
    not move.
    """
    return agent.request_task_transition(
        request.task_stage, request.current_step, request.expected_action
    )


@router.post("/agent/task/plan")
async def approve_agent_task_plan(request: AgentTaskPlanRequest) -> AgentTaskStateResponse:
    """Approve the plan the task will be executed by - the condition for
    leaving planning. 409 unless the task is planning."""
    return agent.approve_task_plan(request.plan)


@router.post("/agent/task/validation")
async def record_agent_task_validation(
    request: AgentTaskValidationRequest,
) -> AgentTaskStateResponse:
    """Record how validation went - the condition for reaching done. 409
    unless the task is validating."""
    return agent.record_task_validation(request.passed, request.notes)


@router.get("/agent/usage")
async def agent_usage() -> AgentUsageResponse:
    """Token usage of every recorded chat request, tagged with the strategy
    that produced it - the raw material for comparing them."""
    return AgentUsageResponse(entries=agent.get_usage())
