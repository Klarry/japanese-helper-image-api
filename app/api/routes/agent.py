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
    AgentStrategyRequest,
    AgentStrategyResponse,
    AgentUsageResponse,
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


@router.get("/agent/usage")
async def agent_usage() -> AgentUsageResponse:
    """Token usage of every recorded chat request, tagged with the strategy
    that produced it - the raw material for comparing them."""
    return AgentUsageResponse(entries=agent.get_usage())
