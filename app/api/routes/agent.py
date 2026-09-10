from fastapi import APIRouter

from app.schemas.agent import (
    AgentChatRequest,
    AgentChatResponse,
    AgentHistoryResponse,
    AgentUsageResponse,
)
from app.services.japanese_learning_agent import agent

router = APIRouter()


@router.post("/agent/chat")
async def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    return await agent.run(request.message, request.compression_enabled)


@router.get("/agent/history")
async def agent_history() -> AgentHistoryResponse:
    history = agent.get_history()
    return AgentHistoryResponse(messages=history.messages, summary=history.summary)


@router.delete("/agent/history")
async def clear_agent_history() -> AgentHistoryResponse:
    agent.clear_history()
    return AgentHistoryResponse(messages=[])


@router.get("/agent/usage")
async def agent_usage() -> AgentUsageResponse:
    """Token usage of every recorded chat request - the raw material for
    comparing spend with and without compression."""
    return AgentUsageResponse(entries=agent.get_usage())
