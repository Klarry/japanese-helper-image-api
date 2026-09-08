from fastapi import APIRouter

from app.schemas.agent import AgentChatRequest, AgentChatResponse, AgentHistoryResponse
from app.services.japanese_learning_agent import agent

router = APIRouter()


@router.post("/agent/chat")
async def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    response_text = await agent.run(request.message)
    return AgentChatResponse(response=response_text)


@router.get("/agent/history")
async def agent_history() -> AgentHistoryResponse:
    return AgentHistoryResponse(messages=agent.get_history())


@router.delete("/agent/history")
async def clear_agent_history() -> AgentHistoryResponse:
    agent.clear_history()
    return AgentHistoryResponse(messages=[])
