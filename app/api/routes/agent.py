from fastapi import APIRouter

from app.schemas.agent import AgentChatRequest, AgentChatResponse
from app.services.japanese_learning_agent import agent

router = APIRouter()


@router.post("/agent/chat")
async def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    response_text = await agent.run(request.message)
    return AgentChatResponse(response=response_text)
