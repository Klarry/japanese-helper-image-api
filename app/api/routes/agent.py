from fastapi import APIRouter, HTTPException

from app.schemas.agent import (
    AgentBranchRequest,
    AgentBranchResponse,
    AgentBranchSwitchRequest,
    AgentChatRequest,
    AgentChatResponse,
    AgentCheckpointRequest,
    AgentCheckpointResponse,
    AgentContextResponse,
    AgentDigestResponse,
    AgentDocumentIndexResponse,
    AgentHistoryResponse,
    AgentInvariantRequest,
    AgentInvariantsResponse,
    AgentLongTermMemoryRequest,
    AgentMemoryResponse,
    AgentRagChunk,
    AgentRagRequest,
    AgentRagResponse,
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
from app.services.document_index import read_summary
from app.services.rag_agent import RagAgent
from app.services.vector_index import VectorIndexError
from app.services.japanese_learning_agent import agent

router = APIRouter()
# One agent for both modes, built once: its retriever loads the index lazily,
# so importing this module costs nothing when nobody asks a question.
rag_agent = RagAgent()


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


@router.get("/agent/digest")
async def agent_digest() -> AgentDigestResponse:
    """What the periodic digest task has collected so far: how many runs,
    when the last one was, how many words, and a line of summary. Reading it
    changes nothing - the runs happen on their own schedule."""
    return agent.get_digest()


@router.post("/agent/rag")
async def agent_rag(request: AgentRagRequest) -> AgentRagResponse:
    """Answer a question about the project's own documents (Day 22).

    ``use_rag=true`` retrieves from the local index first and answers from
    what it found, naming the files; ``use_rag=false`` asks the same model
    the same question with nothing in front of it. One agent, one endpoint,
    the mode decides whether there is a retrieval step in between.
    """
    try:
        answer = await rag_agent.ask(request.question, use_rag=request.use_rag, top_k=request.top_k)
    except VectorIndexError as error:
        # Not an error in the request: nobody has built the index yet.
        raise HTTPException(
            status_code=503,
            detail=f"The document index is not available: {error}",
        ) from error

    return AgentRagResponse(
        answer=answer.answer,
        rag_enabled=answer.rag_enabled,
        sources=answer.sources,
        retrieved_chunks=[
            AgentRagChunk(
                chunk_id=chunk.chunk_id,
                file=chunk.file,
                section=chunk.section,
                score=round(chunk.score, 4),
            )
            for chunk in answer.retrieved_chunks
        ],
        top_k=answer.top_k,
        embedding_model=answer.embedding_model,
        retrieval_seconds=round(answer.retrieval_seconds, 3),
        llm_seconds=round(answer.llm_seconds, 3),
    )


@router.get("/agent/documents")
async def agent_document_index() -> AgentDocumentIndexResponse:
    """What the local document index holds: how many documents went in, how
    many chunks each strategy made of them, and what embedded them. Built by
    `python -m app.index_documents`; this only reads the report."""
    return AgentDocumentIndexResponse(**read_summary().as_dict())


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
