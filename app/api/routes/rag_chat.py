"""Day 28: the existing RAG pipeline, with the generator chosen per request.

One endpoint, two providers, and exactly one retrieval path. The FAISS
index of Day 21, the filter and reranker of Day 23, the relevance gate and
the citation validation of Day 24 all run the same way whichever model is
named - the provider only decides who is handed the finished prompt.

So the comparison this day is for is a fair one: the local half of the work
is literally the same code, and the difference in the numbers is the
difference between the two models.
"""

import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.rag_chat import (
    RagChatCitation,
    RagChatRequest,
    RagChatResponse,
    RagChatSource,
    RagChatTiming,
    RagChatUnavailable,
)
from app.services import llm_provider
from app.services.rag_agent import ENHANCED, RagAgent
from app.services.rag_response import LOW

router = APIRouter()
logger = logging.getLogger(__name__)

#: A fresh agent per request, because the provider is per request. Building
#: one is cheap - the index and its vectors are loaded once by the retriever
#: and shared, which is the expensive part.
def agent_for(provider_name: str) -> RagAgent:
    provider = llm_provider.provider_for(provider_name)

    return RagAgent(provider=provider)


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))


@router.post("/rag/chat")
async def rag_chat(request: RagChatRequest) -> RagChatResponse:
    started = time.perf_counter()
    agent = agent_for(request.provider)

    try:
        answer = await agent.ask(request.message, mode=ENHANCED)
    except llm_provider.ProviderUnavailable as error:
        # Not a fallback. The question was asked of one model; if that model
        # cannot answer, the honest reply is that it could not.
        logger.warning(
            "[RAG] provider=%s unavailable (%s) - refusing rather than falling back",
            error.provider,
            error.reason,
        )
        body = RagChatUnavailable(
            error="Local LLM is unavailable",
            provider=error.provider,
            reason=error.reason,
            detail=error.message,
        )

        return JSONResponse(status_code=503, content=body.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    total = time.perf_counter() - started
    cited = answer.cited
    enhanced = answer.enhanced
    retrieved = len(enhanced.retrieval.chunks) if enhanced else 0
    final = len(enhanced.final) if enhanced else 0

    logger.info(
        "[RAG] provider=%s · model=%s · retrieval=%dms · generation=%dms · total=%dms "
        "· retrieved=%d · final=%d · status=%s",
        agent.provider_name,
        agent.model_name,
        _ms(answer.retrieval_seconds),
        _ms(answer.llm_seconds),
        _ms(total),
        retrieved,
        final,
        cited.rag_status if cited else "",
    )

    return RagChatResponse(
        answer=answer.answer,
        sources=[
            RagChatSource(
                source=source.source,
                file=source.file,
                section=source.section,
                chunk_id=source.chunk_id,
            )
            for source in (cited.sources if cited else ())
        ],
        citations=[
            RagChatCitation(
                quote=citation.quote,
                source=citation.source,
                section=citation.section,
                chunk_id=citation.chunk_id,
            )
            for citation in (cited.citations if cited else ())
        ],
        confidence=cited.confidence if cited else LOW,
        rag_status=cited.rag_status if cited else "",
        provider=agent.provider_name,
        model=agent.model_name,
        retrieved_count=retrieved,
        final_count=final,
        best_relevance=round(cited.best_relevance, 4) if cited else 0.0,
        answer_threshold=round(cited.answer_threshold, 4) if cited else 0.0,
        timing=RagChatTiming(
            retrieval_ms=_ms(answer.retrieval_seconds),
            generation_ms=_ms(answer.llm_seconds),
            total_ms=_ms(total),
        ),
    )
