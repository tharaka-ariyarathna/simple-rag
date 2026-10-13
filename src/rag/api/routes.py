import logging
from dataclasses import asdict
from fastapi import APIRouter, Depends, HTTPException, Request, status
from rag.api.schemas import ChatRequest, ChatResponse
from rag.constants import LLM_TIMEOUT_DETAIL, LLM_UNAVAILABLE_DETAIL
from rag.llm.base import LlmTimeoutError, LlmUnavailableError
from rag.service.rag_service import RagService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service  # created once at startup (see main.py lifespan)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Plain def (not async): the RAG flow is blocking (embedding, psycopg, LLM call),
# so FastAPI runs it in a worker thread instead of freezing the event loop
@router.post("/chat", response_model=ChatResponse)
def chat(chat_request: ChatRequest, rag_service: RagService = Depends(get_rag_service)) -> ChatResponse:
    try:
        rag_answer = rag_service.answer_question(
            chat_request.question, chat_request.folder, chat_request.top_k, entry_point="api"
        )
    except LlmTimeoutError as error:
        logger.warning("LLM timeout: %s", error)
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=LLM_TIMEOUT_DETAIL) from error
    except LlmUnavailableError as error:
        logger.error("LLM unavailable: %s", error)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=LLM_UNAVAILABLE_DETAIL) from error
    return ChatResponse(**asdict(rag_answer))  # refusal_reason is ignored: not a ChatResponse field
