import logging
from dataclasses import asdict
from fastapi import APIRouter, Depends, HTTPException, Request, status
from rag.api.schemas import ChatRequest, ChatResponse
from rag.constants import INTERNAL_ERROR_DETAIL, LLM_TIMEOUT_DETAIL, LLM_UNAVAILABLE_DETAIL
from rag.memory.conversation_graph import ConversationGraph
from rag.llm.base import LlmConfigurationError, LlmTimeoutError, LlmUnavailableError

logger = logging.getLogger(__name__)
router = APIRouter()


def get_conversation_graph(request: Request) -> ConversationGraph:
    return request.app.state.conversation_graph  # created once at startup (see main.py lifespan)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Plain def (not async): the RAG flow is blocking (embedding, psycopg, LLM call),
# so FastAPI runs it in a worker thread instead of freezing the event loop
@router.post("/chat", response_model=ChatResponse)
def chat(chat_request: ChatRequest,
         conversation_graph: ConversationGraph = Depends(get_conversation_graph)) -> ChatResponse:
    try:
        result = conversation_graph.ask(
            chat_request.question, chat_request.folder, chat_request.top_k, chat_request.conversation_id
        )
    except LlmTimeoutError as error:
        logger.warning("LLM timeout: %s", error)
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=LLM_TIMEOUT_DETAIL) from error
    except LlmUnavailableError as error:
        logger.error("LLM unavailable: %s", error)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=LLM_UNAVAILABLE_DETAIL) from error
    except LlmConfigurationError as error:  # bad key, bad request, unknown model: our fault, not the caller's
        logger.error("LLM misconfigured: %s", error)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=INTERNAL_ERROR_DETAIL) from error
    rag_answer = result.rag_answer  # refusal_reason is left out: not a ChatResponse field
    return ChatResponse(answer=rag_answer.answer, sources=[asdict(source) for source in rag_answer.sources], conversation_id=result.conversation_id,
                        standalone_question=result.standalone_question)
