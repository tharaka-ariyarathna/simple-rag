import logging
import uuid
from dataclasses import asdict
from fastapi import APIRouter, Depends, HTTPException, Request, status
from rag.api.schemas import ChatRequest, ChatResponse, MatchRequest, MatchResponse
from rag.constants import (
    INTERNAL_ERROR_DETAIL, LLM_TIMEOUT_DETAIL, LLM_UNAVAILABLE_DETAIL, MATCH_REQUEST_PREVIEW_CHARS,
    REQUIREMENT_EXTRACTION_FAILED_DETAIL,
)
from rag.matching.match_graph import MatchGraph
from rag.matching.requirements import RequirementExtractionError
from rag.memory.conversation_graph import ConversationGraph
from rag.repository.candidate_repository import CandidateRepository
from rag.service.candidate_names import candidate_ids_in, with_names
from rag.llm.base import LlmConfigurationError, LlmTimeoutError, LlmUnavailableError

logger = logging.getLogger(__name__)
router = APIRouter()


def get_conversation_graph(request: Request) -> ConversationGraph:
    return request.app.state.conversation_graph  # created once at startup (see main.py lifespan)


def get_candidate_repository(request: Request) -> CandidateRepository:
    return request.app.state.candidate_repository


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Plain def (not async): the RAG flow is blocking (embedding, psycopg, LLM call),
# so FastAPI runs it in a worker thread instead of freezing the event loop
@router.post("/chat", response_model=ChatResponse)
def chat(chat_request: ChatRequest,
         conversation_graph: ConversationGraph = Depends(get_conversation_graph),
         candidate_repository: CandidateRepository = Depends(get_candidate_repository)) -> ChatResponse:
    try:
        result = conversation_graph.ask(
            chat_request.question, chat_request.folder, chat_request.top_k, chat_request.conversation_id,
            chat_request.candidate_ids,
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
    # The LLM only ever saw "candidate N": names are put back here, in the response, and nowhere else
    mentioned = candidate_ids_in(rag_answer.answer) + candidate_ids_in(result.standalone_question) + [
        source.candidate_id for source in rag_answer.sources if source.candidate_id is not None]
    names = candidate_repository.names_by_id(list(dict.fromkeys(mentioned))) if mentioned else {}
    sources = [asdict(source) | {"candidate_name": names.get(source.candidate_id)} for source in rag_answer.sources]
    return ChatResponse(answer=with_names(rag_answer.answer, names), sources=sources,
                        conversation_id=result.conversation_id,
                        standalone_question=with_names(result.standalone_question, names))


@router.delete("/candidates/{candidate_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_candidate(candidate_id: int,
                     candidate_repository: CandidateRepository = Depends(get_candidate_repository)) -> None:
    """Removes a candidate with all its chunks and embeddings (ON DELETE CASCADE). Conversation checkpoints that
    mention the candidate keep only its id, never its CV text."""
    if not candidate_repository.delete(candidate_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such candidate.")


def get_match_graph(request: Request) -> MatchGraph:
    return request.app.state.match_graph


@router.post("/match", response_model=MatchResponse)
def match(match_request: MatchRequest, match_graph: MatchGraph = Depends(get_match_graph),
          conversation_graph: ConversationGraph = Depends(get_conversation_graph),
          candidate_repository: CandidateRepository = Depends(get_candidate_repository)) -> MatchResponse:
    """Ranks candidates for a job description, with per-requirement evidence. The result is stored in a conversation,
    so /chat with the returned conversation_id can answer follow-ups ("why is candidate 17 first?")."""
    try:
        result = match_graph.match(match_request.jd_text, match_request.top_n)
    except RequirementExtractionError as error:
        logger.warning("Requirement extraction failed: %s", error)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=REQUIREMENT_EXTRACTION_FAILED_DETAIL) from error
    except LlmTimeoutError as error:
        logger.warning("LLM timeout: %s", error)
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=LLM_TIMEOUT_DETAIL) from error
    except LlmUnavailableError as error:
        logger.error("LLM unavailable: %s", error)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=LLM_UNAVAILABLE_DETAIL) from error
    except LlmConfigurationError as error:
        logger.error("LLM misconfigured: %s", error)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=INTERNAL_ERROR_DETAIL) from error
    conversation_id = match_request.conversation_id or str(uuid.uuid4())
    ranked = ", ".join(f"candidate {c['candidate_id']} ({c['score']:.2f})" for c in result["candidates"])
    conversation_graph.remember_match(
        conversation_id, "Match candidates to this job description: "
        + match_request.jd_text[:MATCH_REQUEST_PREVIEW_CHARS], result,
        f"Ranked {len(result['candidates'])} candidates: {ranked or 'none found'}.")
    names = candidate_repository.names_by_id([c["candidate_id"] for c in result["candidates"]])
    return MatchResponse(conversation_id=conversation_id, requirements=result["requirements"],
                         min_total_years=result["min_total_years"],
                         candidates=[c | {"candidate_name": names.get(c["candidate_id"])} for c in result["candidates"]])
