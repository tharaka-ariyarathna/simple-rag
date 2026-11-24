from typing import Annotated
from pydantic import BaseModel, Field, StringConstraints
from rag.constants import CONVERSATION_ID_MAX_LENGTH, JD_MAX_LENGTH, MATCH_TOP_N_DEFAULT, MATCH_TOP_N_MAX, QUESTION_MAX_LENGTH, FOLDER_MAX_LENGTH, TOP_K_MAX, RETRIEVAL_TOP_K

# Whitespace is stripped before the length check, so "   " is rejected as empty
QuestionText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=QUESTION_MAX_LENGTH)]
FolderName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=FOLDER_MAX_LENGTH)]
ConversationId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1,
                                                  max_length=CONVERSATION_ID_MAX_LENGTH)]


class ChatRequest(BaseModel):
    model_config = {"extra": "forbid"}  # unknown fields are a 422, not silently ignored

    question: QuestionText
    folder: FolderName | None = Field(default=None, description="Only search documents in this folder")
    top_k: int = Field(default=RETRIEVAL_TOP_K, ge=1, le=TOP_K_MAX, description="How many chunks to retrieve")
    conversation_id: ConversationId | None = Field(
        default=None, description="Continue this conversation; leave out to start a new one")
    candidate_ids: list[Annotated[int, Field(ge=1)]] | None = Field(
        default=None, max_length=TOP_K_MAX, description="Only search these candidates (ids from earlier responses)")


class SourceResponse(BaseModel):
    file_name: str
    folder: str
    similarity: float
    candidate_id: int | None = None
    candidate_name: str | None = None  # added by the API: names never pass through the LLM


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    conversation_id: str  # send it back with the next question to continue the conversation
    standalone_question: str  # what was searched: the question rewritten to make sense without the conversation


class MatchRequest(BaseModel):
    model_config = {"extra": "forbid"}

    jd_text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=20, max_length=JD_MAX_LENGTH)]
    top_n: int = Field(default=MATCH_TOP_N_DEFAULT, ge=1, le=MATCH_TOP_N_MAX, description="Candidates to return")
    conversation_id: ConversationId | None = Field(
        default=None, description="Put the result in this conversation; leave out to start a new one")


class RequirementResponse(BaseModel):
    skill: str
    must_have: bool
    min_years: float | None
    synonyms: list[str]


class VerdictResponse(BaseModel):
    requirement: str
    must_have: bool
    verdict: str  # met / partial / not_found
    evidence: str  # a quote from the candidate's (redacted) CV
    note: str | None = None


class MatchCandidateResponse(BaseModel):
    candidate_id: int
    candidate_name: str | None  # added by the API: the LLMs only saw the id
    score: float  # 0-1: weighted share of requirements met; capped when a must-have is missing
    verdicts: list[VerdictResponse]
    gaps: list[str]
    missing_must_haves: list[str]


class MatchResponse(BaseModel):
    conversation_id: str  # send it to /chat to ask follow-up questions about this ranking
    requirements: list[RequirementResponse]
    min_total_years: float | None
    candidates: list[MatchCandidateResponse]
