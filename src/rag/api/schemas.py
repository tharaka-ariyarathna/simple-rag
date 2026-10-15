from typing import Annotated
from pydantic import BaseModel, Field, StringConstraints
from rag.constants import CONVERSATION_ID_MAX_LENGTH, QUESTION_MAX_LENGTH, FOLDER_MAX_LENGTH, TOP_K_MAX, RETRIEVAL_TOP_K

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


class SourceResponse(BaseModel):
    file_name: str
    folder: str
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    conversation_id: str  # send it back with the next question to continue the conversation
    standalone_question: str  # what was searched: the question rewritten to make sense without the conversation
