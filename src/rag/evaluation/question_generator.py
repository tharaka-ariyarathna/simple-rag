import json
import logging
import re
from dataclasses import dataclass
from rag.evaluation.text_matching import contains_evidence
from rag.llm.base import LlmClient
from rag.llm.call_context import llm_call_context
from rag.prompt_constants import QUESTION_GENERATION_SYSTEM_PROMPT, QUESTION_GENERATION_USER_TEMPLATE

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("question", "expected_answer", "evidence")


class QuestionGenerationError(Exception):
    """The LLM answered, but not with usable JSON. The chunk stays unprocessed and is retried next run."""


@dataclass
class GeneratedQuestions:
    questions: list[dict]  # [{question, expected_answer, evidence}]; empty = nothing worth asking
    model: str  # the model that actually wrote them (LlmResult.model), not the client's configured name


class QuestionGenerator:
    """Asks the LLM for test questions about one chunk. Knows nothing about the database."""

    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client

    def generate(self, chunk_content: str) -> GeneratedQuestions:
        """Raises LlmError when the LLM fails and QuestionGenerationError when its reply is not usable."""
        user_prompt = QUESTION_GENERATION_USER_TEMPLATE.format(passage=chunk_content.strip())
        with llm_call_context("question_generation"):
            result = self.llm_client.generate(QUESTION_GENERATION_SYSTEM_PROMPT, user_prompt)
        return GeneratedQuestions(questions=self.parse_questions(result.text, chunk_content), model=result.model)

    def parse_questions(self, reply_text: str, chunk_content: str) -> list[dict]:
        reply_json = self.extract_json(reply_text)
        try:
            candidates = json.loads(reply_json)["questions"]
        except (ValueError, KeyError, TypeError) as error:
            raise QuestionGenerationError(f"Reply is not the expected JSON: {reply_text[:200]!r}") from error
        if not isinstance(candidates, list):
            raise QuestionGenerationError(f"'questions' is not a list: {reply_text[:200]!r}")
        questions = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or not all(
                isinstance(candidate.get(field), str) and candidate[field].strip() for field in REQUIRED_FIELDS
            ):
                logger.info("Dropped a question with missing fields: %s", candidate)
                continue
            if not contains_evidence(chunk_content, candidate["evidence"]):
                logger.info("Dropped a question whose evidence is not in the chunk: %s", candidate["question"])
                continue
            questions.append({field: candidate[field].strip() for field in REQUIRED_FIELDS})
        return questions

    @staticmethod
    def extract_json(reply_text: str) -> str:
        """Models often wrap JSON in code fences or add a sentence around it: keep the outermost {...}."""
        first_brace, last_brace = reply_text.find("{"), reply_text.rfind("}")
        if first_brace == -1 or last_brace <= first_brace:
            return reply_text
        return reply_text[first_brace:last_brace + 1]
