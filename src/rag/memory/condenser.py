import json
import re
from dataclasses import dataclass
from rag.constants import CONDENSED_QUESTION_MAX_LENGTH, KEYWORD_STOP_WORDS, NO_RELEVANT_CONTEXT_ANSWER
from rag.llm.base import LlmClient
from rag.llm.call_context import llm_call_context
from rag.prompt_constants import CONDENSE_REPAIR_TEMPLATE, CONDENSE_SYSTEM_PROMPT, CONDENSE_USER_TEMPLATE

CITATION = re.compile(r"\[\d+\]")
WORD = re.compile(r"[a-z0-9]+")
STEM_LENGTH = 5  # "python" in the history allows "pythonic"; "manage" allows "managed" and "management"


@dataclass
class CheckedQuestion:
    """question is the usable rewrite, or None when a check failed (reason says which)."""

    question: str | None
    reason: str = ""


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def check_standalone_question(reply: str, question: str, history: list[tuple[str, str]]) -> CheckedQuestion:
    """Guards for the rewriting LLM. A rewrite only moves words from the conversation into the question, so a reply that
    answers it, cites sources, refuses, or brings new words is rejected."""
    rewritten = extract_standalone_question(reply)
    if not rewritten:
        return CheckedQuestion(None, 'the reply was not JSON like {"standalone_question": "..."} with a question in it')
    if len(rewritten) > CONDENSED_QUESTION_MAX_LENGTH:
        return CheckedQuestion(None, f"it was longer than {CONDENSED_QUESTION_MAX_LENGTH} characters, "
                                     "so it looks like an answer, not a question")
    if CITATION.search(rewritten) or NO_RELEVANT_CONTEXT_ANSWER.lower() in rewritten.lower():
        return CheckedQuestion(None, "it contained a citation or a refusal; do not answer, only rewrite the question")
    known = set(words(question)) | {word for _, content in history for word in words(content)}
    new_words = [word for word in words(rewritten) if not is_known(word, known)]
    if new_words:
        return CheckedQuestion(None, "it used words that are not in the conversation: " + ", ".join(new_words[:5]))
    return CheckedQuestion(rewritten)


def is_known(word: str, known: set[str]) -> bool:
    if len(word) <= 2 or word in KEYWORD_STOP_WORDS or word in known:
        return True
    return len(word) >= STEM_LENGTH and any(other[:STEM_LENGTH] == word[:STEM_LENGTH] for other in known)


def extract_standalone_question(reply: str) -> str:
    """The question inside the reply's JSON (models often add code fences or a sentence around it), or ''."""
    first_brace, last_brace = reply.find("{"), reply.rfind("}")
    if first_brace == -1 or last_brace <= first_brace:
        return ""
    try:
        value = json.loads(reply[first_brace:last_brace + 1])["standalone_question"]
    except (ValueError, KeyError, TypeError):
        return ""
    return value.strip() if isinstance(value, str) else ""


class QuestionCondenser:
    """Asks the LLM to rewrite a follow-up into a standalone question. Knows nothing about LangGraph or the database."""

    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client

    def rewrite(self, history: list[tuple[str, str]], question: str, rejection_reason: str = "") -> str:
        """The LLM's raw reply. rejection_reason (from a failed check) turns this into a repair attempt."""
        conversation = "\n".join(f"{role}: {content}" for role, content in history)
        user_prompt = CONDENSE_USER_TEMPLATE.format(history=conversation, question=question)
        if rejection_reason:
            user_prompt += CONDENSE_REPAIR_TEMPLATE.format(reason=rejection_reason)
        with llm_call_context("condense"):
            return self.llm_client.generate(CONDENSE_SYSTEM_PROMPT, user_prompt).text
