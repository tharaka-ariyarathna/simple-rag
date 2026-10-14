import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from rag.constants import (
    EVAL_MAX_FALSE_REFUSAL_RATE, EVAL_MAX_HALLUCINATION_RATE, EVAL_RANDOM_SEED, UNANSWERABLE_QUESTIONS_PATH,
)
from rag.llm.base import RECOVERABLE_LLM_ERRORS
from rag.service.rag_service import REFUSAL_REASON_LLM, REFUSAL_REASON_RETRIEVAL, RagService

logger = logging.getLogger(__name__)

CATEGORY_ANSWERABLE = "answerable"  # synthetic questions: the system should NOT refuse these
OUTCOME_REFUSED = "refused"
OUTCOME_ANSWERED = "answered"
OUTCOME_ERROR = "error"  # the LLM failed, so we learned nothing about refusing


def load_unanswerable_questions() -> list[dict]:
    path = Path(__file__).resolve().parent.parent / UNANSWERABLE_QUESTIONS_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def sample_answerable_questions(synthetic_questions: list[dict], sample_size: int) -> list[dict]:
    """A fixed random sample (same seed every run) of questions the documents DO answer."""
    shuffled = list(synthetic_questions)
    random.Random(EVAL_RANDOM_SEED).shuffle(shuffled)
    return [
        {"question": row["question"], "category": CATEGORY_ANSWERABLE} for row in shuffled[:sample_size]
    ]


def ask(rag_service: RagService, case: dict, top_k: int, eval_run_id: int) -> dict:
    """Runs one question through the full pipeline (retrieval + LLM) and classifies the outcome.
    An LlmConfigurationError is not caught: every other question would fail the same way, so the run stops."""
    try:
        rag_answer = rag_service.answer_question(
            case["question"], None, top_k, entry_point="eval", eval_run_id=eval_run_id
        )
    except RECOVERABLE_LLM_ERRORS as error:
        logger.warning("LLM failed for %r: %s", case["question"], error)
        return {"question": case["question"], "category": case["category"], "outcome": OUTCOME_ERROR,
                "details": {"error": str(error)[:300]}}
    outcome = OUTCOME_REFUSED if rag_answer.refusal_reason else OUTCOME_ANSWERED
    return {"question": case["question"], "category": case["category"], "outcome": outcome,
            "details": {"refusal_reason": rag_answer.refusal_reason, "answer": rag_answer.answer}}


def compute_metrics(results: list[dict]) -> dict:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        if result["outcome"] != OUTCOME_ERROR:
            by_category[result["category"]].append(result)

    categories = {}
    for category, category_results in by_category.items():
        refused = [result for result in category_results if result["outcome"] == OUTCOME_REFUSED]
        categories[category] = {
            "total": len(category_results),
            "refused": len(refused),
            "refused_by_retrieval": sum(
                1 for result in refused if result["details"]["refusal_reason"] == REFUSAL_REASON_RETRIEVAL),
            "refused_by_llm": sum(
                1 for result in refused if result["details"]["refusal_reason"] == REFUSAL_REASON_LLM),
            "refusal_rate": len(refused) / len(category_results),
        }

    unanswerable = [result for category, rows in by_category.items() if category != CATEGORY_ANSWERABLE
                    for result in rows]
    answerable = by_category.get(CATEGORY_ANSWERABLE, [])
    metrics = {"categories": categories, "errors": sum(1 for r in results if r["outcome"] == OUTCOME_ERROR)}
    if unanswerable:  # answered although the documents do not contain the answer
        metrics["hallucination_rate"] = sum(1 for r in unanswerable if r["outcome"] == OUTCOME_ANSWERED) / len(
            unanswerable)
    if answerable:  # refused although the documents contain the answer
        metrics["false_refusal_rate"] = sum(1 for r in answerable if r["outcome"] == OUTCOME_REFUSED) / len(
            answerable)
    return metrics


def judge(metrics: dict) -> tuple[bool, list[str]]:
    problems = []
    if "hallucination_rate" not in metrics:
        problems.append("no unanswerable question could be evaluated")
    elif metrics["hallucination_rate"] > EVAL_MAX_HALLUCINATION_RATE:
        problems.append(
            f"hallucination rate {metrics['hallucination_rate']:.2f} is above {EVAL_MAX_HALLUCINATION_RATE}")
    if metrics.get("false_refusal_rate", 0.0) > EVAL_MAX_FALSE_REFUSAL_RATE:
        problems.append(
            f"false refusal rate {metrics['false_refusal_rate']:.2f} is above {EVAL_MAX_FALSE_REFUSAL_RATE}")
    return not problems, problems
