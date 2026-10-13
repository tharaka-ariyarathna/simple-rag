from dataclasses import dataclass
from rag.constants import EVAL_MAX_REGRESSION, EVAL_MIN_HIT_RATE, EVAL_MIN_MRR
from rag.evaluation.text_matching import contains_evidence
from rag.service.rag_service import RagService

OUTCOME_HIT = "hit"
OUTCOME_MISS = "miss"
OUTCOME_BELOW_THRESHOLD = "below_threshold"  # the evidence chunk is retrievable, but MIN_SIMILARITY hid it
OUTCOME_UNMATCHABLE = "unmatchable"  # no chunk contains the evidence any more (e.g. chunk size changed)

REGRESSION_METRICS = ("hit_at_k", "mrr")


@dataclass
class RetrievalCase:
    question: str
    source: str
    evidence: str


def first_rank_containing(chunks: list[dict], evidence: str) -> int | None:
    """1-based position of the first retrieved chunk that contains the evidence."""
    for rank, chunk in enumerate(chunks, start=1):
        if contains_evidence(chunk["content"], evidence):
            return rank
    return None


def evaluate_case(rag_service: RagService, case: RetrievalCase, all_chunks: list[dict], top_k: int) -> dict:
    """Runs retrieval only (no LLM) and checks where the evidence chunk lands."""
    if not any(contains_evidence(chunk["content"], case.evidence) for chunk in all_chunks):
        return {"question": case.question, "outcome": OUTCOME_UNMATCHABLE, "details": {"source": case.source}}

    retrieved = rag_service.search_chunks(case.question, None, top_k)
    rank = first_rank_containing(retrieved, case.evidence)
    file_hit = any(chunk["source"] == case.source for chunk in retrieved)
    details = {"source": case.source, "file_hit": file_hit, "retrieved_count": len(retrieved)}
    if rank is not None:
        similarity = float(retrieved[rank - 1]["similarity"])
        return {"question": case.question, "outcome": OUTCOME_HIT, "rank": rank, "score": 1 / rank,
                "similarity": similarity, "details": details}

    # Missed: would it have been found without the similarity threshold?
    unfiltered = rag_service.search_chunks(case.question, None, top_k, min_similarity=0.0)
    unfiltered_rank = first_rank_containing(unfiltered, case.evidence)
    if unfiltered_rank is not None:
        details["rank_without_threshold"] = unfiltered_rank
        details["file_hit"] = file_hit or any(chunk["source"] == case.source for chunk in unfiltered)
        return {"question": case.question, "outcome": OUTCOME_BELOW_THRESHOLD, "score": 0.0,
                "similarity": float(unfiltered[unfiltered_rank - 1]["similarity"]), "details": details}
    return {"question": case.question, "outcome": OUTCOME_MISS, "score": 0.0, "details": details}


def compute_metrics(results: list[dict]) -> dict:
    evaluated = [result for result in results if result["outcome"] != OUTCOME_UNMATCHABLE]
    count = len(evaluated)
    if count == 0:
        return {"evaluated": 0, "unmatchable": len(results)}
    hits = [result for result in evaluated if result["outcome"] == OUTCOME_HIT]
    return {
        "evaluated": count,
        "unmatchable": len(results) - count,
        "hit_at_1": sum(1 for result in hits if result["rank"] == 1) / count,
        "hit_at_k": len(hits) / count,
        "file_hit_at_k": sum(1 for result in evaluated if result["details"].get("file_hit")) / count,
        "mrr": sum(result["score"] for result in evaluated) / count,
        "below_threshold_misses": sum(1 for result in evaluated if result["outcome"] == OUTCOME_BELOW_THRESHOLD),
    }


def judge(metrics: dict, previous_metrics: dict | None) -> tuple[bool, list[str]]:
    """Passed = above the fixed targets and no real drop against the previous run. Returns the reasons it failed."""
    if metrics.get("evaluated", 0) == 0:
        return False, ["no questions could be evaluated"]
    problems = []
    if metrics["hit_at_k"] < EVAL_MIN_HIT_RATE:
        problems.append(f"hit@k {metrics['hit_at_k']:.2f} is below the target {EVAL_MIN_HIT_RATE}")
    if metrics["mrr"] < EVAL_MIN_MRR:
        problems.append(f"MRR {metrics['mrr']:.2f} is below the target {EVAL_MIN_MRR}")
    if previous_metrics:
        for name in REGRESSION_METRICS:
            if name in previous_metrics and previous_metrics[name] - metrics[name] > EVAL_MAX_REGRESSION:
                problems.append(f"{name} dropped from {previous_metrics[name]:.2f} to {metrics[name]:.2f}")
    return not problems, problems
