from pathlib import Path
from rag.constants import EVAL_MAX_REGRESSION, MATCHING_MIN_SIMILARITY
from rag.evaluation.ranking_metrics import (
    first_relevant_rank, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank,
)
from rag.service.rag_service import RagService

OUTCOME_EVALUATED = "evaluated"
METRIC_NAMES = ("ndcg_at_k", "recall_at_k", "precision_at_k", "mrr")
REGRESSION_METRICS = ("ndcg_at_k", "recall_at_k")
PROTOCOL_KEYS = ("cutoff", "chunks_retrieved", "jd_count")  # runs are only comparable when these match


def candidate_id_of(source: str) -> str:
    """The CV file name without .pdf, e.g. cv_001: the same id the labels use."""
    return Path(source).stem


def rank_candidates(rag_service: RagService, query_text: str, chunks_retrieved: int) -> list[tuple[str, float]]:
    """Retrieves chunks for the query and groups them by CV. A CV's score is its best chunk's similarity.
    Returns (cv id, score), best first. CVs with no chunk among the retrieved ones are not ranked at all."""
    # Vector only: a whole JD is not a keyword query (hybrid comes with the JD-condensing step)
    chunks = rag_service.search_chunks(query_text, None, chunks_retrieved, min_similarity=MATCHING_MIN_SIMILARITY,
                                       use_keywords=False, use_reranker=False)
    best_score_by_candidate: dict[str, float] = {}
    for chunk in chunks:
        candidate_id = candidate_id_of(chunk["source"])
        best_score_by_candidate[candidate_id] = max(
            best_score_by_candidate.get(candidate_id, 0.0), float(chunk["similarity"])
        )
    return sorted(best_score_by_candidate.items(), key=lambda item: item[1], reverse=True)


def evaluate_jd(
    rag_service: RagService, jd_id: str, jd_text: str, labels: dict[str, int], chunks_retrieved: int, cutoff: int
) -> dict:
    ranking = rank_candidates(rag_service, jd_text, chunks_retrieved)
    ranked_ids = [candidate_id for candidate_id, _ in ranking]
    relevant_count = sum(1 for grade in labels.values() if grade >= 1)
    metrics = {
        "ndcg_at_k": ndcg_at_k(ranked_ids, labels, cutoff),
        "recall_at_k": recall_at_k(ranked_ids, labels, cutoff),
        "precision_at_k": precision_at_k(ranked_ids, labels, cutoff),
        "mrr": reciprocal_rank(ranked_ids, labels),
    }
    return {
        "question": jd_id,
        "outcome": OUTCOME_EVALUATED,
        "rank": first_relevant_rank(ranked_ids, labels),  # where the first relevant CV landed, None = not found
        "score": metrics["ndcg_at_k"],
        "similarity": ranking[0][1] if ranking else None,
        "details": {
            **metrics,
            "relevant_count": relevant_count,
            "candidates_ranked": len(ranked_ids),  # below the cutoff = the chunks came from too few CVs
            "top": [{"cv": candidate_id, "score": round(score, 3), "grade": labels.get(candidate_id, 0)}
                    for candidate_id, score in ranking[:cutoff]],
        },
    }


def compute_metrics(results: list[dict], cutoff: int) -> dict:
    count = len(results)
    if count == 0:
        return {"jd_count": 0, "cutoff": cutoff}
    metrics = {name: sum(result["details"][name] for result in results) / count for name in METRIC_NAMES}
    metrics["jd_count"] = count
    metrics["cutoff"] = cutoff
    metrics["avg_candidates_ranked"] = sum(result["details"]["candidates_ranked"] for result in results) / count
    return metrics


def is_comparable(previous_settings: dict, settings: dict) -> bool:
    return all(previous_settings.get(key) == settings.get(key) for key in PROTOCOL_KEYS)


def changed_settings(previous_settings: dict, settings: dict) -> dict[str, tuple]:
    """What differs from the previous run: the thing you changed, which explains a change in the scores."""
    return {
        key: (previous_settings.get(key), settings.get(key))
        for key in sorted(set(previous_settings) | set(settings))
        if previous_settings.get(key) != settings.get(key)
    }


def judge(metrics: dict, previous_metrics: dict | None) -> tuple[bool, list[str]]:
    """No fixed targets yet: the first run is the baseline. A later comparable run fails if it got worse."""
    if metrics["jd_count"] == 0:
        return False, ["no JD could be evaluated"]
    problems = []
    if previous_metrics:
        for name in REGRESSION_METRICS:
            if previous_metrics[name] - metrics[name] > EVAL_MAX_REGRESSION:
                problems.append(f"{name} dropped from {previous_metrics[name]:.3f} to {metrics[name]:.3f}")
    return not problems, problems
