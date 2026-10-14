"""Metrics for a ranked list of candidates against graded labels (0 = no match, 1 = partial, 2 = strong).
Pure functions: no database, no LLM."""
import math


def gain(grade: int) -> float:
    return 2 ** grade - 1  # a strong match (2) is worth 3x a partial one (1), not 2x


def dcg(grades_in_rank_order: list[int]) -> float:
    return sum(gain(grade) / math.log2(rank + 1) for rank, grade in enumerate(grades_in_rank_order, start=1))


def ndcg_at_k(ranked_ids: list[str], labels: dict[str, int], k: int) -> float:
    """1.0 = the best possible ordering of the top k; 0.0 = no relevant CV in the top k.
    Unlabelled ids count as 0."""
    achieved = dcg([labels.get(candidate_id, 0) for candidate_id in ranked_ids[:k]])
    ideal = dcg(sorted(labels.values(), reverse=True)[:k])
    return achieved / ideal if ideal > 0 else 0.0


def recall_at_k(ranked_ids: list[str], labels: dict[str, int], k: int) -> float:
    """Share of all relevant CVs (grade >= 1) that appear in the top k."""
    relevant_ids = {candidate_id for candidate_id, grade in labels.items() if grade >= 1}
    if not relevant_ids:
        return 0.0
    return len(relevant_ids & set(ranked_ids[:k])) / len(relevant_ids)


def precision_at_k(ranked_ids: list[str], labels: dict[str, int], k: int) -> float:
    """Share of the top k that is relevant. Capped by how many relevant CVs exist when that is below k."""
    top_ids = ranked_ids[:k]
    if not top_ids:
        return 0.0
    return sum(1 for candidate_id in top_ids if labels.get(candidate_id, 0) >= 1) / k


def first_relevant_rank(ranked_ids: list[str], labels: dict[str, int]) -> int | None:
    for rank, candidate_id in enumerate(ranked_ids, start=1):
        if labels.get(candidate_id, 0) >= 1:
            return rank
    return None


def reciprocal_rank(ranked_ids: list[str], labels: dict[str, int]) -> float:
    rank = first_relevant_rank(ranked_ids, labels)
    return 1 / rank if rank else 0.0
