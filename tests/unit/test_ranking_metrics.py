import math
import pytest
from rag.evaluation.ranking_metrics import (
    dcg, first_relevant_rank, gain, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank,
)

LABELS = {"strong": 2, "partial": 1, "no_match": 0}


def test_gain_rewards_a_strong_match_three_times_a_partial_one():
    assert (gain(0), gain(1), gain(2)) == (0, 1, 3)


def test_dcg_discounts_lower_ranks():
    assert dcg([2, 1]) == pytest.approx(3 / math.log2(2) + 1 / math.log2(3))


def test_ndcg_is_one_for_the_ideal_order_and_lower_otherwise():
    assert ndcg_at_k(["strong", "partial", "no_match"], LABELS, k=3) == pytest.approx(1.0)

    expected = (3 / math.log2(3) + 1 / math.log2(4)) / (3 + 1 / math.log2(3))
    assert ndcg_at_k(["no_match", "strong", "partial"], LABELS, k=3) == pytest.approx(expected)


def test_ndcg_counts_unlabelled_ids_as_no_match_and_is_zero_without_relevant_labels():
    assert ndcg_at_k(["unknown"], LABELS, k=1) == 0.0
    assert ndcg_at_k(["a"], {"a": 0}, k=1) == 0.0


def test_recall_is_the_share_of_relevant_ids_found_in_the_top_k():
    assert recall_at_k(["no_match", "strong", "partial"], LABELS, k=2) == 0.5
    assert recall_at_k(["strong", "partial"], LABELS, k=2) == 1.0
    assert recall_at_k(["a"], {"a": 0}, k=1) == 0.0


def test_precision_divides_by_k_even_when_fewer_ids_were_ranked():
    assert precision_at_k(["no_match", "strong"], LABELS, k=2) == 0.5
    assert precision_at_k(["strong", "partial", "no_match"], LABELS, k=5) == pytest.approx(2 / 5)
    assert precision_at_k([], LABELS, k=5) == 0.0


def test_first_relevant_rank_and_reciprocal_rank():
    assert first_relevant_rank(["no_match", "unknown", "partial"], LABELS) == 3
    assert reciprocal_rank(["no_match", "unknown", "partial"], LABELS) == pytest.approx(1 / 3)
    assert first_relevant_rank(["no_match"], LABELS) is None
    assert reciprocal_rank(["no_match"], LABELS) == 0.0
