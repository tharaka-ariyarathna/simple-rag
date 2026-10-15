"""The cross-encoder adapter, with the real sentence-transformers model replaced (it is about 1 GB)."""
import numpy as np
import pytest
from rag.bootstrap import build_reranker
from rag.reranking import cross_encoder_reranker
from rag.reranking.cross_encoder_reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self, model_name):
        self.model_name = model_name
        self.predicted = []

    def predict(self, pairs, **options):
        self.predicted.append((pairs, options))
        return np.array([float(len(passage)) for _, passage in pairs], dtype=np.float32)


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch):
    monkeypatch.setattr(cross_encoder_reranker, "CrossEncoder", FakeCrossEncoder)


def test_scores_each_query_passage_pair_and_returns_plain_floats():
    reranker = CrossEncoderReranker("BAAI/bge-reranker-base", batch_size=8)

    scores = reranker.score("Who knows Python?", ["short", "a longer passage"])

    assert scores == [5.0, 16.0] and all(type(score) is float for score in scores)
    pairs, options = reranker.model.predicted[0]
    assert pairs == [("Who knows Python?", "short"), ("Who knows Python?", "a longer passage")]
    assert options["batch_size"] == 8 and options["show_progress_bar"] is False


def test_no_passages_skips_the_model():
    reranker = CrossEncoderReranker("m")

    assert reranker.score("q", []) == []
    assert reranker.model.predicted == []


def test_bootstrap_builds_the_default_reranker_and_can_switch_it_off():
    assert build_reranker({}).model_name == "BAAI/bge-reranker-base"
    assert build_reranker({"RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2"}).model_name == (
        "cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert build_reranker({"RERANKER_MODEL": "None"}) is None
