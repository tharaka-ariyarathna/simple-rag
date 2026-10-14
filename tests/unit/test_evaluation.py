"""Scoring and pass/fail logic of the three evaluations, with the RagService replaced by stubs."""
import pytest
from rag.constants import EVAL_MAX_REGRESSION, EVAL_MIN_HIT_RATE
from rag.evaluation import matching_eval, retrieval_eval, unanswerable_eval
from rag.evaluation.retrieval_eval import RetrievalCase
from rag.evaluation.text_matching import contains_evidence, normalize_text
from rag.llm.base import LlmUnavailableError
from rag.service.rag_service import RagAnswer
from tests.fakes import chunk_row


class SearchStub:
    """search_chunks returns `filtered` normally and `unfiltered` when called with min_similarity=0.0."""

    def __init__(self, filtered=None, unfiltered=None):
        self.filtered = filtered or []
        self.unfiltered = unfiltered if unfiltered is not None else self.filtered
        self.calls = []

    def search_chunks(self, question, folder, top_k, min_similarity=None):
        self.calls.append({"question": question, "folder": folder, "top_k": top_k, "min_similarity": min_similarity})
        return self.unfiltered if min_similarity == 0.0 else self.filtered


def test_evidence_matches_across_line_breaks_and_case():
    assert normalize_text("  Line one\n  line   TWO ") == "line one line two"
    assert contains_evidence("He used AWS\nLambda and SQS daily.", "aws lambda and sqs")
    assert not contains_evidence("He used AWS Lambda.", "Azure Functions")


class TestRetrievalEval:
    CASE = RetrievalCase(question="Which queue?", source="cv.pdf", evidence="used SQS")
    ALL_CHUNKS = [chunk_row("cv.pdf", "He used SQS for jobs.")]

    def test_evidence_in_no_chunk_is_unmatchable_and_skips_retrieval(self):
        stub = SearchStub()
        case = RetrievalCase("q", "cv.pdf", "text that is gone")

        result = retrieval_eval.evaluate_case(stub, case, self.ALL_CHUNKS, top_k=5)

        assert result["outcome"] == retrieval_eval.OUTCOME_UNMATCHABLE
        assert stub.calls == []

    def test_hit_records_rank_reciprocal_rank_and_similarity(self):
        stub = SearchStub(filtered=[chunk_row("other.pdf", "nothing", similarity=0.8),
                                    chunk_row("cv.pdf", "He used SQS for jobs.", similarity=0.7)])

        result = retrieval_eval.evaluate_case(stub, self.CASE, self.ALL_CHUNKS, top_k=5)

        assert (result["outcome"], result["rank"], result["score"]) == ("hit", 2, 0.5)
        assert result["similarity"] == pytest.approx(0.7)
        assert result["details"]["file_hit"] is True

    def test_miss_hidden_by_the_threshold_is_reported_as_below_threshold(self):
        stub = SearchStub(filtered=[], unfiltered=[chunk_row("cv.pdf", "He used SQS for jobs.", similarity=0.41)])

        result = retrieval_eval.evaluate_case(stub, self.CASE, self.ALL_CHUNKS, top_k=5)

        assert result["outcome"] == retrieval_eval.OUTCOME_BELOW_THRESHOLD
        assert result["details"]["rank_without_threshold"] == 1
        assert result["similarity"] == pytest.approx(0.41)
        assert [call["min_similarity"] for call in stub.calls] == [None, 0.0]

    def test_real_miss(self):
        stub = SearchStub(filtered=[chunk_row("other.pdf", "nothing")])

        result = retrieval_eval.evaluate_case(stub, self.CASE, self.ALL_CHUNKS, top_k=5)

        assert (result["outcome"], result["score"]) == ("miss", 0.0)
        assert result["details"]["file_hit"] is False

    def test_metrics_ignore_unmatchable_results(self):
        results = [
            {"outcome": "hit", "rank": 1, "score": 1.0, "details": {"file_hit": True}},
            {"outcome": "hit", "rank": 2, "score": 0.5, "details": {"file_hit": True}},
            {"outcome": "below_threshold", "score": 0.0, "details": {"file_hit": True}},
            {"outcome": "miss", "score": 0.0, "details": {"file_hit": False}},
            {"outcome": "unmatchable", "details": {}},
        ]

        metrics = retrieval_eval.compute_metrics(results)

        assert metrics == {"evaluated": 4, "unmatchable": 1, "hit_at_1": 0.25, "hit_at_k": 0.5,
                           "file_hit_at_k": 0.75, "mrr": 0.375, "below_threshold_misses": 1}

    def test_judge_checks_targets_and_regressions(self):
        good = {"evaluated": 10, "hit_at_k": 0.9, "mrr": 0.8}
        assert retrieval_eval.judge(good, None) == (True, [])

        passed, problems = retrieval_eval.judge({"evaluated": 10, "hit_at_k": EVAL_MIN_HIT_RATE - 0.01, "mrr": 0.8},
                                                None)
        assert not passed and "hit@k" in problems[0]

        dropped = {"evaluated": 10, "hit_at_k": 0.9, "mrr": 0.8 - EVAL_MAX_REGRESSION - 0.01}
        passed, problems = retrieval_eval.judge(dropped, previous_metrics=good)
        assert not passed and "mrr dropped" in problems[0]

        assert retrieval_eval.judge({"evaluated": 0}, None)[0] is False


class TestUnanswerableEval:
    def test_ask_classifies_refused_answered_and_error(self):
        class Service:
            def __init__(self, rag_answer=None, error=None):
                self.rag_answer, self.error, self.calls = rag_answer, error, []

            def answer_question(self, question, folder, top_k, entry_point, eval_run_id):
                self.calls.append((question, folder, top_k, entry_point, eval_run_id))
                if self.error:
                    raise self.error
                return self.rag_answer

        case = {"question": "Capital of Peru?", "category": "out_of_domain"}

        refusing = Service(RagAnswer(answer="I couldn't find that.", refusal_reason="retrieval"))
        assert unanswerable_eval.ask(refusing, case, 5, eval_run_id=3)["outcome"] == "refused"
        assert refusing.calls == [("Capital of Peru?", None, 5, "eval", 3)]

        answering = Service(RagAnswer(answer="Lima."))
        assert unanswerable_eval.ask(answering, case, 5, eval_run_id=3)["outcome"] == "answered"

        failing = Service(error=LlmUnavailableError("down"))
        assert unanswerable_eval.ask(failing, case, 5, eval_run_id=3)["outcome"] == "error"

    def test_metrics_split_hallucinations_from_false_refusals_and_skip_errors(self):
        results = [
            {"category": "out_of_domain", "outcome": "refused", "details": {"refusal_reason": "retrieval"}},
            {"category": "near_miss", "outcome": "answered", "details": {"refusal_reason": None}},
            {"category": "answerable", "outcome": "refused", "details": {"refusal_reason": "llm"}},
            {"category": "answerable", "outcome": "answered", "details": {"refusal_reason": None}},
            {"category": "answerable", "outcome": "error", "details": {"error": "down"}},
        ]

        metrics = unanswerable_eval.compute_metrics(results)

        assert metrics["hallucination_rate"] == 0.5
        assert metrics["false_refusal_rate"] == 0.5
        assert metrics["errors"] == 1
        assert metrics["categories"]["answerable"] == {"total": 2, "refused": 1, "refused_by_retrieval": 0,
                                                       "refused_by_llm": 1, "refusal_rate": 0.5}

    def test_judge(self):
        assert unanswerable_eval.judge({"hallucination_rate": 0.0, "false_refusal_rate": 0.1}) == (True, [])
        assert unanswerable_eval.judge({"hallucination_rate": 0.5})[0] is False
        assert unanswerable_eval.judge({"hallucination_rate": 0.0, "false_refusal_rate": 0.9})[0] is False
        assert unanswerable_eval.judge({}) == (False, ["no unanswerable question could be evaluated"])

    def test_answerable_sample_is_the_same_every_run(self):
        questions = [{"question": f"q{number}"} for number in range(50)]

        first = unanswerable_eval.sample_answerable_questions(questions, 5)

        assert first == unanswerable_eval.sample_answerable_questions(questions, 5)
        assert len(first) == 5 and all(case["category"] == "answerable" for case in first)

    def test_bundled_unanswerable_questions_load(self):
        cases = unanswerable_eval.load_unanswerable_questions()

        assert cases and all(case["question"] and case["category"] for case in cases)


class TestMatchingEval:
    def test_candidates_are_ranked_by_their_best_chunk(self):
        stub = SearchStub(filtered=[chunk_row("C:/cvs/cv_001.pdf", similarity=0.7),
                                    chunk_row("C:/cvs/cv_002.pdf", similarity=0.9),
                                    chunk_row("C:/cvs/cv_001.pdf", similarity=0.8)])

        ranking = matching_eval.rank_candidates(stub, "JD text", chunks_retrieved=50)

        assert ranking == [("cv_002", 0.9), ("cv_001", 0.8)]
        assert stub.calls == [{"question": "JD text", "folder": None, "top_k": 50, "min_similarity": 0.0}]

    def test_evaluate_jd(self):
        stub = SearchStub(filtered=[chunk_row("cv_002.pdf", similarity=0.9), chunk_row("cv_001.pdf", similarity=0.8)])

        result = matching_eval.evaluate_jd(stub, "jd_001", "JD text", {"cv_001": 2, "cv_003": 1},
                                           chunks_retrieved=50, cutoff=10)

        assert (result["question"], result["rank"], result["similarity"]) == ("jd_001", 2, 0.9)
        assert result["details"]["relevant_count"] == 2
        assert result["details"]["candidates_ranked"] == 2
        assert result["details"]["recall_at_k"] == 0.5
        assert [entry["grade"] for entry in result["details"]["top"]] == [0, 2]

    def test_compute_metrics_averages_over_jds(self):
        results = [{"details": {"ndcg_at_k": 1.0, "recall_at_k": 1.0, "precision_at_k": 0.2, "mrr": 1.0,
                                "candidates_ranked": 10}},
                   {"details": {"ndcg_at_k": 0.0, "recall_at_k": 0.5, "precision_at_k": 0.1, "mrr": 0.5,
                                "candidates_ranked": 6}}]

        metrics = matching_eval.compute_metrics(results, cutoff=10)

        assert metrics == pytest.approx({"ndcg_at_k": 0.5, "recall_at_k": 0.75, "precision_at_k": 0.15, "mrr": 0.75,
                                         "jd_count": 2, "cutoff": 10, "avg_candidates_ranked": 8.0})
        assert matching_eval.compute_metrics([], cutoff=10) == {"jd_count": 0, "cutoff": 10}

    def test_runs_are_comparable_only_with_the_same_protocol(self):
        base = {"cutoff": 10, "chunks_retrieved": 50, "jd_count": 20, "embedding_model": "a"}

        assert matching_eval.is_comparable(base, base | {"embedding_model": "b"})
        assert not matching_eval.is_comparable(base, base | {"cutoff": 5})
        assert matching_eval.changed_settings(base, base | {"embedding_model": "b", "new": 1}) == {
            "embedding_model": ("a", "b"), "new": (None, 1)}

    def test_judge_first_run_is_a_baseline_and_a_drop_fails(self):
        metrics = {"jd_count": 5, "ndcg_at_k": 0.6, "recall_at_k": 0.7}

        assert matching_eval.judge(metrics, None) == (True, [])
        passed, problems = matching_eval.judge(metrics, {"ndcg_at_k": 0.7, "recall_at_k": 0.7})
        assert not passed and "ndcg_at_k dropped" in problems[0]
        assert matching_eval.judge({"jd_count": 0}, None)[0] is False
