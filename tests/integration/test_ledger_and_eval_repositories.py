import psycopg
import pytest
from rag.repository.eval_repository import EvalRepository
from rag.repository.ledger_repository import LedgerRepository
from rag.repository.question_repository import QuestionRepository


def fetch_all(database, query, params=()):
    with database.connection() as connection:
        return connection.execute(query, params).fetchall()


class TestLedgerRepository:
    def test_request_is_inserted_then_finished(self, database):
        ledger = LedgerRepository(database)

        request_id = ledger.insert_rag_request("api", "Who?", "cvs", 5, chunks_found=2, top_similarity=0.8)
        ledger.finish_rag_request(request_id, "Alice [1].", None, total_latency_ms=1234)

        [request] = fetch_all(database, "SELECT * FROM rag_requests")
        assert request["id"] == request_id
        assert (request["entry_point"], request["question"], request["folder"]) == ("api", "Who?", "cvs")
        assert (request["answer"], request["refusal_reason"], request["total_latency_ms"]) == ("Alice [1].", None,
                                                                                               1234)

    def test_llm_calls_point_at_their_request(self, database):
        ledger = LedgerRepository(database)
        request_id = ledger.insert_rag_request("mcp", "q", None, 5, chunks_found=1, top_similarity=0.7)

        ledger.insert_llm_call(request_id, "answer", "gemini", "gemini-x", "primary", 900, None, None, False,
                               "timeout")
        ledger.insert_llm_call(None, "question_generation", "ollama", "qwen", "fallback", 50, 10, 3, True, None)

        calls = fetch_all(database, "SELECT rag_request_id, success, error_type FROM llm_calls ORDER BY id")
        assert calls == [{"rag_request_id": request_id, "success": False, "error_type": "timeout"},
                         {"rag_request_id": None, "success": True, "error_type": None}]

    def test_unknown_refusal_reason_is_rejected_by_the_database(self, database):
        ledger = LedgerRepository(database)
        request_id = ledger.insert_rag_request("api", "q", None, 5, chunks_found=0, top_similarity=None)

        with pytest.raises(psycopg.errors.CheckViolation):
            ledger.finish_rag_request(request_id, "x", "bored", total_latency_ms=1)


class TestEvalRepository:
    def test_latest_finished_run_skips_unfinished_runs_and_other_kinds(self, database):
        repository = EvalRepository(database)
        first = repository.start_run("retrieval", {"top_k": 5}, "abc1234")
        repository.finish_run(first, {"hit_at_k": 0.9}, passed=True)
        repository.start_run("retrieval", {"top_k": 5}, None)  # crashed midway: no metrics
        other_kind = repository.start_run("matching", {}, None)
        repository.finish_run(other_kind, {"ndcg_at_k": 0.5}, passed=True)
        current = repository.start_run("retrieval", {"top_k": 5}, None)

        previous = repository.latest_finished_run("retrieval", before_run_id=current)

        assert previous["id"] == first
        assert previous["metrics"] == {"hit_at_k": 0.9} and previous["settings"] == {"top_k": 5}
        assert previous["passed"] is True

    def test_before_run_id_excludes_the_current_and_later_runs(self, database):
        repository = EvalRepository(database)
        first = repository.start_run("retrieval", {}, None)
        repository.finish_run(first, {"mrr": 0.5}, passed=False)
        second = repository.start_run("retrieval", {}, None)
        repository.finish_run(second, {"mrr": 0.6}, passed=True)

        assert repository.latest_finished_run("retrieval")["id"] == second
        assert repository.latest_finished_run("retrieval", before_run_id=second)["id"] == first
        assert repository.latest_finished_run("retrieval", before_run_id=first) is None

    def test_results_are_saved_with_optional_fields(self, database):
        repository = EvalRepository(database)
        run_id = repository.start_run("unanswerable", {}, None)

        repository.save_results(run_id, [
            {"question": "q1", "outcome": "hit", "rank": 1, "score": 1.0, "similarity": 0.9, "details": {"a": 1}},
            {"question": "q2", "outcome": "refused", "category": "out_of_domain"},
        ])

        rows = fetch_all(database, "SELECT question, category, rank, outcome, details FROM eval_results ORDER BY id")
        assert rows == [{"question": "q1", "category": None, "rank": 1, "outcome": "hit", "details": {"a": 1}},
                        {"question": "q2", "category": "out_of_domain", "rank": None, "outcome": "refused",
                         "details": None}]


class TestQuestionRepository:
    def test_questions_are_saved_and_chunks_marked_processed(self, database):
        repository = QuestionRepository(database)
        questions = [{"question": "Q1?", "expected_answer": "A1", "evidence": "E1"},
                     {"question": "Q2?", "expected_answer": "A2", "evidence": "E2"}]

        repository.save_questions("a.pdf", 0, questions, "gemini-x")
        repository.save_questions("a.pdf", 1, [], "gemini-x")  # nothing worth asking: still processed
        repository.save_questions("a.pdf", 1, [], "gemini-x")  # processing a chunk twice is not an error

        assert repository.processed_chunk_keys() == {("a.pdf", 0), ("a.pdf", 1)}
        saved = repository.list_questions()
        assert [(row["question"], row["chunk_index"], row["evidence"]) for row in saved] == [
            ("Q1?", 0, "E1"), ("Q2?", 0, "E2")]
