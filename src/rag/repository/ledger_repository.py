from rag.repository.db import Database


class LedgerRepository:
    """All SQL for rag_requests (one row per question asked) and llm_calls (one row per LLM attempt)."""

    def __init__(self, database: Database):
        self.database = database

    def insert_rag_request(
        self,
        entry_point: str,
        question: str,
        folder: str | None,
        top_k: int,
        chunks_found: int,
        top_similarity: float | None,
        eval_run_id: int | None = None,
    ) -> int:
        """Written after retrieval and before the LLM call, so llm_calls can point at it. Returns the new id."""
        with self.database.connection() as connection:
            row = connection.execute(
                """INSERT INTO rag_requests
                       (entry_point, eval_run_id, question, folder, top_k, chunks_found, top_similarity, total_latency_ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, 0)
                   RETURNING id""",
                (entry_point, eval_run_id, question, folder, top_k, chunks_found, top_similarity),
            ).fetchone()
        return row["id"]

    def finish_rag_request(
        self, rag_request_id: int, answer: str | None, refusal_reason: str | None, total_latency_ms: int
    ) -> None:
        with self.database.connection() as connection:
            connection.execute(
                "UPDATE rag_requests SET answer = %s, refusal_reason = %s, total_latency_ms = %s WHERE id = %s",
                (answer, refusal_reason, total_latency_ms, rag_request_id),
            )

    def insert_llm_call(
        self,
        rag_request_id: int | None,
        purpose: str,
        provider: str,
        model: str,
        role: str,
        latency_ms: int,
        input_tokens: int | None,
        output_tokens: int | None,
        success: bool,
        error_type: str | None,
    ) -> None:
        with self.database.connection() as connection:
            connection.execute(
                """INSERT INTO llm_calls
                       (rag_request_id, purpose, provider, model, role, latency_ms,
                        input_tokens, output_tokens, success, error_type)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (rag_request_id, purpose, provider, model, role, latency_ms,
                 input_tokens, output_tokens, success, error_type),
            )
