from psycopg.types.json import Jsonb
from rag.repository.db import Database


class EvalRepository:
    """All SQL for eval_runs (one row per evaluation run) and eval_results (one row per question in it)."""

    def __init__(self, database: Database):
        self.database = database

    def start_run(self, kind: str, settings: dict, git_commit: str | None) -> int:
        """Created before the questions run, so rag_requests.eval_run_id can point at it. Returns the run id."""
        with self.database.connection() as connection:
            row = connection.execute(
                """INSERT INTO eval_runs (kind, git_commit, settings, metrics, passed)
                   VALUES (%s, %s, %s, '{}'::jsonb, FALSE) RETURNING id""",
                (kind, git_commit, Jsonb(settings)),
            ).fetchone()
        return row["id"]

    def finish_run(self, run_id: int, metrics: dict, passed: bool) -> None:
        with self.database.connection() as connection:
            connection.execute(
                "UPDATE eval_runs SET metrics = %s, passed = %s WHERE id = %s", (Jsonb(metrics), passed, run_id)
            )

    def save_results(self, run_id: int, results: list[dict]) -> None:
        """results: dicts with question, outcome and optionally category, rank, score, similarity, details."""
        with self.database.connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """INSERT INTO eval_results
                           (eval_run_id, question, category, rank, score, similarity, outcome, details)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        (run_id, result["question"], result.get("category"), result.get("rank"),
                         result.get("score"), result.get("similarity"), result["outcome"],
                         Jsonb(result["details"]) if result.get("details") is not None else None)
                        for result in results
                    ],
                )

    def latest_finished_run(self, kind: str, before_run_id: int | None = None) -> dict | None:
        """The most recent run of this kind that has metrics (a run that crashed midway has none)."""
        with self.database.connection() as connection:
            return connection.execute(
                """SELECT id, created_at, settings, metrics, passed FROM eval_runs
                   WHERE kind = %s AND metrics <> '{}'::jsonb AND (%s::bigint IS NULL OR id < %s)
                   ORDER BY id DESC LIMIT 1""",
                (kind, before_run_id, before_run_id),
            ).fetchone()
