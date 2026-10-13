from rag.repository.db import Database


class QuestionRepository:
    """All SQL for synthetic_questions and synthetic_question_chunks (which chunks the generator already did)."""

    def __init__(self, database: Database):
        self.database = database

    def save_questions(self, source: str, chunk_index: int, questions: list[dict], generator_model: str) -> None:
        """Saves a chunk's questions and marks the chunk as processed in one transaction.
        An empty list is valid: the chunk had nothing worth asking and is not retried."""
        with self.database.connection() as connection:
            with connection.cursor() as cursor:
                if questions:
                    cursor.executemany(
                        """INSERT INTO synthetic_questions
                               (source, chunk_index, question, expected_answer, evidence, generator_model)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        [
                            (source, chunk_index, question["question"], question["expected_answer"],
                             question["evidence"], generator_model)
                            for question in questions
                        ],
                    )
                cursor.execute(
                    """INSERT INTO synthetic_question_chunks (source, chunk_index) VALUES (%s, %s)
                       ON CONFLICT DO NOTHING""",
                    (source, chunk_index),
                )

    def processed_chunk_keys(self) -> set[tuple[str, int]]:
        with self.database.connection() as connection:
            rows = connection.execute("SELECT source, chunk_index FROM synthetic_question_chunks").fetchall()
        return {(row["source"], row["chunk_index"]) for row in rows}

    def list_questions(self) -> list[dict]:
        with self.database.connection() as connection:
            return connection.execute(
                """SELECT id, source, chunk_index, question, expected_answer, evidence
                   FROM synthetic_questions ORDER BY id"""
            ).fetchall()
