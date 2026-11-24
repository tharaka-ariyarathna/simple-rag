from psycopg.types.json import Jsonb
from rag.repository.chunk_repository import insert_chunks
from rag.repository.db import Database


class CandidateRepository:
    """All SQL for the candidates table: one row per CV, the id that chunks, answers and matches refer to."""

    def __init__(self, database: Database):
        self.database = database

    def get(self, candidate_id: int) -> dict | None:
        with self.database.connection() as connection:
            return connection.execute("SELECT * FROM candidates WHERE id = %s", (candidate_id,)).fetchone()

    def get_by_source(self, source: str) -> dict | None:
        with self.database.connection() as connection:
            return connection.execute("SELECT * FROM candidates WHERE source = %s", (source,)).fetchone()

    def has_chunks(self, candidate_id: int, embedding_model: str) -> bool:
        with self.database.connection() as connection:
            return connection.execute(
                "SELECT EXISTS (SELECT 1 FROM chunks WHERE candidate_id = %s AND embedding_model = %s) AS found",
                (candidate_id, embedding_model),
            ).fetchone()["found"]

    def save(self, candidate: dict, chunk_rows: list[dict], embedding_model: str) -> int:
        """Upserts the candidate and replaces its chunks for this model in ONE transaction, so a crash never leaves
        a new content_hash next to old chunks (the next run would then wrongly skip the CV as unchanged).

        candidate: source, content_hash, name, total_years, skills, jobs, profile_model."""
        with self.database.connection() as connection:
            candidate_id = connection.execute(
                """INSERT INTO candidates (source, content_hash, name, total_years, skills, jobs, profile_model)
                   VALUES (%(source)s, %(content_hash)s, %(name)s, %(total_years)s, %(skills)s, %(jobs)s,
                           %(profile_model)s)
                   ON CONFLICT (source) DO UPDATE SET
                       content_hash = EXCLUDED.content_hash, name = EXCLUDED.name,
                       total_years = EXCLUDED.total_years, skills = EXCLUDED.skills, jobs = EXCLUDED.jobs,
                       profile_model = EXCLUDED.profile_model, extracted_at = now()
                   RETURNING id""",
                {**candidate, "skills": Jsonb(candidate["skills"]), "jobs": Jsonb(candidate["jobs"])},
            ).fetchone()["id"]
            # rows from before candidates existed have no candidate_id, so old chunks are matched by source as well
            connection.execute(
                "DELETE FROM chunks WHERE (candidate_id = %s OR source = %s) AND embedding_model = %s",
                (candidate_id, candidate["source"], embedding_model),
            )
            insert_chunks(connection, [{**row, "candidate_id": candidate_id} for row in chunk_rows], embedding_model)
        return candidate_id

    def find_by_name(self, name: str) -> list[dict]:
        """Candidates whose full name, first name or last name equals this text, ignoring case. Names are not
        unique, so this can return several rows: the caller decides (or asks the user) which one is meant."""
        with self.database.connection() as connection:
            return connection.execute(
                """SELECT id, name, total_years, skills FROM candidates
                   WHERE lower(name) = lower(%(name)s) OR lower(%(name)s) = ANY(string_to_array(lower(name), ' '))
                   ORDER BY id""",
                {"name": name.strip()},
            ).fetchall()

    def ids_with_min_years(self, min_years: float) -> list[int]:
        """The /match pre-filter: candidates with at least this many years (total years is computed in code from
        validated job dates, so it is reliable enough to filter on, unlike LLM-extracted skills)."""
        with self.database.connection() as connection:
            rows = connection.execute("SELECT id FROM candidates WHERE total_years >= %s ORDER BY id",
                                      (min_years,)).fetchall()
        return [row["id"] for row in rows]

    def cv_text(self, candidate_id: int, embedding_model: str) -> str:
        """The candidate's redacted CV as indexed: its chunks in order. This is all a cloud judge LLM ever reads."""
        with self.database.connection() as connection:
            rows = connection.execute(
                """SELECT content FROM chunks WHERE candidate_id = %s AND embedding_model = %s
                   ORDER BY chunk_index""",
                (candidate_id, embedding_model),
            ).fetchall()
        return "\n\n".join(row["content"] for row in rows)

    def sources_by_id(self, candidate_ids: list[int]) -> dict[int, str]:
        with self.database.connection() as connection:
            rows = connection.execute("SELECT id, source FROM candidates WHERE id = ANY(%s)",
                                      (list(candidate_ids),)).fetchall()
        return {row["id"]: row["source"] for row in rows}

    def all_names(self) -> list[dict]:
        """Every candidate's id, name and a little profile, for finding names in questions (500 rows is small)."""
        with self.database.connection() as connection:
            return connection.execute("SELECT id, name, total_years, skills FROM candidates ORDER BY id").fetchall()

    def names_by_id(self, candidate_ids: list[int]) -> dict[int, str | None]:
        with self.database.connection() as connection:
            rows = connection.execute("SELECT id, name FROM candidates WHERE id = ANY(%s)",
                                      (list(candidate_ids),)).fetchall()
        return {row["id"]: row["name"] for row in rows}

    def delete(self, candidate_id: int) -> bool:
        """Removes the candidate; its chunks and embeddings go with it (ON DELETE CASCADE). False = no such id."""
        with self.database.connection() as connection:
            return connection.execute("DELETE FROM candidates WHERE id = %s", (candidate_id,)).rowcount > 0
