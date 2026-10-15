import re
import numpy as np
from psycopg import sql
from rag.repository.db import Database
from rag.constants import HYBRID_VECTOR_CANDIDATES, MIN_SIMILARITY, RETRIEVAL_TOP_K, RRF_K


class ChunkRepository:
    """All SQL for the chunks table. Gets connections from Database; never opens its own.

    The embedding column has no fixed size, so several models can live side by side.
    Every query is scoped to one embedding_model and casts to that model's dimension,
    which is also what lets Postgres use that model's HNSW index.
    """

    def __init__(self, database: Database):
        self.database = database

    def ensure_embedding_index(self, embedding_model: str, dimension: int):
        """One partial HNSW index per model (pgvector can only index vectors of one fixed size)."""
        index_name = "chunks_embedding_" + re.sub(r"\W+", "_", embedding_model).lower()
        with self.database.connection() as connection:
            connection.execute(
                sql.SQL(
                    """CREATE INDEX IF NOT EXISTS {index_name} ON chunks
                       USING hnsw ((embedding::vector({dimension})) vector_cosine_ops)
                       WHERE embedding_model = {embedding_model}"""
                ).format(
                    index_name=sql.Identifier(index_name),
                    dimension=sql.Literal(int(dimension)),
                    embedding_model=sql.Literal(embedding_model),
                )
            )

    def save_chunks(self, chunk_rows: list[dict], embedding_model: str):
        sources = sorted({chunk_row["source"] for chunk_row in chunk_rows})

        with self.database.connection() as connection:
            # Re-indexing a file replaces its old chunks for this model only; other models' rows stay
            connection.execute(
                "DELETE FROM chunks WHERE source = ANY(%s) AND embedding_model = %s",
                (sources, embedding_model),
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    """INSERT INTO chunks (source, folder, chunk_index, content, embedding_model, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    [
                        (chunk_row["source"], chunk_row["folder"], chunk_row["chunk_index"],
                         chunk_row["content"], embedding_model, chunk_row["embedding"])
                        for chunk_row in chunk_rows
                    ],
                )
        print(f"Saved {len(chunk_rows)} chunks from {len(sources)} files for {embedding_model}.")

    def search_similar(
        self,
        query_vector: np.ndarray,
        embedding_model: str,
        dimension: int,
        top_k: int = RETRIEVAL_TOP_K,
        folder: str | None = None,
        min_similarity: float = MIN_SIMILARITY,
        keyword_terms: list[str] | None = None,
    ) -> list[dict]:
        """Hybrid search: chunks ranked by cosine similarity and, when keyword_terms are given, by full-text match,
        merged with reciprocal rank fusion (RRF). similarity = 1 - cosine distance.

        Only chunks at or above min_similarity take part in either ranking, so keywords reorder relevant chunks
        but never bring in one below the threshold, and an empty list still means nothing relevant was found.
        Without keyword_terms the order is plain vector order. Rows also carry vector_rank, keyword_rank
        (None = no keyword match) and fused_score."""
        # The cast must match the index expression exactly, or Postgres falls back to a full scan
        distance = sql.SQL("(embedding::vector({dimension}) <=> %(query_vector)s)").format(
            dimension=sql.Literal(int(dimension))
        )
        query = sql.SQL(
            """WITH keyword_query AS (
                   -- the terms OR-ed together; quote_literal()::tsquery takes each term exactly as given
                   SELECT string_agg(quote_literal(term), ' | ')::tsquery AS query
                   FROM unnest(%(keyword_terms)s::text[]) AS term
               ),
               nearest AS (
                   SELECT id, {distance} AS distance FROM chunks
                   WHERE embedding_model = %(embedding_model)s
                     AND (%(folder)s::text IS NULL OR folder = %(folder)s)
                   ORDER BY {distance}
                   LIMIT %(vector_candidates)s
               ),
               vector_hits AS (
                   SELECT id, row_number() OVER (ORDER BY distance) AS vector_rank
                   FROM nearest WHERE 1 - distance >= %(min_similarity)s
               ),
               keyword_hits AS (
                   -- every keyword match above the threshold, not only the vector candidates
                   SELECT id, row_number() OVER (ORDER BY ts_rank_cd(search_vector, keyword_query.query) DESC, id)
                              AS keyword_rank
                   FROM chunks, keyword_query
                   WHERE keyword_query.query IS NOT NULL
                     AND search_vector @@ keyword_query.query
                     AND embedding_model = %(embedding_model)s
                     AND (%(folder)s::text IS NULL OR folder = %(folder)s)
                     AND 1 - {distance} >= %(min_similarity)s
               )
               SELECT source, folder, chunk_index, content, 1 - {distance} AS similarity,
                      vector_rank, keyword_rank,
                      -- float8, or psycopg returns numeric arithmetic as Decimal
                      (coalesce(1.0 / (%(rrf_k)s + vector_rank), 0)
                       + coalesce(1.0 / (%(rrf_k)s + keyword_rank), 0))::float8 AS fused_score
               FROM chunks
               LEFT JOIN vector_hits USING (id)
               LEFT JOIN keyword_hits USING (id)
               WHERE vector_rank IS NOT NULL OR keyword_rank IS NOT NULL
               ORDER BY fused_score DESC, similarity DESC
               LIMIT %(top_k)s"""
        ).format(distance=distance)
        with self.database.connection() as connection:
            # An HNSW index scan returns at most hnsw.ef_search rows (default 40), fewer than the candidates asked for
            connection.execute("SELECT set_config('hnsw.ef_search', %s, true)",
                               (str(max(HYBRID_VECTOR_CANDIDATES, top_k)),))
            return connection.execute(
                query,
                {"query_vector": query_vector, "embedding_model": embedding_model, "folder": folder,
                 "top_k": top_k, "min_similarity": min_similarity, "keyword_terms": keyword_terms or [],
                 "vector_candidates": max(HYBRID_VECTOR_CANDIDATES, top_k), "rrf_k": RRF_K},
            ).fetchall()

    def question_lexemes(self, question: str) -> list[str]:
        """The question's words exactly as the keyword index stores them (same search_text() normalisation,
        lower-cased, de-duplicated), in the order they first appear."""
        with self.database.connection() as connection:
            rows = connection.execute(
                """SELECT lexeme FROM unnest(to_tsvector('simple', search_text(%s)))
                   ORDER BY positions[1]""",
                (question,),
            ).fetchall()
        return [row["lexeme"] for row in rows]

    def term_chunk_counts(self, terms: list[str], embedding_model: str) -> tuple[dict[str, int], int]:
        """How many chunks of the model contain each term, and how many chunks the model has in total."""
        with self.database.connection() as connection:
            total = connection.execute(
                "SELECT count(*) AS chunk_count FROM chunks WHERE embedding_model = %s", (embedding_model,)
            ).fetchone()["chunk_count"]
            rows = connection.execute(
                """SELECT term, (SELECT count(*) FROM chunks
                                 WHERE embedding_model = %(embedding_model)s
                                   AND search_vector @@ quote_literal(term)::tsquery) AS chunk_count
                   FROM unnest(%(terms)s::text[]) AS term""",
                {"terms": terms, "embedding_model": embedding_model},
            ).fetchall()
        return {row["term"]: row["chunk_count"] for row in rows}, total

    def list_chunks(self, embedding_model: str) -> list[dict]:
        """Every chunk of one model, in document order (used by evaluation, which reads all of them)."""
        with self.database.connection() as connection:
            return connection.execute(
                """SELECT source, folder, chunk_index, content FROM chunks
                   WHERE embedding_model = %s ORDER BY source, chunk_index""",
                (embedding_model,),
            ).fetchall()

    def delete_by_source(self, source: str) -> int:
        with self.database.connection() as connection:
            return connection.execute("DELETE FROM chunks WHERE source = %s", (source,)).rowcount

    def delete_by_embedding_model(self, embedding_model: str) -> int:
        """Clean-up after switching models: removes every row of the old model."""
        with self.database.connection() as connection:
            return connection.execute(
                "DELETE FROM chunks WHERE embedding_model = %s", (embedding_model,)
            ).rowcount

    def count_by_folder(self, embedding_model: str) -> dict[str, int]:
        with self.database.connection() as connection:
            folder_counts = connection.execute(
                """SELECT folder, count(*) AS chunk_count FROM chunks
                   WHERE embedding_model = %s GROUP BY folder ORDER BY folder""",
                (embedding_model,),
            ).fetchall()
        return {folder_count["folder"]: folder_count["chunk_count"] for folder_count in folder_counts}
