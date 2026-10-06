import re
import numpy as np
from psycopg import sql
from rag.repository.db import Database
from rag.constants import RETRIEVAL_TOP_K, MIN_SIMILARITY


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
    ) -> list[dict]:
        """Nearest chunks by cosine distance (<=>). similarity = 1 - distance.
        Chunks below min_similarity are dropped, so an empty list means nothing relevant was found."""
        # The cast must match the index expression exactly, or Postgres falls back to a full scan
        distance = sql.SQL("(embedding::vector({dimension}) <=> %(query_vector)s)").format(
            dimension=sql.Literal(int(dimension))
        )
        query = sql.SQL(
            """SELECT source, folder, chunk_index, content, 1 - {distance} AS similarity
               FROM chunks
               WHERE embedding_model = %(embedding_model)s
                 AND (%(folder)s::text IS NULL OR folder = %(folder)s)
                 AND 1 - {distance} >= %(min_similarity)s
               ORDER BY {distance}
               LIMIT %(top_k)s"""
        ).format(distance=distance)
        with self.database.connection() as connection:
            return connection.execute(
                query,
                {"query_vector": query_vector, "embedding_model": embedding_model, "folder": folder,
                 "top_k": top_k, "min_similarity": min_similarity},
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
