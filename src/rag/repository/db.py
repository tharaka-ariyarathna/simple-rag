import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from rag.constants import DB_SCHEMA, DB_POOL_MIN_SIZE, DB_POOL_MAX_SIZE


def configure_connection(connection: psycopg.Connection):
    """Runs once for each new pooled connection."""
    register_vector(connection)  # lets psycopg send/receive numpy arrays as VECTOR
    connection.commit()          # register_vector runs a query; pool requires an idle connection back


class Database:
    """Owns the connection pool. Create one per app and share it between repositories."""

    def __init__(self, database_url: str):
        self.pool = ConnectionPool(
            database_url,
            min_size=DB_POOL_MIN_SIZE,
            max_size=DB_POOL_MAX_SIZE,
            # chunks table and the vector extension both live in DB_SCHEMA, not public
            kwargs={"options": f"-c search_path={DB_SCHEMA}", "row_factory": dict_row},
            configure=configure_connection,
            open=True,
        )

    def connection(self):
        """Borrow a pooled connection: one transaction, commits on success, rolls back on error."""
        return self.pool.connection()

    def close(self):
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
