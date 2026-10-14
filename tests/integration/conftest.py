"""Integration tests run the real repositories against a real Postgres with pgvector.

They need TEST_DATABASE_URL (in the environment or .env), pointing at a database whose name ends in _test,
e.g. postgresql://postgres:postgres@localhost:5432/rag_test. The database is created if missing, and its
`main` schema is dropped and rebuilt from db/*.sql at the start of every session, so never point it at real data.
Without TEST_DATABASE_URL every integration test is skipped.
"""
import os
from pathlib import Path
import psycopg
import pytest
from dotenv import load_dotenv
from psycopg import conninfo, sql
from rag.constants import DB_SCHEMA
from rag.repository.db import Database

DB_DIR = Path(__file__).resolve().parents[2] / "db"
# The schema a fresh database gets. 002 is left out: it migrates a database made by the *old* init.sql,
# and init.sql already creates chunks in its final shape (running 002 on it would fail on the constraint).
SCHEMA_FILES = ["init.sql", "003_llm_call_ledger.sql", "004_synthetic_questions.sql", "005_eval_runs.sql"]
APP_TABLES = ["chunks", "llm_calls", "rag_requests", "eval_results", "eval_runs", "synthetic_questions",
              "synthetic_question_chunks"]


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.is_relative_to(Path(__file__).parent):
            item.add_marker(pytest.mark.integration)


def test_database_url() -> str:
    load_dotenv()
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    database_name = conninfo.conninfo_to_dict(url).get("dbname", "")
    if not database_name.endswith("_test"):
        pytest.fail(f"TEST_DATABASE_URL must name a database ending in _test (got {database_name!r}): "
                    "the tests drop and recreate its schema")
    return url


test_database_url.__test__ = False  # a helper, not a test, despite the name


def create_database_if_missing(url: str) -> None:
    database_name = conninfo.conninfo_to_dict(url)["dbname"]
    with psycopg.connect(conninfo.make_conninfo(url, dbname="postgres"), autocommit=True) as connection:
        exists = connection.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,)).fetchone()
        if not exists:
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))


def rebuild_schema(url: str) -> None:
    with psycopg.connect(url, autocommit=True) as connection:
        connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(DB_SCHEMA)))
        connection.execute("DROP EXTENSION IF EXISTS vector CASCADE")  # recreated inside DB_SCHEMA by init.sql
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(DB_SCHEMA)))
        for file_name in SCHEMA_FILES:
            connection.execute((DB_DIR / file_name).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def database():
    """One real Database (connection pool) for the whole session, on a freshly built schema."""
    url = test_database_url()
    try:
        create_database_if_missing(url)
    except psycopg.OperationalError as error:
        pytest.skip(f"Postgres at TEST_DATABASE_URL is not reachable: {error}")
    rebuild_schema(url)
    with Database(url) as database:
        yield database


@pytest.fixture(autouse=True)
def empty_tables(database):
    """Every test starts from empty tables, so tests never depend on each other's rows."""
    with database.connection() as connection:
        connection.execute(sql.SQL("TRUNCATE {} RESTART IDENTITY CASCADE").format(
            sql.SQL(", ").join(sql.Identifier(table) for table in APP_TABLES)))
