"""LangGraph's Postgres checkpointer against the real test database: a conversation survives a new graph object."""
import pytest
from rag.bootstrap import build_checkpointer
from rag.memory.condenser import QuestionCondenser
from rag.memory.conversation_graph import ConversationGraph
from rag.service.prompt_builder import PromptBuilder
from rag.service.rag_service import RagService
from tests.fakes import FakeChunkRepository, FakeEmbedder, FakeLlmClient, chunk_row
from tests.integration.conftest import test_database_url


@pytest.fixture
def checkpointer(database):
    saver, pool = build_checkpointer(test_database_url())
    yield saver
    pool.close()


def make_graph(checkpointer, rewriter):
    service = RagService(FakeChunkRepository([chunk_row("a.pdf", "Nimal Perera, Python")]), FakeEmbedder(),
                         PromptBuilder(), FakeLlmClient(answer="Nimal Perera knows Python [1]."))
    return ConversationGraph(service, QuestionCondenser(rewriter), checkpointer)


def test_history_is_loaded_from_postgres_by_conversation_id(checkpointer):
    first = make_graph(checkpointer, FakeLlmClient()).ask("Who knows Python?", None, 5)
    rewriter = FakeLlmClient(answer='{"standalone_question": "Does Nimal Perera know Kubernetes?"}')

    # a brand new graph object: nothing in memory, the history can only come from the database
    second = make_graph(checkpointer, rewriter).ask("Does he know Kubernetes?", None, 5, first.conversation_id)

    assert second.standalone_question == "Does Nimal Perera know Kubernetes?"
    assert "Assistant: Nimal Perera knows Python [1]." in rewriter.calls[0]["user"]
