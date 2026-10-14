"""The whole question flow against the real database: real ChunkRepository and LedgerRepository, real
FallbackLlmClient and LoggingLlmClient. Only the embedding model and the LLM providers are fakes."""
import pytest
from rag.constants import NO_RELEVANT_CONTEXT_ANSWER
from rag.bootstrap import fallback_role
from rag.llm.base import LlmConfigurationError, LlmTimeoutError, LlmUnavailableError
from rag.llm.fallback_client import FallbackLlmClient
from rag.llm.logging_client import LoggingLlmClient
from rag.repository.chunk_repository import ChunkRepository
from rag.repository.eval_repository import EvalRepository
from rag.repository.ledger_repository import LedgerRepository
from rag.service.prompt_builder import PromptBuilder
from rag.service.rag_service import REFUSAL_REASON_RETRIEVAL, RagService
from tests.fakes import FakeEmbedder, FakeLlmClient, unit_vector

EMBEDDER = FakeEmbedder({
    "Who built the billing service?": unit_vector(1, 0.1, 0),
    "What is the capital of Peru?": unit_vector(0, 0, 1),
}, model_name="fake-embedder", dimension=3)


@pytest.fixture
def chunks(database):
    repository = ChunkRepository(database)
    repository.save_chunks([
        {"source": "C:/cvs/alice.pdf", "folder": "cvs", "chunk_index": 0,
         "content": "Alice built the billing service.", "embedding": unit_vector(1, 0, 0)},
        {"source": "C:/cvs/bob.pdf", "folder": "cvs", "chunk_index": 0,
         "content": "Bob runs the data platform.", "embedding": unit_vector(0.7, 0.7, 0)},
    ], embedding_model=EMBEDDER.model_name)
    return repository


def build_service(database, providers: list[tuple[str, FakeLlmClient]]) -> RagService:
    """providers: (provider name, fake client) in fallback order, wired the way bootstrap wires the real ones."""
    ledger = LedgerRepository(database)
    llm_client = FallbackLlmClient([LoggingLlmClient(client, provider, fallback_role(position), ledger)
                                    for position, (provider, client) in enumerate(providers)])
    return RagService(ChunkRepository(database), EMBEDDER, PromptBuilder(), llm_client, ledger)


def fetch_all(database, query):
    with database.connection() as connection:
        return connection.execute(query).fetchall()


def test_third_tier_answer_is_recorded_as_one_request_and_three_llm_calls(database, chunks):
    service = build_service(database, [
        ("groq", FakeLlmClient(error=LlmUnavailableError("429 rate limited"), model_name="gpt-oss")),
        ("openrouter", FakeLlmClient(error=LlmTimeoutError("slow"), model_name="nemotron")),
        ("ollama", FakeLlmClient(answer="Alice [1].", model_name="llama")),
    ])

    rag_answer = service.answer_question("Who built the billing service?", "cvs", top_k=5, entry_point="api")

    assert rag_answer.answer == "Alice [1]."
    assert [source.file_name for source in rag_answer.sources] == ["alice.pdf", "bob.pdf"]

    [request] = fetch_all(database, "SELECT * FROM rag_requests")
    assert (request["answer"], request["refusal_reason"], request["chunks_found"]) == ("Alice [1].", None, 2)
    assert request["top_similarity"] == pytest.approx(rag_answer.sources[0].similarity)

    calls = fetch_all(database, "SELECT rag_request_id, purpose, provider, model, role, success, error_type "
                                "FROM llm_calls ORDER BY id")
    assert calls == [
        {"rag_request_id": request["id"], "purpose": "answer", "provider": "groq", "model": "gpt-oss",
         "role": "primary", "success": False, "error_type": "unavailable"},
        {"rag_request_id": request["id"], "purpose": "answer", "provider": "openrouter", "model": "nemotron",
         "role": "fallback_1", "success": False, "error_type": "timeout"},
        {"rag_request_id": request["id"], "purpose": "answer", "provider": "ollama", "model": "llama",
         "role": "fallback_2", "success": True, "error_type": None},
    ]


def test_configuration_error_is_recorded_and_stops_the_chain(database, chunks):
    local = FakeLlmClient()
    service = build_service(database, [
        ("groq", FakeLlmClient(error=LlmConfigurationError("401 bad key"), model_name="gpt-oss")),
        ("ollama", local),
    ])

    with pytest.raises(LlmConfigurationError):
        service.answer_question("Who built the billing service?", None, top_k=5)

    assert local.calls == []
    [request] = fetch_all(database, "SELECT answer FROM rag_requests")
    assert request["answer"] is None  # the request is still finished, without an answer
    assert fetch_all(database, "SELECT role, error_type FROM llm_calls") == [
        {"role": "primary", "error_type": "configuration"}]


def test_question_about_nothing_in_the_documents_is_refused_before_any_llm_call(database, chunks):
    primary = FakeLlmClient()
    service = build_service(database, [("groq", primary), ("ollama", FakeLlmClient())])

    rag_answer = service.answer_question("What is the capital of Peru?", None, top_k=5)

    assert rag_answer.answer == NO_RELEVANT_CONTEXT_ANSWER
    assert primary.calls == []
    [request] = fetch_all(database, "SELECT refusal_reason, chunks_found, top_similarity FROM rag_requests")
    assert request == {"refusal_reason": REFUSAL_REASON_RETRIEVAL, "chunks_found": 0, "top_similarity": None}
    assert fetch_all(database, "SELECT * FROM llm_calls") == []


def test_eval_questions_are_linked_to_their_eval_run(database, chunks):
    run_id = EvalRepository(database).start_run("unanswerable", {}, None)
    service = build_service(database, [("groq", FakeLlmClient())])

    service.answer_question("Who built the billing service?", None, top_k=5, entry_point="eval", eval_run_id=run_id)

    assert fetch_all(database, "SELECT entry_point, eval_run_id FROM rag_requests") == [
        {"entry_point": "eval", "eval_run_id": run_id}]
