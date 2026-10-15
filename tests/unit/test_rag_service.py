import pytest
from rag.constants import NO_RELEVANT_CONTEXT_ANSWER
from rag.llm.base import LlmUnavailableError
from rag.service.prompt_builder import PromptBuilder
from rag.service.rag_service import REFUSAL_REASON_LLM, REFUSAL_REASON_RETRIEVAL, RagService, SourceReference
from rag.service.keyword_query import KeywordQueryBuilder
from tests.fakes import (
    FakeChunkRepository, FakeEmbedder, FakeKeywordRepository, FakeLedgerRepository, FakeLlmClient, chunk_row,
)


def make_service(rows=None, llm=None, ledger=None, embedder=None) -> RagService:
    return RagService(
        repository=FakeChunkRepository(rows),
        embedder=embedder or FakeEmbedder(),
        prompt_builder=PromptBuilder(),
        llm_client=llm or FakeLlmClient(),
        ledger_repository=ledger,
    )


def test_no_chunk_above_threshold_refuses_without_calling_the_llm():
    llm = FakeLlmClient()
    service = make_service(rows=[], llm=llm)

    rag_answer = service.answer_question("Unrelated?", None, top_k=5)

    assert rag_answer.answer == NO_RELEVANT_CONTEXT_ANSWER
    assert rag_answer.refusal_reason == REFUSAL_REASON_RETRIEVAL
    assert rag_answer.sources == []
    assert llm.calls == []


def test_answer_comes_from_the_llm_with_the_retrieved_chunks_in_the_prompt():
    llm = FakeLlmClient(answer="Alice knows Python [1].")
    service = make_service(rows=[chunk_row("C:/cvs/alice.pdf", "Alice knows Python.")], llm=llm)

    rag_answer = service.answer_question("Who knows Python?", None, top_k=5)

    assert rag_answer.answer == "Alice knows Python [1]."
    assert rag_answer.refusal_reason is None
    assert "Alice knows Python." in llm.calls[0]["user"]
    assert llm.calls[0]["user"].endswith("Question: Who knows Python?")


def test_sources_are_one_per_file_keeping_the_best_chunk():
    rows = [chunk_row("C:/cvs/alice.pdf", similarity=0.9, chunk_index=3),
            chunk_row("C:/cvs/bob.pdf", similarity=0.8),
            chunk_row("C:/cvs/alice.pdf", similarity=0.7, chunk_index=1)]

    rag_answer = make_service(rows=rows).answer_question("q", None, top_k=5)

    assert rag_answer.sources == [SourceReference("alice.pdf", "docs", 0.9), SourceReference("bob.pdf", "docs", 0.8)]


def test_llm_saying_the_refusal_text_is_marked_as_an_llm_refusal():
    llm = FakeLlmClient(answer=f"  {NO_RELEVANT_CONTEXT_ANSWER}\n")

    rag_answer = make_service(rows=[chunk_row("a.pdf")], llm=llm).answer_question("q", None, top_k=5)

    assert rag_answer.refusal_reason == REFUSAL_REASON_LLM


def test_search_uses_the_active_model_folder_and_top_k():
    repository = FakeChunkRepository()
    embedder = FakeEmbedder(model_name="model-x", dimension=3)
    service = RagService(repository, embedder, PromptBuilder(), FakeLlmClient())

    service.search_chunks("q", "cvs", top_k=7)

    call = repository.search_calls[0]
    assert (call["embedding_model"], call["dimension"], call["folder"], call["top_k"]) == ("model-x", 3, "cvs", 7)
    assert "min_similarity" not in call  # the repository's default threshold applies
    assert embedder.queries == ["q"]


def test_search_passes_an_explicit_threshold_through():
    repository = FakeChunkRepository()
    service = RagService(repository, FakeEmbedder(), PromptBuilder(), FakeLlmClient())

    service.search_chunks("q", None, top_k=5, min_similarity=0.0)

    assert repository.search_calls[0]["min_similarity"] == 0.0


def test_list_folders_counts_chunks_of_the_active_model():
    service = RagService(FakeChunkRepository(folder_counts={"cvs": 12}), FakeEmbedder(), PromptBuilder(),
                         FakeLlmClient())

    assert service.list_folders() == {"cvs": 12}



class TestHybridSearch:
    def service_with_keywords(self, repository: FakeChunkRepository) -> RagService:
        builder = KeywordQueryBuilder(FakeKeywordRepository({"python": 24}, total_chunks=100))
        return RagService(repository, FakeEmbedder(), PromptBuilder(), FakeLlmClient(),
                          keyword_query_builder=builder)

    def test_keyword_terms_of_the_question_are_passed_to_the_search(self):
        repository = FakeChunkRepository()

        self.service_with_keywords(repository).search_chunks("Who knows python?", None, top_k=5)

        assert repository.search_calls[0]["keyword_terms"] == ["python"]

    def test_keywords_can_be_switched_off_for_a_search(self):
        repository = FakeChunkRepository()

        self.service_with_keywords(repository).search_chunks("Who knows python?", None, top_k=5, use_keywords=False)

        assert repository.search_calls[0]["keyword_terms"] == []

    def test_without_a_keyword_builder_the_search_is_vector_only(self):
        repository = FakeChunkRepository()

        RagService(repository, FakeEmbedder(), PromptBuilder(), FakeLlmClient()).search_chunks("python", None, 5)

        assert repository.search_calls[0]["keyword_terms"] == []

    def test_ledger_records_the_highest_similarity_even_when_hybrid_order_puts_it_second(self):
        ledger = FakeLedgerRepository()
        rows = [chunk_row("keyword-match.pdf", similarity=0.61), chunk_row("closest.pdf", similarity=0.72)]

        make_service(rows=rows, ledger=ledger).answer_question("q", None, top_k=5)

        assert ledger.requests[1]["top_similarity"] == pytest.approx(0.72)

class TestLedger:
    def test_answered_request_is_recorded_from_start_to_finish(self):
        ledger = FakeLedgerRepository()
        rows = [chunk_row("a.pdf", similarity=0.83), chunk_row("b.pdf", similarity=0.6)]

        make_service(rows=rows, ledger=ledger).answer_question("q", "docs", top_k=4, entry_point="mcp")

        request = ledger.requests[1]
        assert request["entry_point"] == "mcp"
        assert (request["question"], request["folder"], request["top_k"]) == ("q", "docs", 4)
        assert request["chunks_found"] == 2
        assert request["top_similarity"] == pytest.approx(0.83)
        assert request["finished"] and request["answer"] == "An answer [1]." and request["refusal_reason"] is None

    def test_retrieval_refusal_is_recorded(self):
        ledger = FakeLedgerRepository()

        make_service(rows=[], ledger=ledger).answer_question("q", None, top_k=5)

        request = ledger.requests[1]
        assert request["chunks_found"] == 0 and request["top_similarity"] is None
        assert request["refusal_reason"] == REFUSAL_REASON_RETRIEVAL

    def test_llm_call_runs_inside_the_request_context(self):
        llm = FakeLlmClient()

        make_service(rows=[chunk_row("a.pdf")], llm=llm, ledger=FakeLedgerRepository()).answer_question(
            "q", None, top_k=5)

        assert llm.calls[0]["purpose"] == "answer"
        assert llm.calls[0]["rag_request_id"] == 1

    def test_failed_llm_call_still_finishes_the_request_with_no_answer(self):
        ledger = FakeLedgerRepository()
        llm = FakeLlmClient(error=LlmUnavailableError("down"))

        with pytest.raises(LlmUnavailableError):
            make_service(rows=[chunk_row("a.pdf")], llm=llm, ledger=ledger).answer_question("q", None, top_k=5)

        assert ledger.requests[1]["finished"] and ledger.requests[1]["answer"] is None

    def test_broken_ledger_never_breaks_an_answer(self):
        rag_answer = make_service(rows=[chunk_row("a.pdf")], ledger=FakeLedgerRepository(fail=True)).answer_question(
            "q", None, top_k=5)

        assert rag_answer.answer == "An answer [1]."

    def test_eval_run_id_is_recorded(self):
        ledger = FakeLedgerRepository()

        make_service(rows=[], ledger=ledger).answer_question("q", None, top_k=5, entry_point="eval", eval_run_id=9)

        assert ledger.requests[1]["eval_run_id"] == 9
