"""Hybrid search against the real database: the generated search_vector column, the search_text() normalisation
of technology names, and reciprocal rank fusion of the vector and keyword rankings."""
import pytest
from rag.repository.chunk_repository import ChunkRepository
from rag.service.keyword_query import KeywordQueryBuilder
from rag.service.prompt_builder import PromptBuilder
from rag.service.rag_service import RagService
from tests.fakes import FakeEmbedder, FakeLlmClient, FakeReranker, unit_vector

QUERY = unit_vector(1, 0, 0)
MODEL = "model-a"


@pytest.fixture
def repository(database):
    repository = ChunkRepository(database)
    rows = [  # similarity to QUERY in brackets
        ("C:/cvs/ana.pdf", "Built data pipelines with Spark.", unit_vector(1, 0, 0), "cvs"),           # 1.00
        ("C:/cvs/ben.pdf", "Wrote Python services on AWS Lambda.", unit_vector(0.8, 0.6, 0), "cvs"),   # 0.80
        ("C:/cvs/cid.pdf", "Senior C# and .NET developer.", unit_vector(0.6, 0.8, 0), "cvs"),          # 0.60
        ("C:/cvs/dee.pdf", "Python scripting for monthly reports.", unit_vector(0, 0, 1), "cvs"),      # 0.00
        ("C:/jds/eve.pdf", "C++ embedded engineer.", unit_vector(0.7, 0.7, 0), "jds"),                 # 0.71
    ]
    repository.save_chunks([{"source": source, "folder": folder, "chunk_index": 0, "content": content,
                             "embedding": embedding} for source, content, embedding, folder in rows],
                           embedding_model=MODEL)
    return repository


def search(repository, keyword_terms=None, **options):
    return repository.search_similar(QUERY, embedding_model=MODEL, dimension=3, keyword_terms=keyword_terms,
                                     **({"min_similarity": 0.5} | options))


def files(results):
    return [result["source"].rsplit("/", 1)[1] for result in results]


def test_without_keywords_the_order_is_plain_vector_order(repository):
    results = search(repository)

    assert files(results) == ["ana.pdf", "ben.pdf", "eve.pdf", "cid.pdf"]
    assert all(result["keyword_rank"] is None for result in results)


def test_keyword_match_lifts_a_less_similar_chunk_above_a_closer_one(repository):
    results = search(repository, keyword_terms=["python"])

    assert files(results)[:2] == ["ben.pdf", "ana.pdf"]  # ben: vector rank 2 + keyword rank 1 beats ana: vector rank 1
    assert (results[0]["vector_rank"], results[0]["keyword_rank"]) == (2, 1)
    assert results[0]["similarity"] == pytest.approx(0.8, abs=1e-6)  # the cosine similarity is still reported
    assert results[0]["fused_score"] == pytest.approx(1 / 62 + 1 / 61)


def test_keyword_match_never_brings_in_a_chunk_below_the_threshold(repository):
    results = search(repository, keyword_terms=["python"])

    assert "dee.pdf" not in files(results)  # mentions Python, but similarity 0.00
    assert "dee.pdf" in files(search(repository, keyword_terms=["python"], min_similarity=0.0))


def test_csharp_and_cpp_are_told_apart(repository):
    csharp = {result["source"]: result["keyword_rank"] for result in search(repository, keyword_terms=["csharp"])}

    assert csharp["C:/cvs/cid.pdf"] == 1
    assert csharp["C:/jds/eve.pdf"] is None  # "C++" must not match a C# search (plain Postgres makes both "c")


def test_keyword_hits_respect_the_folder_filter(repository):
    assert "eve.pdf" not in files(search(repository, keyword_terms=["cpp"], folder="cvs"))
    assert files(search(repository, keyword_terms=["cpp"], folder="jds")) == ["eve.pdf"]


def test_question_lexemes_are_normalised_like_the_index(repository):
    assert repository.question_lexemes("Who knows C#, C++ and .NET? Who?") == [
        "who", "knows", "csharp", "cpp", "and", "dotnet"]


def test_term_chunk_counts(repository):
    counts, total = repository.term_chunk_counts(["python", "csharp", "dotnet", "cobol"], MODEL)

    assert counts == {"python": 2, "csharp": 1, "dotnet": 1, "cobol": 0}
    assert total == 5
    assert repository.term_chunk_counts(["python"], "other-model") == ({"python": 0}, 0)


def test_search_vector_is_generated_on_insert_and_indexed(repository, database):
    with database.connection() as connection:
        vector = connection.execute(
            "SELECT search_vector::text AS v FROM chunks WHERE source = 'C:/cvs/cid.pdf'").fetchone()["v"]
        index = connection.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_search_vector_idx'").fetchone()
    assert "'csharp'" in vector and "'dotnet'" in vector
    assert "gin" in index["indexdef"]


def test_rag_service_finds_the_keyword_match_first(repository):
    embedder = FakeEmbedder({"Who knows Python?": QUERY}, model_name=MODEL, dimension=3)
    service = RagService(repository, embedder, PromptBuilder(), FakeLlmClient(),
                         keyword_query_builder=KeywordQueryBuilder(repository))

    results = service.search_chunks("Who knows Python?", None, top_k=3)

    assert files(results) == ["ben.pdf", "ana.pdf", "eve.pdf"]
    assert files(service.search_chunks("Who knows Python?", None, top_k=3, use_keywords=False)) == [
        "ana.pdf", "ben.pdf", "eve.pdf"]


def test_reranker_reorders_hybrid_results_but_never_adds_chunks_below_the_threshold(repository):
    embedder = FakeEmbedder({"Who writes reports?": QUERY}, model_name=MODEL, dimension=3)
    reranker = FakeReranker({"Senior C# and .NET developer.": 5.0,
                             "Python scripting for monthly reports.": 99.0})  # below the threshold: never a candidate
    service = RagService(repository, embedder, PromptBuilder(), FakeLlmClient(), reranker=reranker)

    results = service.search_chunks("Who writes reports?", None, top_k=2)

    assert files(results) == ["cid.pdf", "ana.pdf"]  # cid moves from 4th to 1st, ana (score 0) keeps its place
    assert "Python scripting for monthly reports." not in reranker.calls[0][1]
