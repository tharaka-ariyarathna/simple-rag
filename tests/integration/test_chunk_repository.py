import pytest
from rag.repository.chunk_repository import ChunkRepository
from tests.fakes import unit_vector


def row(source, chunk_index, content, embedding, folder="cvs"):
    return {"source": source, "folder": folder, "chunk_index": chunk_index, "content": content,
            "embedding": embedding}


@pytest.fixture
def repository(database):
    repository = ChunkRepository(database)
    repository.save_chunks([
        row("C:/cvs/alice.pdf", 0, "exact match", unit_vector(1, 0, 0)),
        row("C:/cvs/alice.pdf", 1, "related", unit_vector(0.6, 0.8, 0)),
        row("C:/jds/jd.pdf", 0, "unrelated", unit_vector(0, 1, 0), folder="jds"),
    ], embedding_model="model-a")
    return repository


def search(repository, **options):
    return repository.search_similar(unit_vector(1, 0, 0), embedding_model="model-a", dimension=3, **options)


def test_search_returns_chunks_best_first_with_cosine_similarity(repository):
    results = search(repository, min_similarity=0.0)

    assert [result["content"] for result in results] == ["exact match", "related", "unrelated"]
    assert [result["similarity"] for result in results] == pytest.approx([1.0, 0.6, 0.0], abs=1e-6)
    assert results[0]["source"] == "C:/cvs/alice.pdf" and results[0]["folder"] == "cvs"


def test_chunks_below_the_threshold_are_dropped(repository):
    assert [result["content"] for result in search(repository, min_similarity=0.5)] == ["exact match", "related"]
    assert search(repository, min_similarity=1.01) == []


def test_top_k_and_folder_filter(repository):
    assert len(search(repository, min_similarity=0.0, top_k=1)) == 1
    assert [result["content"] for result in search(repository, min_similarity=0.0, folder="jds")] == ["unrelated"]
    assert search(repository, min_similarity=0.0, folder="no-such-folder") == []


def test_models_with_different_dimensions_live_side_by_side(repository):
    repository.save_chunks([row("C:/cvs/alice.pdf", 0, "model b text", unit_vector(1, 0))],
                           embedding_model="model-b")

    model_b_results = repository.search_similar(unit_vector(1, 0), embedding_model="model-b", dimension=2,
                                                min_similarity=0.0)

    assert [result["content"] for result in model_b_results] == ["model b text"]
    assert len(search(repository, min_similarity=0.0)) == 3  # model-a rows are untouched
    assert repository.count_by_folder("model-b") == {"cvs": 1}


def test_re_saving_a_file_replaces_its_chunks_for_that_model_only(repository):
    repository.save_chunks([row("C:/cvs/alice.pdf", 0, "model b text", unit_vector(1, 0))],
                           embedding_model="model-b")

    repository.save_chunks([row("C:/cvs/alice.pdf", 0, "new version", unit_vector(1, 0, 0))],
                           embedding_model="model-a")

    contents = [chunk["content"] for chunk in repository.list_chunks("model-a")]
    assert contents == ["new version", "unrelated"]  # ordered by source (C:/cvs/... before C:/jds/...)
    assert repository.count_by_folder("model-b") == {"cvs": 1}


def test_count_by_folder(repository):
    assert repository.count_by_folder("model-a") == {"cvs": 2, "jds": 1}
    assert repository.count_by_folder("unknown-model") == {}


def test_delete_by_source_and_by_model(repository):
    assert repository.delete_by_source("C:/jds/jd.pdf") == 1
    assert repository.delete_by_embedding_model("model-a") == 2
    assert repository.list_chunks("model-a") == []


def test_ensure_embedding_index_creates_one_partial_hnsw_index_per_model(repository, database):
    repository.ensure_embedding_index("BAAI/bge-small-en-v1.5", 3)
    repository.ensure_embedding_index("BAAI/bge-small-en-v1.5", 3)  # idempotent

    with database.connection() as connection:
        index = connection.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_baai_bge_small_en_v1_5'"
        ).fetchone()
    assert "hnsw" in index["indexdef"]
    assert "vector(3)" in index["indexdef"]
    assert "BAAI/bge-small-en-v1.5" in index["indexdef"]
