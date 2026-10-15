"""Indexing helpers, the embedder wrapper, the matching dataset loader and run settings.
Heavy or external pieces (PDF parsing, the sentence-transformers model, git) are replaced with monkeypatch."""
import importlib
import json
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from rag.constants import BGE_QUERY_PREFIX
from rag.embedding import sentence_transformer_embedder
from rag.evaluation import matching_dataset, run_settings
from tests.fakes import FakeEmbedder, FakeReranker, WordTokenizer, unit_vector
from rag.indexing.chunker import TokenChunker


@pytest.fixture(scope="module")
def pipeline():
    # indexing_pipeline reads DATABASE_URL when it is imported, so it needs a value before the import
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("DATABASE_URL", "postgresql://unused-by-these-tests")
        yield importlib.import_module("rag.indexing.indexing_pipeline")


class TestIndexingPipeline:
    def test_load_documents_reads_pdfs_grouped_by_parent_folder(self, pipeline, tmp_path, monkeypatch):
        (tmp_path / "cvs").mkdir()
        (tmp_path / "cvs" / "alice.pdf").write_bytes(b"")
        (tmp_path / "cvs" / "notes.txt").write_text("not a pdf")
        (tmp_path / "jds").mkdir()
        (tmp_path / "jds" / "Scanned.PDF").write_bytes(b"")
        texts = {"alice.pdf": "Alice CV text", "Scanned.PDF": "   "}
        monkeypatch.setattr(pipeline, "read_pdf_file", lambda path: texts[path.name])

        documents = pipeline.load_documents(str(tmp_path))

        assert dict(documents) == {"cvs": [{"source": str(tmp_path / "cvs" / "alice.pdf"), "content": "Alice CV text"}]}

    def test_chunk_documents_numbers_chunks_per_document(self, pipeline):
        chunker = TokenChunker(WordTokenizer(), chunk_size=2, chunk_overlap=0)
        documents = {"cvs": [{"source": "a.pdf", "content": "one two\n\nthree"},
                             {"source": "b.pdf", "content": "four"}]}

        rows = pipeline.chunk_documents(documents, chunker)

        assert [(row["source"], row["folder"], row["chunk_index"]) for row in rows] == [
            ("a.pdf", "cvs", 0), ("a.pdf", "cvs", 1), ("b.pdf", "cvs", 0)]

    def test_embed_chunk_rows_attaches_each_vector_to_its_row(self, pipeline):
        embedder = FakeEmbedder({"alpha": unit_vector(1, 0, 0), "beta": unit_vector(0, 1, 0)})
        rows = [{"content": "alpha"}, {"content": "beta"}]

        pipeline.embed_chunk_rows(rows, embedder)

        assert np.array_equal(rows[0]["embedding"], unit_vector(1, 0, 0))
        assert np.array_equal(rows[1]["embedding"], unit_vector(0, 1, 0))


class TestSentenceTransformerEmbedder:
    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name
            self.tokenizer = "the-tokenizer"
            self.encoded = []

        def get_sentence_embedding_dimension(self):
            return 384

        def encode(self, texts, **options):
            self.encoded.append((texts, options))
            return np.zeros(384)

    @pytest.fixture(autouse=True)
    def no_real_model(self, monkeypatch):
        monkeypatch.setattr(sentence_transformer_embedder, "SentenceTransformer", self.FakeModel)

    def test_bge_queries_get_the_retrieval_prefix_and_documents_do_not(self):
        embedder = sentence_transformer_embedder.SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")

        embedder.embed_query("python?")
        embedder.embed_documents(["Alice knows Python."])

        (query, query_options), (documents, document_options) = embedder.model.encoded
        assert query == BGE_QUERY_PREFIX + "python?"
        assert documents == ["Alice knows Python."]
        assert query_options["normalize_embeddings"] and document_options["normalize_embeddings"]

    def test_unknown_models_get_no_prefix_and_report_their_dimension(self):
        embedder = sentence_transformer_embedder.SentenceTransformerEmbedder("some/other-model")

        embedder.embed_query("python?")

        assert embedder.model.encoded[0][0] == "python?"
        assert (embedder.model_name, embedder.dimension, embedder.tokenizer) == ("some/other-model", 384,
                                                                                 "the-tokenizer")


def test_matching_dataset_keeps_labelled_jds_that_have_a_pdf(tmp_path, monkeypatch):
    (tmp_path / "jds").mkdir()
    (tmp_path / "jds" / "jd_001.pdf").write_bytes(b"")
    (tmp_path / "labels.json").write_text(json.dumps({"jd_002": {"cv_001": 1}, "jd_001": {"cv_002": 2}}))
    monkeypatch.setattr(matching_dataset, "read_pdf_file", lambda path: f"text of {path.stem}")

    dataset = matching_dataset.load_matching_dataset(tmp_path)

    assert dataset.jd_texts == {"jd_001": "text of jd_001"}
    assert dataset.labels == {"jd_001": {"cv_002": 2}}
    assert dataset.skipped_jd_ids == ["jd_002"]


class TestRunSettings:
    def test_git_commit_is_none_when_git_is_not_available(self, monkeypatch):
        def no_git(*args, **kwargs):
            raise FileNotFoundError("git")
        monkeypatch.setattr(subprocess, "run", no_git)

        assert run_settings.current_git_commit() is None

    def test_git_commit_is_the_short_hash(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="abc1234\n"))

        assert run_settings.current_git_commit() == "abc1234"

    def components(self, keyword_query_builder=None, reranker=None):
        rag_service = SimpleNamespace(embedder=FakeEmbedder(model_name="emb"), keyword_query_builder=keyword_query_builder,
                                      reranker=reranker, rerank_candidates=30)
        return SimpleNamespace(rag_service=rag_service, llm_client=SimpleNamespace(model_name="llm"))

    def test_settings_name_the_models_and_retrieval_parameters(self):
        settings = run_settings.collect_settings(self.components(), top_k=7)

        assert (settings["embedding_model"], settings["llm_model"], settings["top_k"]) == ("emb", "llm", 7)
        assert {"chunk_size_tokens", "chunk_overlap_tokens", "min_similarity"} <= settings.keys()
        assert (settings["keyword_search"], settings["reranker_model"], settings["rerank_candidates"]) == (
            False, None, None)

    def test_settings_record_hybrid_search_and_the_reranker(self):
        settings = run_settings.collect_settings(self.components(object(), FakeReranker(model_name="bge")), top_k=10)

        assert (settings["keyword_search"], settings["reranker_model"], settings["rerank_candidates"]) == (
            True, "bge", 30)
