"""Ingesting one CV: hash check, local profile extraction, redaction, section chunks, one save per CV."""
import json
from rag.indexing.chunker import TokenChunker
from rag.indexing.indexing_pipeline import FAILED, NEW, UNCHANGED, UPDATED, CvIngestor, content_hash
from rag.indexing.profile_extractor import ProfileExtractor
from rag.indexing.section_chunker import SectionChunker
from rag.llm.base import LlmTimeoutError
from tests.fakes import FakeEmbedder, FakeLlmClient, WordTokenizer

CV = """Ines Brooks
Machine Learning Engineer | ines.brooks@example.com
Skills
Python, Spark
Experience
ML Engineer - Northfield Data
2021 - Present
Ines led the fraud model.
Referees
Jane Doe, jane@example.com, +46 70 123 45 67"""
PROFILE = json.dumps({"name": "Ines Brooks", "skills": ["Python", "Spark"],
                      "jobs": [{"title": "ML Engineer", "company": "Northfield Data", "start": "2021", "end": "Present"}]})


class FakeCandidateRepository:
    def __init__(self, existing: dict | None = None, has_chunks: bool = True):
        self.existing = existing
        self.chunks_present = has_chunks
        self.saved = []

    def get_by_source(self, source):
        return self.existing

    def has_chunks(self, candidate_id, embedding_model):
        return self.chunks_present

    def save(self, candidate, chunk_rows, embedding_model):
        self.saved.append((candidate, chunk_rows, embedding_model))
        return 7


def ingestor(repository, llm=None):
    return CvIngestor(repository, FakeEmbedder(), ProfileExtractor(llm or FakeLlmClient(answer=PROFILE)),
                      SectionChunker(TokenChunker(WordTokenizer(), chunk_size=50, chunk_overlap=0)))


def document(text=CV):
    return {"source": "C:/cvs/ines.pdf", "folder": "cvs", "content": text}


def test_a_new_cv_is_saved_with_its_profile_and_redacted_section_chunks():
    repository = FakeCandidateRepository()

    assert ingestor(repository).ingest(document()) == NEW

    [(candidate, rows, model)] = repository.saved
    assert (candidate["name"], candidate["skills"], candidate["content_hash"]) == (
        "Ines Brooks", ["Python", "Spark"], content_hash(CV))
    assert candidate["jobs"] == [{"title": "ML Engineer", "company": "Northfield Data", "start": "2021-01-01",
                                  "end": None}]
    assert [row["metadata"]["section"] for row in rows] == ["header", "skills", "experience"]  # referees dropped
    everything = " ".join(row["content"] for row in rows)
    assert "Ines" not in everything and "@" not in everything and "Jane" not in everything
    assert "the candidate led the fraud model." in everything
    assert all(row["embedding"] is not None for row in rows) and model == "fake-embedder"


def test_an_unchanged_cv_is_skipped_without_calling_the_llm():
    llm = FakeLlmClient(answer=PROFILE)
    repository = FakeCandidateRepository(existing={"id": 7, "content_hash": content_hash(CV)})

    assert ingestor(repository, llm).ingest(document()) == UNCHANGED
    assert llm.calls == [] and repository.saved == []


def test_a_changed_cv_or_one_without_chunks_is_processed_again():
    changed = FakeCandidateRepository(existing={"id": 7, "content_hash": "old"})
    no_chunks = FakeCandidateRepository(existing={"id": 7, "content_hash": content_hash(CV)}, has_chunks=False)

    assert ingestor(changed).ingest(document()) == UPDATED
    assert ingestor(no_chunks).ingest(document()) == UPDATED


def test_extraction_failures_are_reported_and_nothing_is_saved():
    for llm in (FakeLlmClient(answer="no json here"), FakeLlmClient(error=LlmTimeoutError("slow"))):
        repository = FakeCandidateRepository()

        assert ingestor(repository, llm).ingest(document()) == FAILED
        assert repository.saved == []


def test_a_cv_whose_name_cannot_be_found_is_not_indexed():
    repository = FakeCandidateRepository()
    llm = FakeLlmClient(answer=PROFILE.replace("Ines Brooks", "Someone Else"))

    assert ingestor(repository, llm).ingest(document()) == FAILED
    assert repository.saved == []
