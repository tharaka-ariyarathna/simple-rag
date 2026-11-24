"""Builds the index from the CV PDFs in DOCS_DIR: python -m rag.indexing.indexing_pipeline

Per CV: read -> hash (unchanged CVs are skipped) -> extract the profile with a LOCAL model -> cut into sections ->
redact (drop contact and referee sections, mask contacts and the name) -> chunk each section -> embed -> save the
candidate and its chunks in one transaction. Each CV is saved on its own, so a crash loses at most one CV and a
re-run continues where it stopped. CVs removed from the folder are left alone (delete them through the API)."""
import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from rag.bootstrap import setting
from rag.constants import PROFILE_LLM_MODEL
from rag.embedding.base import EmbeddingModel
from rag.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.indexing.chunker import TokenChunker
from rag.indexing.pdf_reader import read_pdf_file
from rag.indexing.profile_extractor import ExtractedProfile, ProfileExtractionError, ProfileExtractor
from rag.indexing.redaction import find_leaks, redact_sections
from rag.indexing.section_chunker import SectionChunker, split_sections
from rag.llm.base import RECOVERABLE_LLM_ERRORS
from rag.llm.ollama_client import OllamaLlmClient
from rag.repository.candidate_repository import CandidateRepository
from rag.repository.chunk_repository import ChunkRepository
from rag.repository.db import Database

logger = logging.getLogger(__name__)

NEW, UPDATED, UNCHANGED, FAILED = "new", "updated", "unchanged", "failed"


@dataclass
class IngestionSummary:
    outcomes: dict[str, list[str]] = field(default_factory=lambda: {NEW: [], UPDATED: [], UNCHANGED: [], FAILED: []})

    def add(self, outcome: str, source: str) -> None:
        self.outcomes[outcome].append(source)

    def line(self) -> str:
        return ", ".join(f"{len(sources)} {outcome}" for outcome, sources in self.outcomes.items())


def load_documents(docs_dir: str) -> list[dict]:
    """Every PDF under docs_dir with text in it: source (full path), folder (parent folder name), content."""
    documents = []
    for file_path in sorted(Path(docs_dir).rglob("*")):
        if file_path.suffix.lower() != ".pdf":
            continue
        text = read_pdf_file(file_path)
        if text.strip():
            documents.append({"source": str(file_path), "folder": file_path.parent.name, "content": text})
    return documents


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_chunk_rows(chunk_rows: list[dict], embedder: EmbeddingModel) -> list[dict]:
    if not chunk_rows:
        return chunk_rows
    embeddings = embedder.embed_documents([chunk_row["content"] for chunk_row in chunk_rows])
    for chunk_row, embedding in zip(chunk_rows, embeddings, strict=True):  # same order as the input list
        chunk_row["embedding"] = embedding
    return chunk_rows


class CvIngestor:
    """One CV at a time, from raw text to a saved candidate with redacted, embedded chunks."""

    def __init__(self, candidate_repository: CandidateRepository, embedder: EmbeddingModel,
                 profile_extractor: ProfileExtractor, section_chunker: SectionChunker):
        self.candidate_repository = candidate_repository
        self.embedder = embedder
        self.profile_extractor = profile_extractor
        self.section_chunker = section_chunker

    def ingest_all(self, documents: list[dict]) -> IngestionSummary:
        summary = IngestionSummary()
        for position, document in enumerate(documents, start=1):
            outcome = self.ingest(document)
            summary.add(outcome, document["source"])
            logger.info("[%d/%d] %s: %s", position, len(documents), outcome, Path(document["source"]).name)
        return summary

    def ingest(self, document: dict) -> str:
        source, text = document["source"], document["content"]
        text_hash = content_hash(text)
        existing = self.candidate_repository.get_by_source(source)
        if existing and existing["content_hash"] == text_hash and self.candidate_repository.has_chunks(
                existing["id"], self.embedder.model_name):
            return UNCHANGED
        try:
            profile = self.profile_extractor.extract(text)
        except (ProfileExtractionError, *RECOVERABLE_LLM_ERRORS) as error:  # retried on the next run
            logger.warning("Profile extraction failed for %s: %s", source, error)
            return FAILED
        if not profile.name:  # without the name it cannot be redacted, and the header line would leak it
            logger.warning("No name found in %s that appears in the CV, not indexed", source)
            return FAILED
        chunks = self.section_chunker.chunk_sections(redact_sections(split_sections(text), profile.name))
        leaking = [chunk.content for chunk in chunks if find_leaks(chunk.content, profile.name)]
        if leaking:  # never index personal data: better a missing CV than a leaking one
            logger.warning("Redaction left personal data in %s, not indexed: %s", source, leaking[0][:120])
            return FAILED
        chunk_rows = embed_chunk_rows([
            {"source": source, "folder": document["folder"], "chunk_index": index, "content": chunk.content,
             "metadata": {"section": chunk.section}}
            for index, chunk in enumerate(chunks)
        ], self.embedder)
        self.candidate_repository.save({
            "source": source, "content_hash": text_hash, "name": profile.name, "total_years": profile.total_years,
            "skills": profile.skills, "profile_model": profile.model,
            "jobs": [job.model_dump(mode="json") for job in profile.jobs],
        }, chunk_rows, self.embedder.model_name)
        return UPDATED if existing else NEW


def build_index() -> IngestionSummary:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    embedder = SentenceTransformerEmbedder()
    # Raw CVs hold personal data: profiles are extracted by a LOCAL model only, never a cloud provider
    profile_llm = OllamaLlmClient(model_name=setting(os.environ, "PROFILE_LLM_MODEL", PROFILE_LLM_MODEL),
                                  json_schema=ExtractedProfile.model_json_schema())
    with Database(os.environ["DATABASE_URL"]) as database:
        ChunkRepository(database).ensure_embedding_index(embedder.model_name, embedder.dimension)
        ingestor = CvIngestor(CandidateRepository(database), embedder, ProfileExtractor(profile_llm),
                              SectionChunker(TokenChunker(embedder.tokenizer)))
        try:
            summary = ingestor.ingest_all(load_documents(os.environ["DOCS_DIR"]))
        finally:
            profile_llm.close()
    logger.info("Done: %s", summary.line())
    return summary


if __name__ == "__main__":
    build_index()
