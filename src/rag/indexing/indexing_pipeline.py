from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
from collections import defaultdict
import os
from rag.indexing.chunker import TokenChunker
from rag.embedding.base import EmbeddingModel
from rag.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.repository.db import Database
from rag.repository.chunk_repository import ChunkRepository

load_dotenv()  # Load environment variables from .env file
docs_dir = os.getenv("DOCS_DIR")
database_url = os.environ["DATABASE_URL"]

def read_pdf_file(file_path: Path) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_documents(docs_dir: str) -> dict[str, list[dict]]:
    documents_map = defaultdict(list)
    for file_path in Path(docs_dir).rglob("*"):
        extension = file_path.suffix.lower()
        parent_name = file_path.parent.name
        if extension != ".pdf":
            print(f"Loading PDF file: {file_path}")
            continue
        text = read_pdf_file(file_path)
        if text.strip():  # Check if the text is not empty
            documents_map[parent_name].append({"source": str(file_path), "content": text})
    return documents_map

def chunk_documents(documents_map: dict[str, list[dict]], chunker: TokenChunker) -> list[dict]:
    chunk_rows = []
    for folder_name, documents in documents_map.items():
        for document in documents:
            for chunk_index, chunk_text in enumerate(chunker.chunk_text(document["content"])):
                chunk_rows.append({
                    "source": document["source"],
                    "folder": folder_name,
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                })
    return chunk_rows

def embed_chunk_rows(chunk_rows: list[dict], embedder: EmbeddingModel) -> list[dict]:
    embeddings = embedder.embed_documents([chunk_row["content"] for chunk_row in chunk_rows])
    print(f"Embeddings shape: {embeddings.shape}")
    for chunk_row, embedding in zip(chunk_rows, embeddings):  # same order as the input list
        chunk_row["embedding"] = embedding
    return chunk_rows

def search_test(chunk_rows: list[dict], embedder: EmbeddingModel, question: str, top_k: int = 3):
    """Quick in-memory sanity check before Postgres: cosine similarity = dot product (vectors are normalized)."""
    query_vector = embedder.embed_query(question)
    scored_rows = sorted(chunk_rows, key=lambda chunk_row: float(chunk_row["embedding"] @ query_vector), reverse=True)
    print(f"\nTest search: {question!r}")
    for chunk_row in scored_rows[:top_k]:
        similarity = float(chunk_row["embedding"] @ query_vector)
        print(f"  {similarity:.3f} [{chunk_row['folder']} #{chunk_row['chunk_index']}] {chunk_row['content'][:120]!r}")

def build_index():
    print("Generating Vectors...")
    documents = load_documents(docs_dir)
    print(f"Loaded {len(documents)} folders.")

    embedder: EmbeddingModel = SentenceTransformerEmbedder()
    chunker = TokenChunker(embedder.tokenizer)
    chunk_rows = chunk_documents(documents, chunker)
    print(f"Created {len(chunk_rows)} chunks.")

    chunk_rows = embed_chunk_rows(chunk_rows, embedder)
    search_test(chunk_rows, embedder, "Which projects used AWS Lambda and SQS?")
    with Database(database_url) as database:
        chunk_repository = ChunkRepository(database)
        chunk_repository.ensure_embedding_index(embedder.model_name, embedder.dimension)
        chunk_repository.save_chunks(chunk_rows, embedder.model_name)

if __name__ == "__main__":
    build_index()