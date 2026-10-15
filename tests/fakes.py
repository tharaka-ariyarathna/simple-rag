"""Hand-written stand-ins for the slow or external parts (embedding model, LLM, database).

They satisfy the same Protocols as the real classes (EmbeddingModel, LlmClient) or the same duck-typed methods
(ChunkRepository, LedgerRepository), so they are injected through the normal constructors, no patching needed.
Each fake also records how it was called, so a test can check what the code under test asked for.
"""
import numpy as np
from rag.llm.base import LlmResult
from rag.llm.call_context import current_purpose, current_rag_request_id


class WordTokenizer:
    """One token per whitespace-separated word, so token counts are easy to work out by hand.
    Tokens without a letter or digit (a lone ".") are not counted: the chunker currently appends a stray ". "
    to the last piece of a text (see the xfail test in test_chunker.py), and counting it would make every
    other chunker test depend on that bug."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return [word for word in text.split() if any(character.isalnum() for character in word)]

    def decode(self, token_ids: list[str]) -> str:
        return " ".join(token_ids)


class CharTokenizer:
    """One token per character: lets a test reach the "no separator left, hard cut" branch of the chunker."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return list(text)

    def decode(self, token_ids: list[str]) -> str:
        return "".join(token_ids)


def unit_vector(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class FakeEmbedder:
    """Returns the vector a test assigned to a text, so similarities are known in advance."""

    def __init__(self, vectors: dict[str, np.ndarray] | None = None, model_name: str = "fake-embedder",
                 dimension: int = 3):
        self.vectors = vectors or {}
        self.model_name = model_name
        self.dimension = dimension
        self.queries: list[str] = []

    @property
    def tokenizer(self):
        return WordTokenizer()

    def vector_for(self, text: str) -> np.ndarray:
        return self.vectors.get(text, unit_vector(*([1.0] * self.dimension)))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.vector_for(text) for text in texts])

    def embed_query(self, question: str) -> np.ndarray:
        self.queries.append(question)
        return self.vector_for(question)


class FakeLlmClient:
    """Answers with a fixed text, or raises a given error. Records each call and the call context it ran in."""

    def __init__(self, answer: str = "An answer [1].", error: Exception | None = None, model_name: str = "fake-llm",
                 input_tokens: int | None = 10, output_tokens: int | None = 5):
        self.answer = answer
        self.error = error
        self.model_name = model_name
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls: list[dict] = []
        self.closed = False

    def generate(self, system_prompt: str, user_prompt: str) -> LlmResult:
        self.calls.append({
            "system": system_prompt,
            "user": user_prompt,
            "purpose": current_purpose.get(),
            "rag_request_id": current_rag_request_id.get(),
        })
        if self.error is not None:
            raise self.error
        return LlmResult(text=self.answer, model=self.model_name, input_tokens=self.input_tokens,
                         output_tokens=self.output_tokens)

    def close(self) -> None:
        self.closed = True


class FakeChunkRepository:
    """search_similar returns the given rows and remembers the arguments it got."""

    def __init__(self, rows: list[dict] | None = None, folder_counts: dict[str, int] | None = None):
        self.rows = rows or []
        self.folder_counts = folder_counts or {}
        self.search_calls: list[dict] = []

    def search_similar(self, query_vector, embedding_model: str, dimension: int, **options) -> list[dict]:
        self.search_calls.append({"query_vector": query_vector, "embedding_model": embedding_model,
                                  "dimension": dimension, **options})
        return self.rows

    def count_by_folder(self, embedding_model: str) -> dict[str, int]:
        return self.folder_counts


class FakeReranker:
    """Scores a passage by the given table (default 0), so the reranked order is known in advance."""

    def __init__(self, scores: dict[str, float] | None = None, model_name: str = "fake-reranker"):
        self.scores = scores or {}
        self.model_name = model_name
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, passages))
        return [self.scores.get(passage, 0.0) for passage in passages]


class FakeKeywordRepository:
    """question_lexemes splits on spaces (lower-cased, de-duplicated); term_chunk_counts reads a given table."""

    def __init__(self, chunk_counts: dict[str, int] | None = None, total_chunks: int = 100):
        self.chunk_counts = chunk_counts or {}
        self.total_chunks = total_chunks
        self.counted_terms: list[list[str]] = []

    def question_lexemes(self, question: str) -> list[str]:
        return list(dict.fromkeys(question.lower().strip("?").split()))

    def term_chunk_counts(self, terms: list[str], embedding_model: str) -> tuple[dict[str, int], int]:
        self.counted_terms.append(terms)
        return {term: self.chunk_counts.get(term, 0) for term in terms}, self.total_chunks


class FakeLedgerRepository:
    """Keeps ledger rows in lists. fail=True makes every write raise, like a database that went away."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.requests: dict[int, dict] = {}
        self.llm_calls: list[dict] = []

    def insert_rag_request(self, **fields) -> int:
        if self.fail:
            raise RuntimeError("ledger is down")
        rag_request_id = len(self.requests) + 1
        self.requests[rag_request_id] = dict(fields)
        return rag_request_id

    def finish_rag_request(self, rag_request_id: int, answer, refusal_reason, total_latency_ms: int) -> None:
        if self.fail:
            raise RuntimeError("ledger is down")
        self.requests[rag_request_id].update(
            answer=answer, refusal_reason=refusal_reason, total_latency_ms=total_latency_ms, finished=True
        )

    def insert_llm_call(self, **fields) -> None:
        if self.fail:
            raise RuntimeError("ledger is down")
        self.llm_calls.append(fields)


def chunk_row(source: str, content: str = "Some text.", folder: str = "docs", similarity: float = 0.9,
              chunk_index: int = 0) -> dict:
    """A row shaped like ChunkRepository.search_similar returns it."""
    return {"source": source, "folder": folder, "chunk_index": chunk_index, "content": content,
            "similarity": similarity}
