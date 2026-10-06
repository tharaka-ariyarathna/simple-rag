from typing import Protocol
import numpy as np


class EmbeddingModel(Protocol):
    """What indexing and the RAG service rely on. Any embedding provider must offer these."""

    model_name: str   # stored with every chunk so vectors from different models never get compared
    dimension: int    # vector length, e.g. 384 for bge-small

    @property
    def tokenizer(self): ...  # used by the chunker to count tokens the way this model does

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...  # shape (len(texts), dimension)

    def embed_query(self, question: str) -> np.ndarray: ...  # shape (dimension,)
