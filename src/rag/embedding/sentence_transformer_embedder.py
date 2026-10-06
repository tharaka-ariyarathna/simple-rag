import numpy as np
from sentence_transformers import SentenceTransformer
from rag.constants import ACTIVE_EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL_PREFIXES


class SentenceTransformerEmbedder:
    """Any local sentence-transformers model. Model-specific prefixes come from EMBEDDING_MODEL_PREFIXES."""

    def __init__(self, model_name: str = ACTIVE_EMBEDDING_MODEL, batch_size: int = EMBEDDING_BATCH_SIZE):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)  # downloaded once, then loaded from cache
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        model_prefixes = EMBEDDING_MODEL_PREFIXES.get(model_name, {})
        self.query_prefix = model_prefixes.get("query", "")
        self.document_prefix = model_prefixes.get("document", "")

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            [self.document_prefix + text for text in texts],
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=True,
        )

    def embed_query(self, question: str) -> np.ndarray:
        return self.model.encode(self.query_prefix + question, normalize_embeddings=True)
