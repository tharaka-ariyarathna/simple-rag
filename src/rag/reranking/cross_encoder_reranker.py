from sentence_transformers import CrossEncoder
from rag.constants import RERANK_BATCH_SIZE, RERANKER_MODEL


class CrossEncoderReranker:
    """Any local sentence-transformers cross-encoder (e.g. BAAI/bge-reranker-base).

    Unlike the embedding model, it reads the query and a passage together, so it can weigh every word of one
    against the other. Its scores are relevance logits: only comparable within one query, never a threshold."""

    def __init__(self, model_name: str = RERANKER_MODEL, batch_size: int = RERANK_BATCH_SIZE):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)  # downloaded once, then loaded from cache
        self.batch_size = batch_size

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict([(query, passage) for passage in passages], batch_size=self.batch_size,
                                    show_progress_bar=False)
        return [float(score) for score in scores]
