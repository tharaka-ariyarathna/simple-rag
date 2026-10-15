from typing import Protocol


class Reranker(Protocol):
    """What the RAG service relies on to reorder candidate chunks. Any reranking model must offer this."""

    model_name: str  # saved with eval runs, so runs with and without reranking are not compared as equals

    def score(self, query: str, passages: list[str]) -> list[float]: ...  # one score per passage, higher = better
