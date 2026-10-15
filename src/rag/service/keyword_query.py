import logging
from rag.constants import KEYWORD_MAX_CHUNK_FRACTION, KEYWORD_STOP_WORDS
from rag.repository.chunk_repository import ChunkRepository

logger = logging.getLogger(__name__)


class KeywordQueryBuilder:
    """Picks the words of a question worth a keyword search.

    "Who is the best candidate for a python based role?" -> ["python"]: stop words go, and so do words found in
    so many chunks that they cannot tell chunks apart. The words come from the database (question_lexemes), so
    they are normalised exactly like the stored chunks (C# -> csharp, ...)."""

    def __init__(self, repository: ChunkRepository, stop_words: frozenset[str] = KEYWORD_STOP_WORDS,
                 max_chunk_fraction: float = KEYWORD_MAX_CHUNK_FRACTION):
        self.repository = repository
        self.stop_words = stop_words
        self.max_chunk_fraction = max_chunk_fraction

    def terms_for(self, question: str, embedding_model: str) -> list[str]:
        candidates = [term for term in self.repository.question_lexemes(question) if term not in self.stop_words]
        if not candidates:
            return []
        chunk_counts, total_chunks = self.repository.term_chunk_counts(candidates, embedding_model)
        if total_chunks == 0:
            return []
        terms = [term for term in candidates
                 if 0 < chunk_counts.get(term, 0) <= self.max_chunk_fraction * total_chunks]
        logger.debug("Keyword terms for %r: %s (candidates %s)", question, terms, chunk_counts)
        return terms
