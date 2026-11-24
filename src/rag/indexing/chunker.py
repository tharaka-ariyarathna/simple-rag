from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.constants import CHUNK_SEPARATORS, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS


class TokenChunker:
    """Splits text into chunks of at most chunk_size tokens, trying paragraphs, then lines, sentences and words.

    The splitting itself is LangChain's RecursiveCharacterTextSplitter; tokens are counted with the embedding model's
    own tokenizer, so a chunk never exceeds what the model reads. It replaced a hand-written splitter that appended a
    separator to the last piece, adding text (a full stop) the document never had."""

    def __init__(self, tokenizer, chunk_size: int = CHUNK_SIZE_TOKENS, chunk_overlap: int = CHUNK_OVERLAP_TOKENS):
        self.tokenizer = tokenizer
        self.splitter = RecursiveCharacterTextSplitter(
            separators=CHUNK_SEPARATORS, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            length_function=self.count_tokens, keep_separator="end", strip_whitespace=True,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_text(self, text: str) -> list[str]:
        return [chunk for chunk in self.splitter.split_text(text) if chunk.strip()]
