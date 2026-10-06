from rag.constants import CHUNK_SEPARATORS, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS


class TokenChunker:
    def __init__(self, tokenizer, chunk_size: int = CHUNK_SIZE_TOKENS, chunk_overlap: int = CHUNK_OVERLAP_TOKENS):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split_into_pieces(self, text: str, separators: list[str]) -> list[str]:
        """Break text into pieces that each fit in chunk_size, trying big separators first."""
        if self.count_tokens(text) <= self.chunk_size:
            return [text]
        if not separators:  # no separator left: hard cut by tokens
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            return [
                self.tokenizer.decode(token_ids[start:start + self.chunk_size])
                for start in range(0, len(token_ids), self.chunk_size)
            ]
        separator, *smaller_separators = separators
        pieces = []
        for part in text.split(separator):
            if part.strip():
                pieces.extend(self.split_into_pieces(part + separator, smaller_separators))
        return pieces

    def chunk_text(self, text: str) -> list[str]:
        pieces = self.split_into_pieces(text, CHUNK_SEPARATORS)
        chunks, current_pieces, current_tokens = [], [], 0

        for piece in pieces:
            piece_tokens = self.count_tokens(piece)
            if current_tokens + piece_tokens > self.chunk_size and current_pieces:
                chunks.append("".join(current_pieces).strip())
                # keep trailing pieces as overlap for the next chunk
                while current_pieces and current_tokens > self.chunk_overlap:
                    current_tokens -= self.count_tokens(current_pieces.pop(0))
            current_pieces.append(piece)
            current_tokens += piece_tokens

        if current_pieces:
            chunks.append("".join(current_pieces).strip())
        return chunks
