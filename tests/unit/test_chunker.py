import pytest
from rag.indexing.chunker import TokenChunker
from tests.fakes import CharTokenizer, WordTokenizer


def words_of(chunks: list[str]) -> list[list[str]]:
    return [WordTokenizer().encode(chunk) for chunk in chunks]


def test_text_that_fits_is_one_chunk():
    chunker = TokenChunker(WordTokenizer(), chunk_size=10, chunk_overlap=2)

    assert chunker.chunk_text("a short text") == ["a short text"]


def test_count_tokens_uses_the_injected_tokenizer():
    assert TokenChunker(WordTokenizer()).count_tokens("three little words") == 3


def test_paragraphs_are_kept_whole_when_each_fits():
    chunker = TokenChunker(WordTokenizer(), chunk_size=4, chunk_overlap=0)
    text = "one two three\n\nfour five six"

    assert words_of(chunker.chunk_text(text)) == [["one", "two", "three"], ["four", "five", "six"]]


def test_consecutive_chunks_share_an_overlap():
    chunker = TokenChunker(WordTokenizer(), chunk_size=4, chunk_overlap=2)
    text = " ".join(f"w{number}" for number in range(1, 11))

    assert words_of(chunker.chunk_text(text)) == [
        ["w1", "w2", "w3", "w4"],
        ["w3", "w4", "w5", "w6"],
        ["w5", "w6", "w7", "w8"],
        ["w7", "w8", "w9", "w10"],
    ]


def test_chunks_stay_within_size_plus_overlap_and_lose_no_words():
    chunker = TokenChunker(WordTokenizer(), chunk_size=8, chunk_overlap=3)
    sentences = [" ".join(f"s{sentence}w{word}" for word in range(1, 6)) for sentence in range(1, 8)]
    text = ". ".join(sentences) + "."

    chunks = chunker.chunk_text(text)

    # The overlap kept from the previous chunk is added on top of a full-size piece, so the bound is size + overlap
    assert all(chunker.count_tokens(chunk) <= 8 + 3 for chunk in chunks)
    original_words = {word.strip(".") for word in text.split()}
    chunk_words = {word.strip(".") for chunk in words_of(chunks) for word in chunk}
    assert chunk_words == original_words


def test_hard_cut_by_tokens_when_no_separator_is_left():
    chunker = TokenChunker(CharTokenizer(), chunk_size=4, chunk_overlap=0)

    assert chunker.split_into_pieces("abcdefghij", separators=[]) == ["abcd", "efgh", "ij"]


def test_blank_parts_between_separators_are_dropped():
    chunker = TokenChunker(WordTokenizer(), chunk_size=1, chunk_overlap=0)

    assert words_of(chunker.chunk_text("alpha\n\n\n\n\n\nbeta")) == [["alpha"], ["beta"]]


@pytest.mark.xfail(strict=True, reason="Known bug: split_into_pieces appends the separator to the last part too, "
                                       "so splitting on '. ' adds a full stop the document never had")
def test_chunking_does_not_add_text_that_is_not_in_the_document():
    chunker = TokenChunker(WordTokenizer(), chunk_size=3, chunk_overlap=0)
    text = "one two three. four five six"  # no full stop after "six"

    chunks = chunker.chunk_text(text)

    assert chunks[-1].endswith("six")
