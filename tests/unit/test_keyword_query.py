from rag.service.keyword_query import KeywordQueryBuilder
from tests.fakes import FakeKeywordRepository


def test_stop_words_and_question_framing_are_dropped():
    repository = FakeKeywordRepository({"python": 24, "best": 3}, total_chunks=100)

    terms = KeywordQueryBuilder(repository).terms_for("Who is the best candidate for a python based role?", "m")

    assert terms == ["python"]  # "best" is a stop word here, even though it occurs in few chunks


def test_filler_verbs_are_dropped_even_when_they_are_rare():
    repository = FakeKeywordRepository({"worked": 13, "kubernetes": 31}, total_chunks=160)  # counts from the CV set

    assert KeywordQueryBuilder(repository).terms_for("Who has worked with Kubernetes?", "m") == ["kubernetes"]


def test_terms_in_too_many_chunks_are_dropped():
    repository = FakeKeywordRepository({"kafka": 5, "developer": 80}, total_chunks=100)

    terms = KeywordQueryBuilder(repository, max_chunk_fraction=0.5).terms_for("kafka developer", "m")

    assert terms == ["kafka"]


def test_terms_found_in_no_chunk_are_dropped():
    repository = FakeKeywordRepository({"kafka": 5}, total_chunks=100)

    assert KeywordQueryBuilder(repository).terms_for("kafka cobol", "m") == ["kafka"]


def test_term_exactly_at_the_limit_is_kept():
    repository = FakeKeywordRepository({"java": 50}, total_chunks=100)

    assert KeywordQueryBuilder(repository, max_chunk_fraction=0.5).terms_for("java", "m") == ["java"]


def test_order_of_the_question_is_kept():
    repository = FakeKeywordRepository({"kafka": 5, "aws": 9, "lambda": 4}, total_chunks=100)

    assert KeywordQueryBuilder(repository).terms_for("lambda aws kafka", "m") == ["lambda", "aws", "kafka"]


def test_question_of_only_stop_words_skips_the_database_count():
    repository = FakeKeywordRepository()

    assert KeywordQueryBuilder(repository).terms_for("Who is the best?", "m") == []
    assert repository.counted_terms == []


def test_empty_index_gives_no_terms():
    repository = FakeKeywordRepository({"python": 0}, total_chunks=0)

    assert KeywordQueryBuilder(repository).terms_for("python", "m") == []
