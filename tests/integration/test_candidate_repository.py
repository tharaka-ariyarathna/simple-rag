"""CandidateRepository and migration 008 against the real test database."""
from rag.repository.candidate_repository import CandidateRepository
from rag.repository.chunk_repository import ChunkRepository
from tests.fakes import unit_vector

MODEL = "test-model"


def candidate(source="C:/cvs/ines.pdf", name="Ines Brooks", content_hash="h1"):
    return {"source": source, "content_hash": content_hash, "name": name, "total_years": 5.5,
            "skills": ["Python"], "jobs": [{"title": "ML Engineer"}], "profile_model": "llama3.2:3b"}


def chunk(content, section="skills"):
    return {"source": "C:/cvs/ines.pdf", "folder": "cvs", "chunk_index": 0, "content": content,
            "embedding": unit_vector(1, 0, 0), "metadata": {"section": section}}


def rows(database, query, *params):
    with database.connection() as connection:
        return connection.execute(query, params).fetchall()


def test_save_upserts_the_candidate_and_replaces_its_chunks(database):
    repository = CandidateRepository(database)
    first_id = repository.save(candidate(), [chunk("old text")], MODEL)

    second_id = repository.save(candidate(content_hash="h2"), [chunk("new text", "experience")], MODEL)

    assert first_id == second_id
    assert repository.get(first_id)["content_hash"] == "h2"
    assert rows(database, "SELECT content, metadata, candidate_id FROM chunks") == [
        {"content": "new text", "metadata": {"section": "experience"}, "candidate_id": first_id}]
    assert repository.get_by_source("C:/cvs/ines.pdf")["skills"] == ["Python"]
    assert repository.has_chunks(first_id, MODEL) and not repository.has_chunks(first_id, "other-model")


def test_chunks_indexed_before_candidates_existed_are_replaced_by_source(database):
    ChunkRepository(database).save_chunks([chunk("pre-candidate text")], MODEL)  # no candidate_id

    CandidateRepository(database).save(candidate(), [chunk("new text")], MODEL)

    assert [row["content"] for row in rows(database, "SELECT content FROM chunks")] == ["new text"]


def test_names_are_found_ignoring_case_by_full_first_or_last_name_and_can_be_shared(database):
    repository = CandidateRepository(database)
    ines = repository.save(candidate(), [], MODEL)
    other_ines = repository.save(candidate(source="C:/cvs/ines2.pdf"), [], MODEL)
    repository.save(candidate(source="C:/cvs/leah.pdf", name="Leah Walsh"), [], MODEL)

    assert [row["id"] for row in repository.find_by_name("ines brooks")] == [ines, other_ines]
    assert [row["id"] for row in repository.find_by_name("Brooks")] == [ines, other_ines]
    assert [row["name"] for row in repository.find_by_name("WALSH")] == ["Leah Walsh"]
    assert repository.find_by_name("Brook") == []
    assert repository.names_by_id([ines]) == {ines: "Ines Brooks"}


def test_deleting_a_candidate_removes_its_chunks_and_embeddings(database):
    repository = CandidateRepository(database)
    candidate_id = repository.save(candidate(), [chunk("text")], MODEL)

    assert repository.delete(candidate_id) is True
    assert repository.delete(candidate_id) is False
    assert rows(database, "SELECT count(*) AS n FROM chunks") == [{"n": 0}]


def test_profile_queries_used_by_matching(database):
    repository = CandidateRepository(database)
    senior = repository.save(candidate() | {"total_years": 8}, [chunk("Skills\nPython"), chunk("Experience\nLed") |
                                                                {"chunk_index": 1}], MODEL)
    junior = repository.save(candidate(source="C:/cvs/leah.pdf", name="Leah Walsh") | {"total_years": 1}, [], MODEL)

    assert repository.ids_with_min_years(5) == [senior]
    assert repository.cv_text(senior, MODEL) == "Skills\nPython\n\nExperience\nLed"
    assert repository.sources_by_id([junior]) == {junior: "C:/cvs/leah.pdf"}
    assert [row["name"] for row in repository.all_names()] == ["Ines Brooks", "Leah Walsh"]


def test_search_can_be_limited_to_some_candidates(database):
    candidates = CandidateRepository(database)
    ines = candidates.save(candidate(), [chunk("Python and Spark")], MODEL)
    leah = candidates.save(candidate(source="C:/cvs/leah.pdf", name="Leah Walsh"),
                           [chunk("Python and Go") | {"source": "C:/cvs/leah.pdf"}], MODEL)
    chunks = ChunkRepository(database)

    everyone = chunks.search_similar(unit_vector(1, 0, 0), MODEL, 3, min_similarity=0)
    only_leah = chunks.search_similar(unit_vector(1, 0, 0), MODEL, 3, min_similarity=0, candidate_ids=[leah])

    assert {row["candidate_id"] for row in everyone} == {ines, leah}
    assert [row["candidate_id"] for row in only_leah] == [leah]
