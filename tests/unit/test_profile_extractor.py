import json
from datetime import date
import pytest
from rag.indexing.profile_extractor import ProfileExtractionError, ProfileExtractor, parse_month
from tests.fakes import FakeLlmClient

TODAY = date(2026, 10, 1)
CV = """Ines Brooks
Skills: TensorFlow, k8s, Python
Machine Learning Engineer - Northfield Data
2024 - Present
Junior ML Engineer - Cobalt Harbor Studios
2021-03 - 2024"""


def reply(name="Ines Brooks", skills=("TensorFlow", "k8s", "Python"), jobs=None) -> str:
    jobs = jobs if jobs is not None else [
        {"title": "Machine Learning Engineer", "company": "Northfield Data", "start": "2024", "end": "Present"},
        {"title": "Junior ML Engineer", "company": "Cobalt Harbor Studios", "start": "2021-03", "end": "2024"},
    ]
    return json.dumps({"name": name, "skills": list(skills), "jobs": jobs})


def extract(llm_reply: str):
    llm = FakeLlmClient(answer=llm_reply)
    return ProfileExtractor(llm).extract(CV, today=TODAY), llm


def test_a_valid_reply_becomes_a_profile_with_total_years_and_canonical_skills():
    profile, llm = extract(reply())

    assert profile.name == "Ines Brooks"
    assert profile.skills == ["TensorFlow", "Kubernetes", "Python"]
    assert profile.total_years == 5.6  # 2021-03 to 2026-10, the two jobs touch in 2024
    assert [job.end for job in profile.jobs] == [None, date(2024, 1, 1)]
    assert llm.calls[0]["purpose"] == "profile_extraction" and "Northfield Data" in llm.calls[0]["user"]


@pytest.mark.parametrize("value, expected", [("2021", date(2021, 1, 1)), ("2021-03", date(2021, 3, 1)),
                                             ("2021/3", date(2021, 3, 1)), (" Present ", None)])
def test_dates_as_cvs_write_them(value, expected):
    assert parse_month(value) == expected


@pytest.mark.parametrize("value", ["March 2021", "2021-13", "soon"])
def test_unparseable_dates_are_rejected(value):
    with pytest.raises(ValueError):
        parse_month(value)


@pytest.mark.parametrize("bad_job, reason", [
    ({"title": "CTO", "company": "Invented Corp", "start": "2020", "end": "2021"}, "not in the CV"),
    ({"title": "x", "company": "Northfield Data", "start": "2023", "end": "2021"}, "impossible dates"),
    ({"title": "x", "company": "Northfield Data", "start": "2030", "end": "Present"}, "impossible dates"),
    ({"title": "x", "company": "Northfield Data", "start": "Present", "end": "Present"}, "date"),
])
def test_a_bad_job_is_dropped_and_the_rest_of_the_profile_kept(bad_job, reason):
    good = {"title": "Machine Learning Engineer", "company": "Northfield Data", "start": "2024", "end": "Present"}

    profile, _ = extract(reply(jobs=[good, bad_job]))

    assert [job.company for job in profile.jobs] == ["Northfield Data"]
    assert len(profile.rejected_jobs) == 1 and reason in profile.rejected_jobs[0]


def test_overlapping_jobs_are_counted_once():
    jobs = [{"title": "a", "company": "Northfield Data", "start": "2020", "end": "2024"},
            {"title": "b", "company": "Cobalt Harbor Studios", "start": "2022", "end": "2023"}]

    assert extract(reply(jobs=jobs))[0].total_years == 4.0


def test_names_and_skills_that_are_not_in_the_cv_are_not_trusted():
    profile, _ = extract(reply(name="Jane Smith", skills=("Python", "Rust")))

    assert profile.name is None  # never redact by a wrong name
    assert profile.skills == ["Python"]


@pytest.mark.parametrize("bad_reply", ["not json", '{"name": "x"}', '{"name": "x", "skills": "Python", "jobs": []}'])
def test_a_reply_that_is_not_the_expected_json_fails_the_cv(bad_reply):
    with pytest.raises(ProfileExtractionError):
        extract(bad_reply)


def test_a_word_glued_onto_the_name_is_dropped_when_the_rest_is_in_the_cv():
    profile, _ = extract(reply(name="Stockholm Ines Brooks"))  # the city from a two-column layout

    assert profile.name == "Ines Brooks"


def test_a_single_matching_word_is_not_enough_for_a_name():
    assert extract(reply(name="Ines Smith"))[0].name is None
