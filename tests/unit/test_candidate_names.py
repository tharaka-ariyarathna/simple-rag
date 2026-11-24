from rag.service.candidate_names import NameResolver, candidate_ids_in, with_names


class FakeDirectory:
    def __init__(self, rows):
        self.rows = rows

    def all_names(self):
        return self.rows

    def names_by_id(self, candidate_ids):
        return {row["id"]: row["name"] for row in self.rows if row["id"] in candidate_ids}


def person(candidate_id, name, years=5, skills=("Python",)):
    return {"id": candidate_id, "name": name, "total_years": years, "skills": list(skills)}


PEOPLE = [person(17, "Ines Brooks", 8, ["Spark", "Python"]), person(52, "Ines Brooks", 3, ["SQL"]),
          person(23, "Leah Walsh"), person(30, "Grace Kim"), person(31, None)]


def resolve(question, known=None, people=PEOPLE):
    return NameResolver(FakeDirectory(people)).resolve(question, known)


def test_a_unique_full_name_becomes_the_candidate_id():
    resolution = resolve("What is Leah Walsh's experience with Spark?")

    assert resolution.question == "What is candidate 23's experience with Spark?"
    assert resolution.candidate_ids == [23] and resolution.clarification is None


def test_a_capitalised_first_or_last_name_alone_is_enough_when_unique():
    assert resolve("Does Walsh know Go?").candidate_ids == [23]
    assert resolve("does grace know go?").candidate_ids == []  # lower case: too likely to be an ordinary word


def test_a_shared_name_is_narrowed_by_the_candidates_already_mentioned():
    resolution = resolve("What about Ines Brooks?", known=[52, 23])

    assert resolution.question == "What about candidate 52?" and resolution.candidate_ids == [52]


def test_a_shared_name_without_context_asks_instead_of_guessing():
    resolution = resolve("What does Ines Brooks know?")

    assert resolution.candidate_ids == []
    assert "candidate 17 (8 years, Spark, Python)" in resolution.clarification
    assert "candidate 52 (3 years, SQL)" in resolution.clarification
    assert "Ines" not in resolution.clarification  # it is stored in the history a cloud LLM reads


def test_several_people_and_explicit_ids_are_all_collected():
    resolution = resolve("Compare candidate 30 with Leah Walsh")

    assert resolution.candidate_ids == [30, 23]


def test_ids_in_text_and_names_put_back():
    assert candidate_ids_in("Candidate 17 and candidate #23, then candidate 17 again") == [17, 23]
    assert with_names("candidate 17 beats Candidate 99", {17: "Ines Brooks"}) == (
        "Ines Brooks (candidate 17) beats Candidate 99")
