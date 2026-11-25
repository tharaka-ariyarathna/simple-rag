"""Skill Finder tools (M1) as plain functions, with the fakes from the matching tests."""
from rag.agent import criteria as rules
from rag.agent.state import Requirement, SearchCriteria
from rag.agent.tools import SkillFinderTools
from rag.llm.base import LlmUnavailableError
from rag.matching.judge import CandidateJudge
from rag.matching.match_graph import MatchGraph
from rag.matching.requirements import RequirementExtractor
from tests.unit.test_matching import (
    CVS, REQUIREMENTS_REPLY, FakeCandidates, FakeSearch, JudgeLlm, ScriptedLlm,
)

JD = "We need Python and Kubernetes; Terraform is a plus. " * 8  # long enough to count as a JD


class Candidates(FakeCandidates):
    def all_names(self):
        return [{"candidate_id": cid} for cid in CVS]


class FailingLlm:
    model_name = "down"

    def generate(self, system_prompt, user_prompt):
        raise LlmUnavailableError("down")


def tools(extraction_replies=(REQUIREMENTS_REPLY,), judge_llm=None):
    search, candidates = FakeSearch(), Candidates()
    judge_llm = judge_llm or JudgeLlm()
    extractor = RequirementExtractor(ScriptedLlm(extraction_replies))
    match_graph = MatchGraph(search, candidates, extractor, CandidateJudge(judge_llm))
    return SkillFinderTools(search, candidates, extractor, match_graph), search, judge_llm


def criteria(*requirements, role=None):
    return SearchCriteria(role=role, requirements=list(requirements)).model_dump()


def req(name, kind="must", **extra):
    return Requirement(name=name, kind=kind, **extra)


# --- criteria rules ---------------------------------------------------------------------------------------------

def test_merge_replaces_same_name_and_keeps_the_rest():
    current = SearchCriteria(role="Backend developer", requirements=[req("Python"), req("AWS")], source="jd")
    new = SearchCriteria(requirements=[req("aws", "nice"), req("Docker")], source="conversation")

    merged = rules.merge(current, new)

    assert [(r.name, r.kind) for r in merged.requirements] == [("Python", "must"), ("aws", "nice"),
                                                               ("Docker", "must")]
    assert merged.role == "Backend developer" and merged.source == "mixed"


def test_edit_removes_adds_and_changes_kind_and_reports_unknown_names():
    current = SearchCriteria(requirements=[req("Python"), req("AWS"), req("Docker", "nice")])

    edited, unknown = rules.edit(current, [req("Go", "nice")], ["aws", "Rust"], {"Docker": "must"})

    assert [(r.name, r.kind) for r in edited.requirements] == [("Python", "must"), ("Docker", "must"),
                                                               ("Go", "nice")]
    assert unknown == ["Rust"]


def test_floor_needs_a_must_have_or_a_role_with_a_skill():
    assert rules.floor_problem(None)
    assert rules.floor_problem(SearchCriteria(role="Developer"))
    assert rules.floor_problem(SearchCriteria(requirements=[req("Python", "nice")]))
    assert rules.floor_problem(SearchCriteria(requirements=[req("Python")])) is None
    assert rules.floor_problem(SearchCriteria(role="Developer", requirements=[req("Python", "nice")])) is None


# --- extract_criteria / update_criteria -------------------------------------------------------------------------

def test_extract_criteria_maps_the_existing_output_and_keeps_a_long_text_as_the_jd():
    skill_tools, _, _ = tools()

    result = skill_tools.extract_criteria({}, JD, role="Platform engineer")

    names = [(r["name"], r["kind"]) for r in result.update["criteria"]["requirements"]]
    assert names == [("Python", "must"), ("Kubernetes", "must"), ("Terraform", "nice")]
    assert result.update["criteria"]["role"] == "Platform engineer" and result.update["jd_text"] == JD
    assert result.content["missing_info"] == []


def test_extract_criteria_merges_and_short_text_is_not_kept_as_a_jd():
    skill_tools, _, _ = tools()
    state = {"criteria": criteria(req("Go"), role="Developer")}

    result = skill_tools.extract_criteria(state, "Python and Kubernetes, Terraform a plus")

    assert [r["name"] for r in result.update["criteria"]["requirements"]][0] == "Go"
    assert "jd_text" not in result.update


def test_extract_criteria_returns_errors_as_text():
    skill_tools, _, _ = tools(extraction_replies=("not json", "still not json"))
    result = skill_tools.extract_criteria({}, "Python")
    assert "Could not extract" in result.content["error"] and result.update == {}


def test_update_criteria_needs_criteria_and_clears_the_jd():
    skill_tools, _, _ = tools()
    assert "extract_criteria" in skill_tools.update_criteria({}, remove=["AWS"]).content["error"]

    result = skill_tools.update_criteria({"criteria": criteria(req("Python"), req("AWS")), "jd_text": JD},
                                         remove=["AWS"], change_kind={"Python": "nice"}, role="Data engineer")

    assert [(r["name"], r["kind"]) for r in result.update["criteria"]["requirements"]] == [("Python", "nice")]
    assert result.update["criteria"]["role"] == "Data engineer" and result.update["jd_text"] is None
    assert "error" in skill_tools.update_criteria({"criteria": criteria(req("Python"))},
                                                  change_kind={"Python": "maybe"}).content


# --- count_matches / search_candidates --------------------------------------------------------------------------

def test_count_matches_counts_distinct_candidates_per_skill():
    skill_tools, _, _ = tools()

    result = skill_tools.count_matches({}, ["Terraform", "Kubernetes"])

    assert result.content == {"counts": {"Terraform": 2, "Kubernetes": 1}, "pool_size": 3}


def test_search_refuses_below_the_floor():
    skill_tools, search, _ = tools()

    result = skill_tools.search_candidates({"criteria": criteria(req("Python", "nice"))})

    assert "must-have" in result.content["error"] and search.calls == []


def test_search_stores_the_shortlist_and_resets_clarification_rounds():
    skill_tools, _, _ = tools()
    state = {"criteria": criteria(req("Python"), req("Kubernetes"), req("Terraform", "nice")),
             "clarification_rounds": 2}

    result = skill_tools.search_candidates(state)

    assert [c["candidate_id"] for c in result.update["candidates"]] == [1, 2, 3]
    assert result.update["candidates"][0]["found_for"] == ["Python", "Kubernetes", "Terraform"]
    assert result.update["clarification_rounds"] == 0 and result.update["ranking"] is None
    assert result.content["weak_coverage"] == ["Python", "Kubernetes", "Terraform"]  # pool of 3 < 3 per skill


# --- rank_candidates and the regression guard -------------------------------------------------------------------

def test_rank_needs_candidates():
    skill_tools, _, _ = tools()
    assert "search_candidates" in skill_tools.rank_candidates({"criteria": criteria(req("Python"))}) \
        .content["error"]


def test_ranking_through_the_tools_matches_match_graph_exactly():
    """Regression guard: the same JD gives the same order and scores as /match (scores here are x100)."""
    skill_tools, _, _ = tools(extraction_replies=(REQUIREMENTS_REPLY, REQUIREMENTS_REPLY))
    expected = skill_tools.match_graph.match(JD, top_n=10)["candidates"]

    state = {}
    for step in (lambda s: skill_tools.extract_criteria(s, JD), skill_tools.search_candidates,
                 skill_tools.rank_candidates):
        state |= step(state).update
    ranked = state["ranking"]["candidates"]

    assert [c["candidate_id"] for c in ranked] == [c["candidate_id"] for c in expected]
    assert [c["score"] for c in ranked] == [round(c["score"] * 100) for c in expected]
    assert [c["verdicts"] for c in ranked] == [c["verdicts"] for c in expected]


def test_rank_presents_strengths_and_gaps_with_must_haves_first_and_honours_top_k():
    skill_tools, _, _ = tools()
    state = {"criteria": criteria(req("Python"), req("Kubernetes"), req("Terraform", "nice"))}
    state |= skill_tools.search_candidates(state).update

    result = skill_tools.rank_candidates(state, top_k=2)

    second = result.content["candidates"][1]
    assert len(result.content["candidates"]) == 2
    assert second == {**second, "candidate_id": 2, "score": 40, "strengths": ["Terraform"],
                      "gaps": ["Python", "Kubernetes"], "missing_must_haves": ["Kubernetes"]}


def test_rank_returns_llm_failures_as_text():
    skill_tools, _, _ = tools(judge_llm=FailingLlm())
    state = {"criteria": criteria(req("Python")), "candidates": [{"candidate_id": 1, "found_for": []}]}
    assert "unavailable" in skill_tools.rank_candidates(state).content["error"]


# --- get_candidate_detail / search_cv_chunks --------------------------------------------------------------------

def test_detail_only_for_candidates_in_the_current_results():
    skill_tools, _, _ = tools()
    state = {"candidates": [{"candidate_id": 1, "found_for": []}]}

    assert skill_tools.get_candidate_detail(state, 1).content["cv"] == CVS[1]
    assert "not in the current results" in skill_tools.get_candidate_detail(state, 3).content["error"]


def test_search_cv_chunks_returns_compact_chunks():
    skill_tools, search, _ = tools()
    search.search_chunks = lambda query, folder, top_k, candidate_ids=None, **o: [
        {"candidate_id": 1, "content": "x" * 1000}]

    chunks = skill_tools.search_cv_chunks({}, "who knows Python?").content["chunks"]

    assert chunks == [{"candidate_id": 1, "text": "x" * 400}]
