"""JD matching: requirement parsing, verdict checking, scoring, and the LangGraph workflow with fakes."""
import json
from types import SimpleNamespace
import pytest
from rag.llm.base import LlmResult
from rag.matching.judge import CandidateJudge, checked_verdicts, score
from rag.matching.match_graph import MatchGraph
from rag.matching.requirements import RequirementExtractionError, RequirementExtractor, parse_requirements

REQUIREMENTS_REPLY = json.dumps({"requirements": [
    {"skill": "Python", "must_have": True, "min_years": 3, "synonyms": []},
    {"skill": "Kubernetes", "must_have": True, "min_years": None, "synonyms": ["K8s"]},
    {"skill": "Terraform", "must_have": False, "min_years": None, "synonyms": []},
], "min_total_years": None})
CVS = {1: "Python for 6 years. Kubernetes clusters. Terraform modules.",
       2: "Python scripting. Terraform modules.",
       3: "Java and Spring."}


def verdicts_reply(*triples):
    return json.dumps({"verdicts": [{"requirement": r, "verdict": v, "evidence": e} for r, v, e in triples]})


JUDGE_REPLIES = {
    1: verdicts_reply(("Python", "met", "Python for 6 years"), ("Kubernetes", "met", "Kubernetes clusters"),
                      ("Terraform", "met", "Terraform modules")),
    2: verdicts_reply(("Python", "partial", "Python scripting"), ("Kubernetes", "not_found", ""),
                      ("Terraform", "met", "Terraform modules")),
    3: verdicts_reply(("Python", "not_found", ""), ("Kubernetes", "not_found", ""), ("Terraform", "not_found", "")),
}


class ScriptedLlm:
    model_name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append(user_prompt)
        return LlmResult(text=self.replies.pop(0), model=self.model_name)

    def close(self):
        pass


class JudgeLlm:
    """Answers by which CV is in the prompt, so parallel judge calls can come in any order."""
    model_name = "judge"

    def __init__(self):
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append(user_prompt)
        candidate_id = next(cid for cid, text in CVS.items() if text in user_prompt)
        return LlmResult(text=JUDGE_REPLIES[candidate_id], model=self.model_name)

    def close(self):
        pass


class FakeSearch:
    """search_chunks by query: each skill finds the CVs that mention it; the whole JD finds everyone."""

    def __init__(self):
        self.embedder = SimpleNamespace(model_name="fake-embedder")
        self.calls = []

    def search_chunks(self, query, folder, top_k, candidate_ids=None, **options):
        self.calls.append({"query": query, "candidate_ids": candidate_ids, **options})
        word = query.split()[0].lower()
        hits = [cid for cid, text in CVS.items() if word in text.lower()] or list(CVS)
        return [{"candidate_id": cid} for cid in hits if candidate_ids is None or cid in candidate_ids]


class FakeCandidates:
    def __init__(self, years=None):
        self.years = years or {1: 6, 2: 2, 3: 9}

    def ids_with_min_years(self, min_years):
        return [cid for cid, years in self.years.items() if years >= min_years]

    def cv_text(self, candidate_id, embedding_model):
        return CVS[candidate_id]


def graph(extraction_replies=(REQUIREMENTS_REPLY,), candidates=None):
    search, judge_llm = FakeSearch(), JudgeLlm()
    extractor_llm = ScriptedLlm(extraction_replies)
    match_graph = MatchGraph(search, candidates or FakeCandidates(), RequirementExtractor(extractor_llm),
                             CandidateJudge(judge_llm))
    return match_graph, search, judge_llm, extractor_llm


def test_candidates_are_ranked_by_verdicts_with_evidence_and_gaps():
    match_graph, search, judge_llm, _ = graph()

    result = match_graph.match("We need Python and Kubernetes; Terraform is a plus.", top_n=10)

    assert [c["candidate_id"] for c in result["candidates"]] == [1, 2, 3]
    best, second, _ = result["candidates"]
    assert best["score"] == 1.0 and best["gaps"] == []
    assert second["missing_must_haves"] == ["Kubernetes"] and second["score"] == 0.4  # (1 + 0 + 1) / 5
    assert best["verdicts"][0] == {"requirement": "Python", "must_have": True, "verdict": "met",
                                   "evidence": "Python for 6 years"}
    assert len(judge_llm.calls) == 3  # one judge call per shortlisted candidate (the fan-out)
    assert {"Python", "Kubernetes K8s", "Terraform"} <= {call["query"] for call in search.calls}


def test_top_n_cuts_the_ranking():
    match_graph, *_ = graph()

    assert len(match_graph.match("jd", top_n=1)["candidates"]) == 1


def test_minimum_years_filters_candidates_before_searching():
    reply = json.loads(REQUIREMENTS_REPLY) | {"min_total_years": 5}
    match_graph, search, judge_llm, _ = graph(extraction_replies=(json.dumps(reply),))

    result = match_graph.match("5+ years required", top_n=10)

    assert all(call["candidate_ids"] == [1, 3] for call in search.calls)
    assert {c["candidate_id"] for c in result["candidates"]} == {1, 3} and len(judge_llm.calls) == 2


def test_a_bad_requirements_reply_gets_one_repair_attempt():
    match_graph, _, _, extractor_llm = graph(extraction_replies=("not json", REQUIREMENTS_REPLY))

    assert match_graph.match("jd", top_n=10)["candidates"]
    assert "could not be used" in extractor_llm.calls[1]


def test_two_bad_requirements_replies_fail_the_match():
    match_graph, *_ = graph(extraction_replies=("not json", '{"requirements": []}'))

    with pytest.raises(RequirementExtractionError):
        match_graph.match("jd", top_n=10)


def test_requirements_are_validated_and_trimmed():
    many = {"requirements": [{"skill": f"s{i}", "must_have": False} for i in range(20)]}
    assert len(parse_requirements(json.dumps(many)).requirements) == 15
    with pytest.raises(ValueError, match="must_have"):
        parse_requirements('{"requirements": [{"skill": "Python"}]}')


def test_a_quote_that_is_not_in_the_cv_turns_the_verdict_into_not_found():
    requirements = [{"skill": "Python", "must_have": True}, {"skill": "Go", "must_have": False}]
    reply = verdicts_reply(("python", "met", "Expert Python developer"), ("Go", "met", "Go services"))

    verdicts = checked_verdicts(reply, requirements, "Python for 6 years. Go services at scale.")

    assert [v["verdict"] for v in verdicts] == ["not_found", "met"]
    assert verdicts[0]["note"] == "the quoted evidence is not in the CV"


def test_an_unreadable_judge_reply_makes_every_requirement_not_found():
    verdicts = checked_verdicts("sorry", [{"skill": "Python", "must_have": True}], "Python")

    assert [v["verdict"] for v in verdicts] == ["not_found"]


def test_score_weights_must_haves_and_caps_a_missing_one():
    def v(verdict, must_have):
        return {"verdict": verdict, "must_have": must_have}

    assert score([v("met", True), v("partial", False)]) == round((2 + 0.5) / 3, 3)
    assert score([v("not_found", True), v("met", False), v("met", False), v("met", False)]) == 0.5
    assert score([]) == 0.0


def test_match_results_are_scored_judged_first_then_the_rest_with_shortlist_recall():
    from rag.evaluation.matching_eval import evaluate_match
    match_result = {"requirements": [{"skill": "Python"}], "shortlist": [1, 2],
                    "candidates": [{"candidate_id": 2, "score": 0.9}, {"candidate_id": 1, "score": 0.4}],
                    "whole_jd_order": [1, 3, 2]}
    labels = {"cv_a": 2, "cv_b": 0, "cv_c": 1}

    result = evaluate_match("jd_1", match_result, {1: "cv_a", 2: "cv_b", 3: "cv_c"}, labels, cutoff=3)

    assert [entry["cv"] for entry in result["details"]["top"]] == ["cv_b", "cv_a", "cv_c"]
    assert result["details"]["shortlist_recall"] == 0.5  # cv_c was relevant but never reached the judge
