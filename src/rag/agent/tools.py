"""The Skill Finder tools as plain functions (Milestone 1): no agent yet.

Each tool takes the current AgentState and its own arguments, and returns a ToolResult:
- content: what the LLM will read. Compact (ids, scores, short quotes). Errors are text here, never raised, so
  the agent can recover.
- update: the state keys the tool changes. In M2 this becomes LangGraph's Command(update=...).

Search and scoring are not reimplemented: the tools call MatchGraph's own methods, so the ranking is the same as
/match for the same criteria."""
import re
from dataclasses import dataclass, field
from pydantic import ValidationError
from rag.agent import criteria as criteria_rules
from rag.agent.state import AgentState, Requirement, SearchCriteria
from rag.constants import MATCH_EXTRACTION_MAX_ATTEMPTS, MATCH_SHORTLIST_SIZE
from rag.llm.base import RECOVERABLE_LLM_ERRORS
from rag.matching.judge import MET, NOT_FOUND
from rag.matching.match_graph import JudgeTask, MatchGraph
from rag.matching.requirements import RequirementExtractor, parse_requirements
from rag.repository.candidate_repository import CandidateRepository
from rag.service.rag_service import RagService

SEARCH_LIMIT_MAX = 30  # candidates kept from a search
RANK_TOP_K_MAX = 10  # candidates judged and shown; the judge reads each CV in full, one LLM call each
WEAK_COVERAGE = 3  # a requirement found in fewer candidates than this is reported as weak
COUNT_SEARCH_CHUNKS = 200  # chunks fetched per skill when counting; deduplicated by candidate
CHUNK_PREVIEW_CHARS = 400
DETAIL_MAX_CHARS = 6000  # a longer CV is cut down to the paragraphs about `focus`
JD_MIN_CHARS = 300  # longer input to extract_criteria is treated as a JD and kept for the whole-JD search


@dataclass
class ToolResult:
    content: dict
    update: dict = field(default_factory=dict)


def error(message: str) -> ToolResult:
    return ToolResult({"error": message})


def current_criteria(state: AgentState) -> SearchCriteria | None:
    raw = state.get("criteria")
    return SearchCriteria.model_validate(raw) if raw else None


class SkillFinderTools:
    def __init__(self, rag_service: RagService, candidate_repository: CandidateRepository,
                 extractor: RequirementExtractor, match_graph: MatchGraph,
                 max_attempts: int = MATCH_EXTRACTION_MAX_ATTEMPTS):
        self.rag_service = rag_service
        self.candidate_repository = candidate_repository
        self.extractor = extractor
        self.match_graph = match_graph
        self.max_attempts = max_attempts

    # --- 6.1 extract_criteria -----------------------------------------------------------------------------

    def extract_criteria(self, state: AgentState, text: str, role: str | None = None) -> ToolResult:
        """Existing extraction (with its one repair attempt), mapped to SearchCriteria and merged with the current
        criteria. A long text is taken as a JD and kept for the whole-JD search."""
        reason = ""
        try:
            for _ in range(self.max_attempts):
                try:
                    parsed = parse_requirements(self.extractor.extract(text, reason))
                    break
                except ValueError as problem:
                    reason = str(problem)
            else:
                return error(f"Could not extract requirements: {reason}. Try rephrasing, or use update_criteria.")
        except RECOVERABLE_LLM_ERRORS as problem:
            return error(f"The language model is unavailable ({type(problem).__name__}). Try again later.")
        is_jd = len(text) >= JD_MIN_CHARS
        new = criteria_rules.from_job_requirements(parsed, "jd" if is_jd else "conversation")
        if role:
            new = new.model_copy(update={"role": role})
        merged = criteria_rules.merge(current_criteria(state), new)
        update = {"criteria": merged.model_dump()}
        if is_jd:
            update["jd_text"] = text
        return ToolResult({"criteria": merged.model_dump(), "missing_info": criteria_rules.missing_info(merged)},
                          update)

    # --- 6.2 update_criteria ------------------------------------------------------------------------------

    def update_criteria(self, state: AgentState, add: list[dict] | None = None, remove: list[str] | None = None,
                        change_kind: dict[str, str] | None = None, role: str | None = None) -> ToolResult:
        criteria = current_criteria(state)
        if criteria is None:
            return error("No criteria exist yet: call extract_criteria first.")
        try:
            additions = [Requirement.model_validate(item) for item in add or []]
        except ValidationError as problem:
            return error(f"Invalid requirement to add: {problem.errors()[0]['msg']}")
        bad_kinds = [kind for kind in (change_kind or {}).values() if kind not in ("must", "nice")]
        if bad_kinds:
            return error(f"change_kind values must be 'must' or 'nice', got {bad_kinds}.")
        edited, unknown = criteria_rules.edit(criteria, additions, remove or [], change_kind or {})
        if role:
            edited = edited.model_copy(update={"role": role})
        # edited criteria no longer describe the original JD: the whole-JD search uses the criteria from now on
        content = {"criteria": edited.model_dump()}
        if unknown:
            content["not_found"] = unknown
        return ToolResult(content, {"criteria": edited.model_dump(), "jd_text": None})

    # --- 6.3 count_matches --------------------------------------------------------------------------------

    def count_matches(self, state: AgentState, skills: list[str]) -> ToolResult:
        """How many candidates have evidence of each skill, using the normal similarity threshold."""
        skills = [skill.strip() for skill in skills if skill.strip()]
        if not skills:
            return error("Give at least one skill.")
        counts = {}
        for skill in skills:
            chunks = self.rag_service.search_chunks(skill, None, COUNT_SEARCH_CHUNKS, use_reranker=False)
            counts[skill] = len({chunk["candidate_id"] for chunk in chunks if chunk.get("candidate_id")})
        return ToolResult({"counts": counts, "pool_size": len(self.candidate_repository.all_names())})

    # --- 6.4 search_candidates ----------------------------------------------------------------------------

    def search_candidates(self, state: AgentState, limit: int = SEARCH_LIMIT_MAX) -> ToolResult:
        """MatchGraph's prefilter, per-requirement search and shortlist, unchanged."""
        criteria = current_criteria(state)
        problem = criteria_rules.floor_problem(criteria)
        if problem:
            return error(problem)
        limit = max(1, min(limit, SEARCH_LIMIT_MAX))
        search_state = {"requirements": criteria_rules.to_match_requirements(criteria),
                        "min_total_years": criteria.min_total_years,
                        "jd_text": state.get("jd_text") or criteria_rules.query_text(criteria),
                        "shortlist_size": limit}
        search_state |= self.match_graph.prefilter(search_state)
        search_state |= self.match_graph.search(search_state)
        shortlist = self.match_graph.shortlist(search_state)["shortlist"]
        found_by = {r["skill"]: r.get("found_candidates", []) for r in search_state["requirements"]}
        candidates = [{"candidate_id": cid, "found_for": [skill for skill, ids in found_by.items() if cid in ids]}
                      for cid in shortlist]
        weak = [skill for skill, ids in found_by.items() if len(ids) < WEAK_COVERAGE]
        return ToolResult(
            {"found": len(candidates), "candidates_per_requirement": {s: len(ids) for s, ids in found_by.items()},
             "weak_coverage": weak},
            {"candidates": candidates, "ranking": None, "clarification_rounds": 0},
        )

    # --- 6.5 rank_candidates ------------------------------------------------------------------------------

    def rank_candidates(self, state: AgentState, top_k: int = RANK_TOP_K_MAX) -> ToolResult:
        """MatchGraph's judge and score, unchanged, on the first MATCH_SHORTLIST_SIZE candidates of the search.
        The score is shown on 0-100: the existing score (0-1) times 100."""
        candidates = state.get("candidates") or []
        if not candidates:
            return error("No candidates to rank: call search_candidates first.")
        criteria = current_criteria(state)
        top_k = max(1, min(top_k, RANK_TOP_K_MAX))
        requirements = criteria_rules.to_match_requirements(criteria)
        judge_requirements = [{key: r[key] for key in ("skill", "must_have", "min_years")} for r in requirements]
        shortlist = [c["candidate_id"] for c in candidates][:MATCH_SHORTLIST_SIZE]
        try:
            judgements = [self.match_graph.judge(JudgeTask(candidate_id=cid, requirements=judge_requirements))
                          ["judgements"][0] for cid in shortlist]
        except RECOVERABLE_LLM_ERRORS as problem:
            return error(f"The language model is unavailable ({type(problem).__name__}). Try again later.")
        ranked = self.match_graph.rank({"shortlist": shortlist, "judgements": judgements,
                                        "requirements": requirements, "top_n": top_k})["candidates"]
        ranking = {"requirements": requirements, "candidates": [presented(c) for c in ranked]}
        return ToolResult(ranking, {"ranking": ranking})

    # --- 6.6 get_candidate_detail -------------------------------------------------------------------------

    def get_candidate_detail(self, state: AgentState, candidate_id: int, focus: str | None = None) -> ToolResult:
        allowed = {c["candidate_id"] for c in state.get("candidates") or []}
        allowed |= {c["candidate_id"] for c in (state.get("ranking") or {}).get("candidates", [])}
        if candidate_id not in allowed:
            return error(f"Candidate {candidate_id} is not in the current results. Search first, or pick one of: "
                         f"{sorted(allowed)}.")
        cv_text = self.candidate_repository.cv_text(candidate_id, self.rag_service.embedder.model_name)
        return ToolResult({"candidate_id": candidate_id, "cv": focused(cv_text, focus)})

    # --- extra: search_cv_chunks (general questions over the CVs, replaces the old /chat answer path) ---------

    def search_cv_chunks(self, state: AgentState, query: str, candidate_ids: list[int] | None = None,
                         top_k: int = 10) -> ToolResult:
        if not query.strip():
            return error("Give a search query.")
        chunks = self.rag_service.search_chunks(query, None, max(1, min(top_k, 20)), candidate_ids=candidate_ids)
        return ToolResult({"chunks": [{"candidate_id": c.get("candidate_id"),
                                       "text": c["content"][:CHUNK_PREVIEW_CHARS]} for c in chunks]})


def presented(candidate: dict) -> dict:
    """One ranked candidate for the LLM and the API: score 0-100, verdicts with quotes, strengths and gaps with
    must-haves first."""
    verdicts = candidate["verdicts"]
    must_first = sorted(verdicts, key=lambda v: not v["must_have"])  # sorted is stable: requirement order kept
    return {
        "candidate_id": candidate["candidate_id"],
        "score": round(candidate["score"] * 100),
        "verdicts": verdicts,
        "strengths": [v["requirement"] for v in must_first if v["verdict"] == MET],
        "gaps": [v["requirement"] for v in must_first if v["verdict"] != MET],
        "missing_must_haves": [v["requirement"] for v in verdicts if v["must_have"] and v["verdict"] == NOT_FOUND],
    }


def focused(cv_text: str, focus: str | None) -> str:
    """The whole CV, or for a long CV with a focus, the paragraphs that mention a focus word."""
    if len(cv_text) <= DETAIL_MAX_CHARS or not focus:
        return cv_text[:DETAIL_MAX_CHARS * 2]
    words = {word for word in re.findall(r"\w+", focus.lower()) if len(word) > 2}
    paragraphs = [p for p in re.split(r"\n\s*\n", cv_text) if words & set(re.findall(r"\w+", p.lower()))]
    return "\n\n".join(paragraphs)[:DETAIL_MAX_CHARS] or cv_text[:DETAIL_MAX_CHARS]
