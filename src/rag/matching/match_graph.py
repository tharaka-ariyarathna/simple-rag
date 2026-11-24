import logging
import operator
import time
from typing import Annotated, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from rag.constants import (
    MATCH_CHUNKS_PER_REQUIREMENT, MATCH_EXTRACTION_MAX_ATTEMPTS, MATCH_JUDGE_CONCURRENCY, MATCH_SHORTLIST_SIZE,
    MATCH_WHOLE_JD_CHUNKS, MATCHING_MIN_SIMILARITY, RRF_K,
)
from rag.matching.judge import NOT_FOUND, CandidateJudge, score
from rag.matching.requirements import RequirementExtractionError, RequirementExtractor, parse_requirements
from rag.repository.candidate_repository import CandidateRepository
from rag.service.rag_service import RagService

logger = logging.getLogger(__name__)


class MatchState(TypedDict, total=False):
    jd_text: str
    top_n: int
    shortlist_size: int
    reply: str  # the extraction LLM's raw reply
    rejection_reason: str
    attempts: int
    requirements: list[dict]
    min_total_years: float | None
    allowed_ids: list[int] | None  # None = no years filter
    shortlist: list[int]
    whole_jd_order: list[int]  # every candidate the whole-JD search found, best first (eval and tie-breaks)
    judgements: Annotated[list[dict], operator.add]  # one per judged candidate, collected from the parallel judges
    candidates: list[dict]  # the final ranking


class JudgeTask(TypedDict):
    candidate_id: int
    requirements: list[dict]


class MatchGraph:
    """JD in, ranked candidates with evidence out, as a LangGraph workflow:

        START -> extract -> validate --(repair once)--> extract
                            validate -> prefilter -> search -> shortlist --Send x N--> judge --> rank -> END

    shortlist fans out one judge run per candidate (LangGraph's Send: map), the judgements reducer collects them,
    rank runs once they are all in (reduce). No checkpointer: a match is one-shot; follow-ups live in /chat."""

    def __init__(self, rag_service: RagService, candidate_repository: CandidateRepository,
                 extractor: RequirementExtractor, judge: CandidateJudge, max_attempts: int = MATCH_EXTRACTION_MAX_ATTEMPTS,
                 judge_concurrency: int = MATCH_JUDGE_CONCURRENCY):
        self.rag_service = rag_service
        self.candidate_repository = candidate_repository
        self.extractor = extractor
        self.judge_llm = judge
        self.max_attempts = max_attempts
        self.judge_concurrency = judge_concurrency
        builder = StateGraph(MatchState)
        builder.add_node("extract", self.extract)
        builder.add_node("validate", self.validate)
        builder.add_node("prefilter", self.prefilter)
        builder.add_node("search", self.search)
        builder.add_node("shortlist", self.shortlist)
        builder.add_node("judge", self.judge)
        builder.add_node("rank", self.rank)
        builder.add_edge(START, "extract")
        builder.add_edge("extract", "validate")
        builder.add_conditional_edges("validate", self.route_after_validate, ["extract", "prefilter"])
        builder.add_edge("prefilter", "search")
        builder.add_edge("search", "shortlist")
        builder.add_conditional_edges("shortlist", self.fan_out, ["judge", "rank"])
        builder.add_edge("judge", "rank")
        builder.add_edge("rank", END)
        self.graph = builder.compile()

    def match(self, jd_text: str, top_n: int, shortlist_size: int = MATCH_SHORTLIST_SIZE) -> dict:
        """The match result: requirements, min_total_years, candidates (ranked, with verdicts and gaps), plus
        shortlist and whole_jd_order for evaluation. Raises RequirementExtractionError or LLM errors."""
        started_at = time.perf_counter()
        state = self.graph.invoke(
            {"jd_text": jd_text, "top_n": top_n, "shortlist_size": max(shortlist_size, top_n), "attempts": 0,
             "rejection_reason": "", "judgements": []},
            {"max_concurrency": self.judge_concurrency},
        )
        logger.info("Matched %d candidates in %.1fs", len(state["candidates"]), time.perf_counter() - started_at)
        return {key: state[key] for key in
                ("requirements", "min_total_years", "candidates", "shortlist", "whole_jd_order")}

    # --- extract and validate (with one repair attempt) -------------------------------------------------------

    def extract(self, state: MatchState) -> dict:
        return {"reply": self.extractor.extract(state["jd_text"], state["rejection_reason"]),
                "attempts": state["attempts"] + 1}

    def validate(self, state: MatchState) -> dict:
        try:
            parsed = parse_requirements(state["reply"])
        except ValueError as error:
            logger.warning("Requirements rejected (attempt %d of %d): %s", state["attempts"], self.max_attempts, error)
            if state["attempts"] >= self.max_attempts:
                raise RequirementExtractionError(str(error)) from error
            return {"rejection_reason": str(error), "requirements": []}
        return {"requirements": [requirement.model_dump() for requirement in parsed.requirements],
                "min_total_years": parsed.min_total_years, "rejection_reason": ""}

    def route_after_validate(self, state: MatchState) -> str:
        return "prefilter" if state["requirements"] else "extract"

    # --- find candidates --------------------------------------------------------------------------------------

    def prefilter(self, state: MatchState) -> dict:
        """Years of experience is a SQL filter, not an LLM judgement: it was computed once at ingestion."""
        if state.get("min_total_years") is None:
            return {"allowed_ids": None}
        return {"allowed_ids": self.candidate_repository.ids_with_min_years(state["min_total_years"])}

    def search(self, state: MatchState) -> dict:
        """One hybrid search per requirement (skill + synonyms), plus one whole-JD vector search for recall. Each
        requirement records which candidates it found: that is the coverage the shortlist ranks by."""
        allowed = state["allowed_ids"]
        if allowed == []:
            return {"requirements": state["requirements"], "whole_jd_order": []}
        requirements = []
        for requirement in state["requirements"]:
            query = " ".join([requirement["skill"], *requirement["synonyms"]])
            chunks = self.rag_service.search_chunks(query, None, MATCH_CHUNKS_PER_REQUIREMENT, candidate_ids=allowed)
            found = list(dict.fromkeys(chunk["candidate_id"] for chunk in chunks if chunk.get("candidate_id")))
            requirements.append({**requirement, "found_candidates": found})
        whole_jd = self.rag_service.search_chunks(
            state["jd_text"], None, MATCH_WHOLE_JD_CHUNKS, min_similarity=MATCHING_MIN_SIMILARITY,
            use_keywords=False, use_reranker=False, candidate_ids=allowed)
        whole_jd_order = list(dict.fromkeys(chunk["candidate_id"] for chunk in whole_jd if chunk.get("candidate_id")))
        return {"requirements": requirements, "whole_jd_order": whole_jd_order}

    def shortlist(self, state: MatchState) -> dict:
        """Two rankings fused with reciprocal rank fusion (as in hybrid search): requirement coverage (must-haves
        count double) and the whole-JD search. Coverage alone dropped relevant CVs the whole-JD search ranks well:
        on 5 JDs only 71% of relevant CVs reached the judge."""
        coverage: dict[int, float] = {candidate_id: 0.0 for candidate_id in state["whole_jd_order"]}
        for requirement in state["requirements"]:
            for candidate_id in requirement.get("found_candidates", []):
                coverage[candidate_id] = coverage.get(candidate_id, 0.0) + (2.0 if requirement["must_have"] else 1.0)
        coverage_order = sorted(coverage, key=lambda cid: -coverage[cid])
        fused: dict[int, float] = {}
        for order in (coverage_order, state["whole_jd_order"]):
            for rank, candidate_id in enumerate(order, start=1):
                fused[candidate_id] = fused.get(candidate_id, 0.0) + 1.0 / (RRF_K + rank)
        ordered = sorted(fused, key=lambda cid: -fused[cid])
        return {"shortlist": ordered[:state["shortlist_size"]]}

    def fan_out(self, state: MatchState) -> list[Send] | str:
        requirements = [{key: r[key] for key in ("skill", "must_have", "min_years")} for r in state["requirements"]]
        if not state["shortlist"]:
            return "rank"
        return [Send("judge", JudgeTask(candidate_id=cid, requirements=requirements)) for cid in state["shortlist"]]

    # --- judge each candidate (in parallel) and rank -----------------------------------------------------------

    def judge(self, task: JudgeTask) -> dict:
        cv_text = self.candidate_repository.cv_text(task["candidate_id"], self.rag_service.embedder.model_name)
        verdicts = self.judge_llm.judge(cv_text, task["requirements"])
        return {"judgements": [{"candidate_id": task["candidate_id"], "verdicts": verdicts}]}

    def rank(self, state: MatchState) -> dict:
        shortlist_rank = {cid: rank for rank, cid in enumerate(state["shortlist"])}
        candidates = [
            {"candidate_id": judgement["candidate_id"], "score": score(judgement["verdicts"]),
             "verdicts": judgement["verdicts"],
             "gaps": [v["requirement"] for v in judgement["verdicts"] if v["verdict"] != "met"],
             "missing_must_haves": [v["requirement"] for v in judgement["verdicts"]
                                    if v["must_have"] and v["verdict"] == NOT_FOUND]}
            for judgement in state["judgements"]
        ]
        candidates.sort(key=lambda c: (-c["score"], shortlist_rank.get(c["candidate_id"], len(shortlist_rank))))
        requirements = [{key: r[key] for key in ("skill", "must_have", "min_years", "synonyms")}
                        for r in state["requirements"]]
        return {"candidates": candidates[:state["top_n"]], "requirements": requirements}
