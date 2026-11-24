import json
import logging
from rag.constants import MUST_HAVE_CAP, MUST_HAVE_WEIGHT, NICE_TO_HAVE_WEIGHT, VERDICT_POINTS
from rag.evaluation.text_matching import contains_evidence
from rag.llm.base import LlmClient
from rag.llm.call_context import llm_call_context
from rag.prompt_constants import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE

logger = logging.getLogger(__name__)

MET, PARTIAL, NOT_FOUND = "met", "partial", "not_found"


class CandidateJudge:
    """One LLM call per candidate: all requirements and the candidate's redacted CV in, one verdict per requirement
    out. Verdicts are checked in code: a quote that is not in the CV turns met/partial into not_found."""

    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client

    def judge(self, cv_text: str, requirements: list[dict]) -> list[dict]:
        listed = "\n".join(
            f"- {r['skill']}" + (f" ({r['min_years']:g}+ years)" if r.get("min_years") else "")
            + (" [must have]" if r["must_have"] else " [nice to have]")
            for r in requirements
        )
        with llm_call_context("judge"):
            reply = self.llm_client.generate(JUDGE_SYSTEM_PROMPT,
                                             JUDGE_USER_TEMPLATE.format(requirements=listed, cv_text=cv_text)).text
        return checked_verdicts(reply, requirements, cv_text)


def checked_verdicts(reply: str, requirements: list[dict], cv_text: str) -> list[dict]:
    """One verdict per requirement, in requirement order. Missing or unreadable verdicts are not_found, so a bad reply
    can only make a candidate look weaker, never stronger."""
    by_requirement = {}
    first_brace, last_brace = reply.find("{"), reply.rfind("}")
    try:
        raw_verdicts = json.loads(reply[first_brace:last_brace + 1])["verdicts"]
    except (ValueError, KeyError, TypeError):
        logger.warning("Judge reply is not the expected JSON: %r", reply[:200])
        raw_verdicts = []
    for raw in raw_verdicts if isinstance(raw_verdicts, list) else []:
        if isinstance(raw, dict) and isinstance(raw.get("requirement"), str):
            by_requirement.setdefault(raw["requirement"].strip().lower(), raw)
    verdicts = []
    for requirement in requirements:
        raw = by_requirement.get(requirement["skill"].lower(), {})
        verdict = raw.get("verdict") if raw.get("verdict") in VERDICT_POINTS else NOT_FOUND
        evidence = raw.get("evidence") if isinstance(raw.get("evidence"), str) else ""
        note = ""
        if verdict != NOT_FOUND and not (evidence.strip() and contains_evidence(cv_text, evidence)):
            verdict, note = NOT_FOUND, "the quoted evidence is not in the CV"
        verdicts.append({"requirement": requirement["skill"], "must_have": requirement["must_have"],
                         "verdict": verdict, "evidence": evidence.strip() if verdict != NOT_FOUND else "",
                         **({"note": note} if note else {})})
    return verdicts


def score(verdicts: list[dict]) -> float:
    """Weighted share of requirements met (must-haves count double, partial counts half). Missing any must-have caps
    the score, so one strong nice-to-have can never lift a candidate over someone who meets the must-haves."""
    weights = [MUST_HAVE_WEIGHT if v["must_have"] else NICE_TO_HAVE_WEIGHT for v in verdicts]
    if not weights:
        return 0.0
    value = sum(weight * VERDICT_POINTS[v["verdict"]] for weight, v in zip(weights, verdicts)) / sum(weights)
    if any(v["must_have"] and v["verdict"] == NOT_FOUND for v in verdicts):
        value = min(value, MUST_HAVE_CAP)
    return round(value, 3)
