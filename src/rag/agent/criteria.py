"""Pure functions over SearchCriteria: convert from the existing extraction, merge, edit, and check the floor.
No LLM and no database here, so every rule is easy to test."""
from rag.agent.state import MUST, NICE, Requirement, SearchCriteria
from rag.matching.requirements import JobRequirements


def from_job_requirements(parsed: JobRequirements, source: str) -> SearchCriteria:
    """The existing extractor's output (skill / must_have) in the agent's shape (name / kind)."""
    return SearchCriteria(
        requirements=[Requirement(name=r.skill, synonyms=r.synonyms, kind=MUST if r.must_have else NICE,
                                  min_years=r.min_years) for r in parsed.requirements],
        min_total_years=parsed.min_total_years, source=source,
    )


def to_match_requirements(criteria: SearchCriteria) -> list[dict]:
    """The agent's criteria in the shape MatchGraph and the judge consume, so scoring stays unchanged."""
    return [{"skill": r.name, "must_have": r.kind == MUST, "min_years": r.min_years, "synonyms": r.synonyms}
            for r in criteria.requirements]


def merge(current: SearchCriteria | None, new: SearchCriteria) -> SearchCriteria:
    """New information adds to or overrides the current criteria. A requirement with the same name (any case)
    is replaced; others are kept. Role and years are replaced only when the new criteria set them."""
    if current is None:
        return new
    new_names = {r.name.lower() for r in new.requirements}
    kept = [r for r in current.requirements if r.name.lower() not in new_names]
    return SearchCriteria(
        role=new.role or current.role,
        requirements=kept + new.requirements,
        min_total_years=new.min_total_years if new.min_total_years is not None else current.min_total_years,
        other_constraints=list(dict.fromkeys(current.other_constraints + new.other_constraints)),
        assumptions=list(dict.fromkeys(current.assumptions + new.assumptions)),
        source=current.source if current.source == new.source else "mixed",
    )


def edit(criteria: SearchCriteria, add: list[Requirement], remove: list[str],
         change_kind: dict[str, str]) -> tuple[SearchCriteria, list[str]]:
    """Deterministic edit. Returns the new criteria and the names that were not found (reported, not fatal)."""
    by_name = {r.name.lower(): r for r in criteria.requirements}
    unknown = []
    for name in remove:
        if by_name.pop(name.strip().lower(), None) is None:
            unknown.append(name)
    for name, kind in change_kind.items():
        requirement = by_name.get(name.strip().lower())
        if requirement is None:
            unknown.append(name)
        else:
            by_name[name.strip().lower()] = requirement.model_copy(update={"kind": kind})
    for requirement in add:
        by_name[requirement.name.lower()] = requirement
    return criteria.model_copy(update={"requirements": list(by_name.values())}), unknown


def floor_problem(criteria: SearchCriteria | None) -> str | None:
    """D1: the minimum to search is one must-have skill, or a role plus at least one skill. None when met."""
    if criteria is None:
        return "No criteria yet: call extract_criteria first."
    if any(r.kind == MUST for r in criteria.requirements):
        return None
    if criteria.role and criteria.requirements:
        return None
    if not criteria.requirements:
        return "No skills yet: at least one concrete skill is needed."
    return "No must-have skill and no role: name a must-have skill or the role."


def missing_info(criteria: SearchCriteria) -> list[str]:
    """What would materially improve the search, for the agent to decide whether to ask."""
    missing = []
    if not criteria.role:
        missing.append("role")
    if not any(r.kind == MUST for r in criteria.requirements):
        missing.append("which skills are must-haves")
    if len(criteria.requirements) < 2:
        missing.append("more concrete skills")
    return missing


def query_text(criteria: SearchCriteria) -> str:
    """A search text for the whole-candidate search when there is no JD (criteria built in conversation)."""
    parts = [criteria.role or ""] + [" ".join([r.name, *r.synonyms]) for r in criteria.requirements]
    return ". ".join(part for part in parts if part)
