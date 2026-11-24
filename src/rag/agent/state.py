"""What the Skill Finder agent remembers across turns, and the shape of its search criteria.

Criteria and results are kept in the state as plain dicts (model_dump), not pydantic objects: the Postgres
checkpointer stores dicts without needing to know our classes, and the tools validate them on the way in."""
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

MUST, NICE = "must", "nice"


class Requirement(BaseModel):
    name: str = Field(min_length=1, max_length=80)  # e.g. "AWS ECS deployments"
    synonyms: list[str] = Field(default_factory=list)  # e.g. ["Fargate", "EKS"]
    kind: Literal["must", "nice"]
    min_years: float | None = Field(default=None, ge=0, le=50)  # kept from the existing extraction

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        return value.strip()


class SearchCriteria(BaseModel):
    role: str | None = None
    requirements: list[Requirement] = Field(default_factory=list)
    min_total_years: float | None = Field(default=None, ge=0, le=50)  # SQL prefilter, as in /match
    other_constraints: list[str] = Field(default_factory=list)  # free text, informational only
    assumptions: list[str] = Field(default_factory=list)  # what the agent assumed without asking
    source: Literal["jd", "conversation", "mixed"] = "conversation"


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    criteria: dict | None  # SearchCriteria.model_dump(): the current working criteria
    jd_text: str | None  # the full JD when one was given: the whole-JD search uses it, as /match does
    candidates: list[dict]  # last search: shortlist order, ids + which requirements found each one
    ranking: dict | None  # last rank_candidates output
    clarification_rounds: int  # reset whenever a search succeeds
    mentioned_candidates: list[int]  # ids the conversation referred to, for name resolution
