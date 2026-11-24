import json
from pydantic import BaseModel, Field, ValidationError, field_validator
from rag.llm.base import LlmClient
from rag.llm.call_context import llm_call_context
from rag.prompt_constants import (
    REQUIREMENT_EXTRACTION_SYSTEM_PROMPT, REQUIREMENT_EXTRACTION_USER_TEMPLATE, REQUIREMENT_REPAIR_TEMPLATE,
)

MAX_REQUIREMENTS = 15


class RequirementExtractionError(Exception):
    """The LLM's reply could not be turned into requirements, even after the repair attempt."""


class Requirement(BaseModel):
    skill: str = Field(min_length=1, max_length=80)
    must_have: bool
    min_years: float | None = Field(default=None, ge=0, le=50)
    synonyms: list[str] = Field(default_factory=list)

    @field_validator("skill")
    @classmethod
    def strip_skill(cls, value: str) -> str:
        return value.strip()

    @field_validator("synonyms")
    @classmethod
    def clean_synonyms(cls, value: list[str]) -> list[str]:
        return [synonym.strip() for synonym in value if synonym.strip()][:5]


class JobRequirements(BaseModel):
    requirements: list[Requirement] = Field(min_length=1, max_length=MAX_REQUIREMENTS)
    min_total_years: float | None = Field(default=None, ge=0, le=50)


def parse_requirements(reply: str) -> JobRequirements:
    """Raises ValueError (with a reason the repair prompt can quote) when the reply is not usable."""
    first_brace, last_brace = reply.find("{"), reply.rfind("}")
    if first_brace == -1 or last_brace <= first_brace:
        raise ValueError("the reply contained no JSON object")
    try:
        data = json.loads(reply[first_brace:last_brace + 1])
    except ValueError as error:
        raise ValueError(f"the JSON was invalid ({error.args[0]})") from error
    if isinstance(data, dict) and isinstance(data.get("requirements"), list):
        data["requirements"] = data["requirements"][:MAX_REQUIREMENTS]  # too many is trimmed, not rejected
    try:
        return JobRequirements.model_validate(data)
    except ValidationError as error:
        first = error.errors()[0]
        raise ValueError(f"{'.'.join(map(str, first['loc']))}: {first['msg']}") from error


class RequirementExtractor:
    """One LLM call: JD text in, raw reply out. A JD holds no personal data, so the cloud answer chain is used."""

    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client

    def extract(self, jd_text: str, rejection_reason: str = "") -> str:
        user_prompt = REQUIREMENT_EXTRACTION_USER_TEMPLATE.format(jd_text=jd_text.strip())
        if rejection_reason:
            user_prompt += REQUIREMENT_REPAIR_TEMPLATE.format(reason=rejection_reason)
        with llm_call_context("requirement_extraction"):
            return self.llm_client.generate(REQUIREMENT_EXTRACTION_SYSTEM_PROMPT, user_prompt).text
