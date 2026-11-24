import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator, model_validator
from rag.constants import SKILL_ALIASES
from rag.evaluation.text_matching import contains_evidence
from rag.llm.base import LlmClient
from rag.llm.call_context import llm_call_context
from rag.prompt_constants import PROFILE_EXTRACTION_SYSTEM_PROMPT, PROFILE_EXTRACTION_USER_TEMPLATE

logger = logging.getLogger(__name__)

PRESENT_WORDS = {"present", "current", "now", "today", "ongoing"}
YEAR_MONTH = re.compile(r"^(\d{4})(?:[-/.](\d{1,2}))?$")


class ProfileExtractionError(Exception):
    """The local LLM answered, but not with the expected JSON. The CV is reported as failed and retried next run."""


# What the LLM must return. Sent to Ollama as a JSON schema, so the reply is constrained to this shape.
class ExtractedJob(BaseModel):
    title: str
    company: str
    start: str
    end: str


class ExtractedProfile(BaseModel):
    name: str
    skills: list[str]
    jobs: list[ExtractedJob]


def parse_month(value: str) -> date | None:
    """ "2021" -> 2021-01-01, "2021-03" -> 2021-03-01, "Present" -> None (still in the job)."""
    text = str(value).strip().lower()
    if text in PRESENT_WORDS:
        return None
    match = YEAR_MONTH.match(text)
    if not match:
        raise ValueError(f"not a year or year-month: {value!r}")
    month = int(match.group(2) or 1)
    if not 1 <= month <= 12:
        raise ValueError(f"no such month: {value!r}")
    return date(int(match.group(1)), month, 1)


class Job(BaseModel):
    """One validated job. Validation context: {"cv_text": ..., "today": date}."""

    title: str
    company: str
    start: date
    end: date | None  # None = Present

    @field_validator("start", "end", mode="before")
    @classmethod
    def parse_dates(cls, value):
        return value if isinstance(value, date) or value is None else parse_month(value)

    @model_validator(mode="after")
    def check_against_the_cv(self, info: ValidationInfo):
        today = info.context["today"]
        if self.start > today or (self.end is not None and not self.start <= self.end <= today):
            raise ValueError(f"impossible dates: {self.start} to {self.end or 'Present'}")
        if not contains_evidence(info.context["cv_text"], self.company):  # the LLM must not invent employers
            raise ValueError(f"company {self.company!r} is not in the CV")
        return self


@dataclass
class CandidateProfile:
    name: str | None  # None when the extracted name is not in the CV: better no redaction by name than a wrong one
    skills: list[str]
    jobs: list[Job]
    total_years: float
    model: str
    rejected_jobs: list[str] = field(default_factory=list)  # why each dropped job failed validation


def name_in_cv(extracted_name: str, cv_text: str) -> str | None:
    """The extracted name if the CV contains it; else its longest run of 2+ consecutive words that the CV contains.
    Small models sometimes glue a neighbouring word on, e.g. the city from a two-column layout ("Austin Pedro Patel").
    None when nothing of 2+ words matches: better no name (the CV is then not indexed) than a wrong one."""
    words = extracted_name.split()
    for length in range(len(words), 1, -1):
        for start in range(len(words) - length + 1):
            candidate = " ".join(words[start:start + length])
            if contains_evidence(cv_text, candidate):
                return candidate
    return None


def canonical_skill(skill: str) -> str:
    return SKILL_ALIASES.get(skill.strip().lower(), skill.strip())


def total_years(jobs: list[Job], today: date) -> float:
    """Years of experience, with overlapping jobs counted once (two jobs held in parallel are not double time)."""
    periods = sorted((job.start, job.end or today) for job in jobs)
    months, current_start, current_end = 0, None, None
    for start, end in periods:
        if current_end is None or start > current_end:
            if current_end is not None:
                months += months_between(current_start, current_end)
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        months += months_between(current_start, current_end)
    return round(months / 12, 1)


def months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + end.month - start.month


class ProfileExtractor:
    """Reads one raw CV with a LOCAL model and returns a validated profile. Knows nothing about the database."""

    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client

    def extract(self, cv_text: str, today: date | None = None) -> CandidateProfile:
        today = today or date.today()
        with llm_call_context("profile_extraction"):
            result = self.llm_client.generate(PROFILE_EXTRACTION_SYSTEM_PROMPT,
                                              PROFILE_EXTRACTION_USER_TEMPLATE.format(cv_text=cv_text.strip()))
        extracted = self.parse(result.text)
        jobs, rejected = [], []
        for extracted_job in extracted.jobs:  # one at a time: a bad job is dropped, the profile is kept
            try:
                jobs.append(Job.model_validate(extracted_job.model_dump(),
                                               context={"cv_text": cv_text, "today": today}))
            except ValidationError as error:
                rejected.append(f"{extracted_job.title} at {extracted_job.company}: {error.errors()[0]['msg']}")
        if rejected:
            logger.info("Dropped %d job(s): %s", len(rejected), "; ".join(rejected))
        skills = list(dict.fromkeys(canonical_skill(skill) for skill in extracted.skills
                                    if skill.strip() and contains_evidence(cv_text, skill)))
        return CandidateProfile(
            name=name_in_cv(extracted.name, cv_text),
            skills=skills, jobs=jobs, total_years=total_years(jobs, today), model=result.model, rejected_jobs=rejected,
        )

    @staticmethod
    def parse(reply_text: str) -> ExtractedProfile:
        first_brace, last_brace = reply_text.find("{"), reply_text.rfind("}")
        try:
            return ExtractedProfile.model_validate(json.loads(reply_text[first_brace:last_brace + 1]))
        except (ValueError, ValidationError) as error:  # json errors are ValueErrors too
            raise ProfileExtractionError(f"Reply is not the expected JSON: {reply_text[:200]!r}") from error
