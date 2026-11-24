"""Names never reach a cloud LLM: chunks are redacted, the LLM speaks of "candidate 17", and only the API's final
response turns ids back into names. This module does both directions: names in a question -> candidate ids
(NameResolver), and candidate ids in an answer -> names (with_names)."""
import re
from dataclasses import dataclass, field
from typing import Protocol

CANDIDATE_REFERENCE = re.compile(r"\bcandidate\s*#?\s*(\d+)\b", re.IGNORECASE)
WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")
MIN_NAME_PART_LENGTH = 3


class CandidateDirectory(Protocol):
    """What name handling needs from CandidateRepository."""

    def all_names(self) -> list[dict]: ...  # [{id, name, total_years, skills}]

    def names_by_id(self, candidate_ids: list[int]) -> dict[int, str | None]: ...


def candidate_ids_in(text: str) -> list[int]:
    """Ids from "candidate 17" / "Candidate #17" references, in order of first appearance."""
    return list(dict.fromkeys(int(match.group(1)) for match in CANDIDATE_REFERENCE.finditer(text)))


def with_names(text: str, names: dict[int, str | None]) -> str:
    """ "candidate 17" -> "Ines Brooks (candidate 17)" for every id with a known name; others stay as they are."""
    def replace(match: re.Match) -> str:
        name = names.get(int(match.group(1)))
        return f"{name} ({match.group(0)})" if name else match.group(0)
    return CANDIDATE_REFERENCE.sub(replace, text)


@dataclass
class Resolution:
    question: str  # names replaced by "candidate N"
    candidate_ids: list[int] = field(default_factory=list)  # everyone the question refers to
    clarification: str | None = None  # set when a name fits several candidates: ask instead of guessing


class NameResolver:
    """Finds candidate names in a question and swaps them for ids, so the search can be filtered by candidate and
    the name never goes further. Full names are matched first, then a single first or last name written with a
    capital letter. A name shared by several candidates is narrowed by the candidates the conversation already
    mentioned (or the client sent); if that still leaves several, the user is asked which one."""

    def __init__(self, directory: CandidateDirectory):
        self.directory = directory

    def resolve(self, question: str, known_ids: list[int] | None = None) -> Resolution:
        known = set(known_ids or [])
        people = [row for row in self.directory.all_names() if row["name"]]
        ids = candidate_ids_in(question)
        for mention, matches in self.mentions(question, people):
            if len(matches) > 1 and known & {row["id"] for row in matches}:
                matches = [row for row in matches if row["id"] in known]
            if len(matches) > 1:
                return Resolution(question, ids, clarification=self.clarification(matches))
            candidate_id = matches[0]["id"]
            question = re.sub(rf"\b{re.escape(mention)}\b", f"candidate {candidate_id}", question, flags=re.IGNORECASE)
            ids.append(candidate_id)
        return Resolution(question, list(dict.fromkeys(ids)))

    @staticmethod
    def mentions(question: str, people: list[dict]) -> list[tuple[str, list[dict]]]:
        """(text as found in the question, candidates it could be), full names before single names."""
        found, taken = [], question
        by_full_name: dict[str, list[dict]] = {}
        for row in people:
            by_full_name.setdefault(row["name"].lower(), []).append(row)
        for full_name, rows in sorted(by_full_name.items(), key=lambda item: -len(item[0])):
            match = re.search(rf"\b{re.escape(full_name)}\b", taken, re.IGNORECASE)
            if match:
                found.append((match.group(0), rows))
                taken = taken.replace(match.group(0), " ")  # "Ines Brooks" must not match again as "Ines"
        for word in dict.fromkeys(WORD.findall(taken)):
            if len(word) < MIN_NAME_PART_LENGTH or not word[0].isupper():
                continue
            rows = [row for row in people if word.lower() in row["name"].lower().split()]
            if rows:
                found.append((word, rows))
        return found

    @staticmethod
    def clarification(matches: list[dict]) -> str:
        options = "; ".join(
            f"candidate {row['id']} ({float(row['total_years']):g} years"
            + (f", {', '.join(row['skills'][:3])}" if row.get("skills") else "") + ")"
            for row in matches
        )
        # the name itself is left out: this text is kept in the conversation history, which a cloud LLM reads
        return (f"More than one candidate has that name: {options}. "
                "Which one do you mean? Ask again with the candidate number.")
