import re


def normalize_text(text: str) -> str:
    """Lower-case and collapse whitespace, so evidence still matches after PDF line breaks or chunk joins."""
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_evidence(chunk_content: str, evidence: str) -> bool:
    return normalize_text(evidence) in normalize_text(chunk_content)
