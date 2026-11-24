import re
from rag.constants import DROPPED_SECTIONS
from rag.indexing.section_chunker import Section

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github|gitlab)\.com/\S*", re.IGNORECASE)
# 9+ digits with phone punctuation between them; a year range like "2019 - 2021" has only 8
PHONE = re.compile(r"(?<![\w])\+?\(?\d(?:[\s().-]*\d){8,}(?![\w])")
NAME_PLACEHOLDER = "the candidate"
MIN_NAME_PART_LENGTH = 3  # shorter parts ("Li", "Al") would also match inside ordinary text too often


def name_patterns(name: str | None) -> list[re.Pattern]:
    """The full name first, then each part alone ("Ines", "Brooks"), as whole words, ignoring case."""
    if not name or not name.strip():
        return []
    parts = [part for part in re.split(r"\s+", name.strip()) if len(part) >= MIN_NAME_PART_LENGTH]
    return [re.compile(rf"\b{re.escape(text)}\b", re.IGNORECASE) for text in [name.strip(), *parts]]


def redact_text(text: str, name: str | None) -> str:
    """Contacts become [EMAIL] / [URL] / [PHONE] and the candidate's name becomes "the candidate"."""
    text = EMAIL.sub("[EMAIL]", text)  # before URL: an email domain is not a URL
    text = URL.sub("[URL]", text)
    text = PHONE.sub("[PHONE]", text)
    for pattern in name_patterns(name):
        text = pattern.sub(NAME_PLACEHOLDER, text)
    return text


def redact_sections(sections: list[Section], name: str | None) -> list[Section]:
    """Contact and referee sections are dropped whole (referees are other people's personal data); the rest is
    redacted. What comes out is all that is ever chunked, embedded or sent to a cloud LLM."""
    return [Section(section.name, redact_text(section.text, name))
            for section in sections if section.name not in DROPPED_SECTIONS]


def find_leaks(text: str, name: str | None) -> list[str]:
    """Personal data still in a text that was meant to be redacted (empty = clean). Used by tests and the pipeline."""
    contacts = EMAIL.findall(text) + URL.findall(text) + PHONE.findall(text)
    without_contacts = PHONE.sub(" ", URL.sub(" ", EMAIL.sub(" ", text)))  # "ines@..." is one leak, not two
    return contacts + [match.group() for pattern in name_patterns(name) for match in pattern.finditer(without_contacts)]
