from dataclasses import dataclass
from rag.constants import HEADER_SECTION, SECTION_HEADINGS
from rag.indexing.chunker import TokenChunker


@dataclass
class Section:
    name: str  # summary / experience / ... (see SECTION_HEADINGS), "header" before the first heading
    text: str


@dataclass
class SectionChunk:
    section: str
    content: str


def heading_of(line: str) -> str | None:
    """The section a line starts, or None. Only an exact known heading counts, so a short line of body text
    ("running in parallel") is never mistaken for one."""
    return SECTION_HEADINGS.get(line.strip().rstrip(":").strip().lower())


def split_sections(text: str) -> list[Section]:
    """The CV cut at its headings, in order. Heading lines themselves are not kept; empty sections are dropped."""
    sections, name, lines = [], HEADER_SECTION, []
    for line in text.splitlines():
        heading = heading_of(line)
        if heading is None:
            lines.append(line)
            continue
        sections.append(Section(name, "\n".join(lines).strip()))
        name, lines = heading, []
    sections.append(Section(name, "\n".join(lines).strip()))
    return [section for section in sections if section.text]


class SectionChunker:
    """Section first, then size: each section is chunked on its own (LangChain's splitter, via TokenChunker),
    so no chunk mixes two sections and every chunk knows which section it came from."""

    def __init__(self, token_chunker: TokenChunker):
        self.token_chunker = token_chunker

    def chunk_sections(self, sections: list[Section]) -> list[SectionChunk]:
        """Each chunk starts with its section's name ("Skills"), so a search for "skills" or "experience" can find it."""
        return [SectionChunk(section.name, self.titled(section.name, content))
                for section in sections for content in self.token_chunker.chunk_text(section.text)]

    @staticmethod
    def titled(section_name: str, content: str) -> str:
        return content if section_name == HEADER_SECTION else f"{section_name.capitalize()}\n{content}"
