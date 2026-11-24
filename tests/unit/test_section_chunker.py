from rag.indexing.chunker import TokenChunker
from rag.indexing.section_chunker import SectionChunker, heading_of, split_sections
from tests.fakes import WordTokenizer

CV = """Ines Brooks
Machine Learning Engineer | Stockholm
Summary
ML engineer with 5 years of experience.
Skills
TensorFlow, Spark, Python
Experience
Machine Learning Engineer - Northfield Data
2024 - Present
running in parallel
Referees
Jane Doe, jane@example.com"""


def test_a_cv_is_cut_at_its_known_headings():
    sections = split_sections(CV)

    assert [section.name for section in sections] == ["header", "summary", "skills", "experience", "referees"]
    assert sections[2].text == "TensorFlow, Spark, Python"


def test_only_exact_known_headings_start_a_section():
    assert heading_of("Work History:") == "experience"
    assert heading_of("  CORE SKILLS ") == "skills"
    assert heading_of("running in parallel") is None  # a short line of body text
    assert heading_of("Experience with Spark") is None


def test_a_cv_without_headings_is_one_header_section():
    assert [(section.name, section.text) for section in split_sections("just text\nmore text")] == [
        ("header", "just text\nmore text")]


def test_empty_sections_are_dropped():
    assert [section.name for section in split_sections("Summary\n\nSkills\nPython")] == ["skills"]


def test_no_chunk_mixes_two_sections_and_each_is_titled_with_its_section():
    chunker = SectionChunker(TokenChunker(WordTokenizer(), chunk_size=4, chunk_overlap=0))

    chunks = chunker.chunk_sections(split_sections(CV))

    assert {chunk.section for chunk in chunks} == {"header", "summary", "skills", "experience", "referees"}
    skills = [chunk.content for chunk in chunks if chunk.section == "skills"]
    assert skills == ["Skills\nTensorFlow, Spark, Python"]
    assert not any("Spark" in chunk.content for chunk in chunks if chunk.section == "experience")
    assert [chunk.content for chunk in chunks if chunk.section == "header"][0].startswith("Ines Brooks")
