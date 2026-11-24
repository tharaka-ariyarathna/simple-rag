import pytest
from rag.indexing.redaction import find_leaks, redact_sections, redact_text
from rag.indexing.section_chunker import Section


@pytest.mark.parametrize("text, expected", [
    ("Mail ines.brooks86@example.com today", "Mail [EMAIL] today"),
    ("See https://github.com/ines/repo and www.ines.dev", "See [URL] and [URL]"),
    ("linkedin.com/in/ines-brooks", "[URL]"),
    ("Call +46 70 123 45 67 or (070) 123-4567", "Call [PHONE] or [PHONE]"),
])
def test_contacts_are_replaced(text, expected):
    assert redact_text(text, None) == expected


@pytest.mark.parametrize("text", ["2019 - 2021", "2024 - Present", "reduced cost by 40% in 2023", "Python 3.12"])
def test_dates_and_numbers_are_not_mistaken_for_phones(text):
    assert redact_text(text, None) == text


def test_the_name_and_its_parts_become_the_candidate_ignoring_case():
    text = "Ines Brooks led the team. BROOKS also mentored; ines wrote docs. Brookside Ltd is a client."

    redacted = redact_text(text, "Ines Brooks")

    assert redacted == ("the candidate led the team. the candidate also mentored; the candidate wrote docs. "
                        "Brookside Ltd is a client.")  # whole words only


def test_very_short_name_parts_are_not_replaced_alone():
    assert redact_text("Li Wei and Li-ion batteries", "Li Wei") == "the candidate and Li-ion batteries"


def test_contact_and_referee_sections_are_dropped_and_the_rest_redacted():
    sections = [Section("header", "Ines Brooks | ines@example.com"), Section("contact", "+46 70 123 45 67"),
                Section("skills", "Python"), Section("referees", "Jane Doe, jane@example.com")]

    redacted = redact_sections(sections, "Ines Brooks")

    assert [(section.name, section.text) for section in redacted] == [
        ("header", "the candidate | [EMAIL]"), ("skills", "Python")]


def test_find_leaks_reports_what_redaction_missed():
    assert find_leaks("the candidate | [EMAIL]", "Ines Brooks") == []
    assert find_leaks("Ines, mail ines@example.com", "Ines Brooks") == ["ines@example.com", "Ines"]
