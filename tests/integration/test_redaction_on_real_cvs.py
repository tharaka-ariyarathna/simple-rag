"""Redaction over every real CV in DOCS_DIR: after redaction no section may hold an email, phone, URL or the name.
The true names come from the dataset's manifest.csv (next to the cvs folder), standing in for profile extraction.
Half the CVs are two-column, where the name is not on the first line, so the manifest is needed."""
import csv
import os
from pathlib import Path
import pytest
from dotenv import load_dotenv
from rag.indexing.pdf_reader import read_pdf_file
from rag.indexing.redaction import find_leaks, redact_sections
from rag.indexing.section_chunker import split_sections


def names_by_file() -> dict[Path, str]:
    load_dotenv()
    docs_dir = Path(os.environ.get("DOCS_DIR") or "missing")
    manifest = docs_dir.parent / "manifest.csv"
    if not manifest.exists():
        pytest.skip("DOCS_DIR with the CV PDFs and manifest.csv is not available")
    with manifest.open(encoding="utf-8") as file:
        return {docs_dir / f"{row['cv_id']}.pdf": row["name"] for row in csv.DictReader(file)}


def test_no_personal_data_survives_redaction():
    leaks = {}
    for path, name in names_by_file().items():
        for section in redact_sections(split_sections(read_pdf_file(path)), name):
            found = find_leaks(section.text, name)
            if found:
                leaks.setdefault(path.name, []).extend(found)

    assert leaks == {}
