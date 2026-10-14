from pathlib import Path
from pypdf import PdfReader


def read_pdf_file(file_path: Path) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)
