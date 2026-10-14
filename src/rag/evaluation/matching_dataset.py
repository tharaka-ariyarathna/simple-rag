import json
from dataclasses import dataclass
from pathlib import Path
from rag.indexing.pdf_reader import read_pdf_file

LABELS_FILE_NAME = "labels.json"  # {"jd_001": {"cv_001": 1, "cv_002": 0, ...}, ...} grades: 0 none, 1 partial, 2 strong
JDS_FOLDER_NAME = "jds"


@dataclass
class MatchingDataset:
    jd_texts: dict[str, str]  # jd id -> text of the JD PDF
    labels: dict[str, dict[str, int]]  # jd id -> {cv id -> grade}
    skipped_jd_ids: list[str]  # labelled, but no PDF to search with


def load_matching_dataset(dataset_dir: Path) -> MatchingDataset:
    """Reads the labels and the JD PDFs that have labels. A JD id is the PDF's file name without .pdf."""
    labels = json.loads((dataset_dir / LABELS_FILE_NAME).read_text(encoding="utf-8"))
    jd_texts = {}
    for jd_id in sorted(labels):
        jd_path = dataset_dir / JDS_FOLDER_NAME / f"{jd_id}.pdf"
        if jd_path.exists():
            jd_texts[jd_id] = read_pdf_file(jd_path)
    skipped_jd_ids = [jd_id for jd_id in sorted(labels) if jd_id not in jd_texts]
    return MatchingDataset(jd_texts=jd_texts, labels={jd_id: labels[jd_id] for jd_id in jd_texts},
                           skipped_jd_ids=skipped_jd_ids)
