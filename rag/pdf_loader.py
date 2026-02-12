from __future__ import annotations

from pathlib import Path

import fitz



def list_pdf_files(pdf_dir: Path) -> list[Path]:
    if not pdf_dir.exists():
        return []
    return sorted(path for path in pdf_dir.iterdir() if path.suffix.lower() == ".pdf")



def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            cleaned = " ".join(text.split())
            if cleaned:
                pages.append((page_index, cleaned))
    return pages
