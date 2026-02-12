from __future__ import annotations

from pathlib import Path

from .models import DocumentChunk



def estimate_token_count(text: str) -> int:
    return len(text.split())



def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    n = len(normalized)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            split = normalized.rfind(" ", start, end)
            if split > start + (chunk_size // 2):
                end = split

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(0, end - overlap)

    return chunks



def build_document_chunks(
    pdf_path: Path,
    page_texts: list[tuple[int, str]],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[DocumentChunk]:
    file_name = pdf_path.name
    source_path = str(pdf_path)
    chunks: list[DocumentChunk] = []

    for page_num, text in page_texts:
        for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap), start=1):
            chunk_id = f"{file_name}:p{page_num}:c{idx}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    file_name=file_name,
                    page=page_num,
                    text=chunk,
                    token_count=estimate_token_count(chunk),
                    source_path=source_path,
                )
            )

    return chunks
