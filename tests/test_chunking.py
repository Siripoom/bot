from pathlib import Path

import pytest

from rag.chunker import build_document_chunks, chunk_text



def test_chunk_text_basic_split() -> None:
    text = " ".join(["word"] * 600)
    chunks = chunk_text(text, chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert all(0 < len(c) <= 120 for c in chunks)



def test_chunk_text_invalid_params() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0, overlap=0)

    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, overlap=100)



def test_build_document_chunks_shape() -> None:
    pages = [(1, "A" * 1000)]
    chunks = build_document_chunks(Path("demo.pdf"), pages, chunk_size=300, overlap=50)

    assert len(chunks) >= 3
    assert chunks[0].id.startswith("demo.pdf:p1:c")
    assert chunks[0].file_name == "demo.pdf"
    assert chunks[0].page == 1
