from __future__ import annotations

import json
import logging
from pathlib import Path

from .chunker import estimate_token_count
from .embedder import GeminiEmbedder
from .models import DocumentChunk, RetrievedChunk
from .vector_store import ChromaVectorStore


def load_qa_entries(json_path: Path, logger: logging.Logger | None = None) -> list[DocumentChunk]:
    path = Path(json_path)
    if not path.exists():
        if logger:
            logger.warning("qa_fallback_missing_file path=%s", path)
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        if logger:
            logger.warning("qa_fallback_invalid_json path=%s error=%s", path, exc)
        return []

    if not isinstance(payload, list):
        if logger:
            logger.warning("qa_fallback_invalid_shape path=%s type=%s", path, type(payload).__name__)
        return []

    entries: list[DocumentChunk] = []
    seen_ids: dict[str, int] = {}
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue

        question = _clean_text(item.get("question"))
        answer = _clean_text(item.get("answer"))
        if not question or not answer:
            continue

        qa_id = _clean_text(item.get("id")) or f"row-{index}"
        chunk_id = f"qa:{qa_id}"
        if chunk_id in seen_ids:
            seen_ids[chunk_id] += 1
            chunk_id = f"{chunk_id}:{seen_ids[chunk_id]}"
        else:
            seen_ids[chunk_id] = 1

        source_document = _clean_text(item.get("source_document")) or path.stem
        category = _clean_text(item.get("category")) or "ทั่วไป"
        combined_text = (
            f"คำถาม: {question}\n"
            f"หมวดหมู่: {category}\n"
            f"คำตอบ: {answer}"
        )
        entries.append(
            DocumentChunk(
                id=chunk_id,
                file_name=source_document,
                page=0,
                text=combined_text,
                token_count=estimate_token_count(combined_text),
                source_path=str(path),
                metadata={
                    "qa_id": qa_id,
                    "source_document": source_document,
                    "category": category,
                    "type": "qa_fallback",
                },
            )
        )

    return entries


class QAFallbackRetriever:
    def __init__(
        self,
        embedder: GeminiEmbedder,
        persist_dir: Path,
        collection_name: str,
        qa_json_path: Path,
        logger: logging.Logger | None = None,
        vector_store: ChromaVectorStore | None = None,
    ) -> None:
        self.embedder = embedder
        self.logger = logger or logging.getLogger(__name__)
        self.qa_json_path = Path(qa_json_path)
        self.vector_store = vector_store or ChromaVectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
        self.enabled = False
        self.index_size = 0
        self.ensure_index()

    def ensure_index(self) -> int:
        entries = load_qa_entries(self.qa_json_path, logger=self.logger)
        if not entries:
            self.enabled = False
            self.index_size = 0
            return 0

        try:
            current_count = self.vector_store.count()
            if current_count != len(entries):
                self.vector_store.reset()
                vectors = self.embedder.embed_texts([entry.text for entry in entries])
                self.vector_store.upsert_chunks(entries, vectors)
                self.logger.info(
                    "qa_fallback_indexed path=%s chunks=%s",
                    self.qa_json_path,
                    len(entries),
                )
            self.enabled = True
            self.index_size = len(entries)
            return self.index_size
        except Exception as exc:
            self.logger.warning("qa_fallback_index_failed path=%s error=%s", self.qa_json_path, exc)
            self.enabled = False
            self.index_size = 0
            return 0

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.enabled or not query.strip():
            return []

        try:
            query_vector = self.embedder.embed_query(query)
            raw = self.vector_store.query(query_vector=query_vector, top_k=top_k)
        except Exception as exc:
            self.logger.warning("qa_fallback_query_failed error=%s", exc)
            return []

        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[RetrievedChunk] = []
        for text, meta, distance in zip(docs, metas, distances):
            metadata = meta if isinstance(meta, dict) else {}
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            page = _as_int(metadata.get("page"), default=0)
            file_name = str(metadata.get("source_document") or metadata.get("file_name") or "qa_fallback")
            chunk_id = str(metadata.get("chunk_id") or metadata.get("qa_id") or "")
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    file_name=file_name,
                    page=page,
                    text=str(text or ""),
                    score=score,
                )
            )

        results.sort(key=lambda chunk: chunk.score, reverse=True)
        return results


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())
