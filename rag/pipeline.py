from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .chunker import build_document_chunks
from .config import REFUSAL_MESSAGE, Settings, load_settings
from .embedder import GeminiEmbedder
from .gates import LLMRelevanceGate, apply_score_gate
from .generator import GeminiAnswerGenerator
from .models import GateDecision, RAGAnswer, RetrievedChunk
from .pdf_loader import extract_pdf_pages, list_pdf_files
from .retriever import Retriever
from .vector_store import ChromaVectorStore


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: GeminiEmbedder | None = None,
        vector_store: ChromaVectorStore | None = None,
        retriever: Retriever | None = None,
        relevance_gate: LLMRelevanceGate | None = None,
        generator: GeminiAnswerGenerator | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.logger = _build_logger(self.settings.log_file)

        self.embedder = embedder or GeminiEmbedder(
            api_key=self.settings.gemini_api_key,
            model=self.settings.embed_model,
        )
        self.vector_store = vector_store or ChromaVectorStore(
            persist_dir=self.settings.chroma_dir,
            collection_name=self.settings.collection_name,
        )
        self.retriever = retriever or Retriever(self.embedder, self.vector_store)
        self.relevance_gate = relevance_gate or LLMRelevanceGate(
            api_key=self.settings.gemini_api_key,
            model=self.settings.gen_model,
        )
        self.generator = generator or GeminiAnswerGenerator(
            api_key=self.settings.gemini_api_key,
            model=self.settings.gen_model,
        )

    def index_pdfs(self, pdf_dir: str | None = None, reset: bool = False) -> dict[str, Any]:
        directory = Path(pdf_dir) if pdf_dir else self.settings.pdf_dir
        pdf_files = list_pdf_files(directory)

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {directory}")

        if reset:
            self.vector_store.reset()

        total_chunks = 0
        indexed_docs = 0

        for pdf_path in pdf_files:
            pages = extract_pdf_pages(pdf_path)
            chunks = build_document_chunks(pdf_path=pdf_path, page_texts=pages)
            if not chunks:
                continue

            vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
            self.vector_store.upsert_chunks(chunks, vectors)
            total_chunks += len(chunks)
            indexed_docs += 1

        stats = self.get_index_stats()
        result = {
            "indexed_files": indexed_docs,
            "indexed_chunks": total_chunks,
            "collection_count": stats["chunk_count"],
            "document_count": stats["document_count"],
            "pdf_dir": str(directory),
        }
        self.logger.info("index_complete %s", result)
        return result

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedChunk]:
        return self.retriever.retrieve(query=query, top_k=top_k)

    def judge_relevance(self, query: str, chunks: list[RetrievedChunk]) -> GateDecision:
        return self.relevance_gate.judge_relevance(query=query, chunks=chunks)

    def generate_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[dict],
    ) -> RAGAnswer:
        return self.generator.generate_answer(query=query, chunks=chunks, history=history)

    def ask(self, query: str, history: list[dict]) -> RAGAnswer:
        start_time = time.perf_counter()

        retrieved = self.retrieve(query=query, top_k=self.settings.top_k)
        scored = apply_score_gate(retrieved, threshold=self.settings.sim_threshold)

        if not scored:
            answer = self._refusal_answer()
            self._log_query(query, retrieved, None, answer, elapsed=time.perf_counter() - start_time)
            return answer

        decision = self.judge_relevance(query=query, chunks=scored)
        if not decision.is_relevant or decision.confidence < self.settings.gate_conf_threshold:
            answer = self._refusal_answer(citations=scored[:3])
            self._log_query(query, retrieved, decision, answer, elapsed=time.perf_counter() - start_time)
            return answer

        if decision.kept_chunk_ids:
            selected = [c for c in scored if c.chunk_id in set(decision.kept_chunk_ids)]
        else:
            selected = scored

        selected = selected[:5]

        answer = self.generate_answer(query=query, chunks=selected, history=history)
        self._log_query(query, retrieved, decision, answer, elapsed=time.perf_counter() - start_time)
        return answer

    def has_index(self) -> bool:
        return self.vector_store.count() > 0

    def get_index_stats(self) -> dict[str, int]:
        chunk_count = self.vector_store.count()
        metadatas = self.vector_store.get_all_metadata()

        files = {
            str(meta.get("file_name"))
            for meta in metadatas
            if isinstance(meta, dict) and meta.get("file_name")
        }
        return {
            "chunk_count": int(chunk_count),
            "document_count": len(files),
        }

    def _refusal_answer(self, citations: list[RetrievedChunk] | None = None) -> RAGAnswer:
        return RAGAnswer(
            answer_text=REFUSAL_MESSAGE,
            citations=citations or [],
            grounded=False,
            refusal=True,
        )

    def _log_query(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        decision: GateDecision | None,
        answer: RAGAnswer,
        elapsed: float,
    ) -> None:
        self.logger.info(
            "query=%s elapsed_ms=%.2f retrieved_ids=%s scores=%s gate=%s refusal=%s",
            query,
            elapsed * 1000,
            [r.chunk_id for r in retrieved],
            [round(r.score, 4) for r in retrieved],
            decision,
            answer.refusal,
        )


def _build_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("rag_pipeline")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
