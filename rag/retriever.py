from __future__ import annotations

from .embedder import GeminiEmbedder
from .models import RetrievedChunk
from .vector_store import ChromaVectorStore


class Retriever:
    def __init__(self, embedder: GeminiEmbedder, vector_store: ChromaVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        query_vector = self.embedder.embed_query(query)
        raw = self.vector_store.query(query_vector=query_vector, top_k=top_k)

        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[RetrievedChunk] = []
        for text, meta, distance in zip(docs, metas, distances):
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            results.append(
                RetrievedChunk(
                    chunk_id=str(meta.get("chunk_id", "")),
                    file_name=str(meta.get("file_name", "unknown")),
                    page=int(meta.get("page", 0)),
                    text=str(text or ""),
                    score=similarity,
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results
