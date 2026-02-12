from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from .models import DocumentChunk


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            return

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        embeddings = vectors
        metadatas = [
            {
                "file_name": chunk.file_name,
                "page": chunk.page,
                "chunk_id": chunk.id,
                "source_path": chunk.source_path,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_vector: list[float], top_k: int) -> dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self.collection.count()

    def get_all_metadata(self) -> list[dict[str, Any]]:
        data = self.collection.get(include=["metadatas"])
        return data.get("metadatas", [])
