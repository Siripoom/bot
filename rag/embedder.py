from __future__ import annotations

from typing import Any

from google import genai


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.models.embed_content(model=self.model, contents=batch)
            vectors.extend(self._extract_vectors(response))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(model=self.model, contents=[text])
        vectors = self._extract_vectors(response)
        if not vectors:
            raise RuntimeError("embedding response is empty")
        return vectors[0]

    def _extract_vectors(self, response: Any) -> list[list[float]]:
        vectors: list[list[float]] = []

        embeddings = getattr(response, "embeddings", None)
        if embeddings:
            for emb in embeddings:
                values = getattr(emb, "values", None)
                if values:
                    vectors.append(list(values))

        if vectors:
            return vectors

        # Fallback if SDK shape changes.
        as_dict = response.model_dump() if hasattr(response, "model_dump") else {}
        items = as_dict.get("embeddings", [])
        for item in items:
            values = item.get("values")
            if values:
                vectors.append(list(values))

        return vectors
