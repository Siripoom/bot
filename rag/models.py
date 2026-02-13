from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    id: str
    file_name: str
    page: int
    text: str
    token_count: int
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk_id: str
    file_name: str
    page: int
    text: str
    score: float


@dataclass
class GateDecision:
    is_relevant: bool
    confidence: float
    reason: str
    kept_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class RAGAnswer:
    answer_text: str
    citations: list[RetrievedChunk]
    grounded: bool
    refusal: bool
