from pathlib import Path

from rag.config import Settings, REFUSAL_MESSAGE
from rag.models import GateDecision, RAGAnswer, RetrievedChunk
from rag.pipeline import RAGPipeline


class FakeEmbedder:
    pass


class FakeVectorStore:
    def __init__(self) -> None:
        self._count = 2

    def count(self) -> int:
        return self._count

    def get_all_metadata(self):
        return [{"file_name": "a.pdf"}, {"file_name": "b.pdf"}]


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedChunk]:
        return self.chunks[:top_k]


class FakeGate:
    def __init__(self, decision: GateDecision) -> None:
        self.decision = decision

    def judge_relevance(self, query: str, chunks: list[RetrievedChunk]) -> GateDecision:
        return self.decision


class FakeGenerator:
    def generate_answer(self, query: str, chunks: list[RetrievedChunk], history: list[dict]) -> RAGAnswer:
        return RAGAnswer(answer_text="คำตอบทดสอบ", citations=chunks, grounded=True, refusal=False)



def _settings(tmp_path: Path) -> Settings:
    log_file = tmp_path / "logs" / "app.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return Settings(
        gemini_api_key="test",
        pdf_dir=tmp_path / "data/pdfs",
        chroma_dir=tmp_path / "storage/chroma",
        log_file=log_file,
    )



def test_ask_refuses_when_score_gate_fails(tmp_path: Path) -> None:
    chunks = [RetrievedChunk("c1", "a.pdf", 1, "x", 0.2)]
    pipeline = RAGPipeline(
        settings=_settings(tmp_path),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        retriever=FakeRetriever(chunks),
        relevance_gate=FakeGate(GateDecision(True, 0.9, "ok", ["c1"])),
        generator=FakeGenerator(),
    )

    result = pipeline.ask("ถาม", history=[])
    assert result.refusal is True
    assert result.answer_text == REFUSAL_MESSAGE



def test_ask_refuses_when_llm_gate_rejects(tmp_path: Path) -> None:
    chunks = [RetrievedChunk("c1", "a.pdf", 1, "x", 0.8)]
    pipeline = RAGPipeline(
        settings=_settings(tmp_path),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        retriever=FakeRetriever(chunks),
        relevance_gate=FakeGate(GateDecision(False, 0.2, "insufficient", [])),
        generator=FakeGenerator(),
    )

    result = pipeline.ask("ถาม", history=[])
    assert result.refusal is True
    assert result.answer_text == REFUSAL_MESSAGE



def test_ask_returns_grounded_answer(tmp_path: Path) -> None:
    chunks = [
        RetrievedChunk("c1", "a.pdf", 1, "ข้อมูลสำคัญ", 0.91),
        RetrievedChunk("c2", "a.pdf", 2, "ข้อมูลเสริม", 0.88),
    ]
    pipeline = RAGPipeline(
        settings=_settings(tmp_path),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        retriever=FakeRetriever(chunks),
        relevance_gate=FakeGate(GateDecision(True, 0.9, "ok", ["c1"])),
        generator=FakeGenerator(),
    )

    result = pipeline.ask("ถาม", history=[])
    assert result.refusal is False
    assert result.grounded is True
    assert result.citations[0].chunk_id == "c1"
