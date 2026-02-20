from rag.gates import LLMRelevanceGate
from rag.generator import GeminiAnswerGenerator
from rag.models import RetrievedChunk


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.text)


class _FakeClient:
    def __init__(self, models: _FakeModels) -> None:
        self.models = models


def test_relevance_gate_passes_generation_config(monkeypatch) -> None:
    models = _FakeModels('{"is_relevant": true, "confidence": 0.9, "reason": "ok", "kept_chunk_ids": ["c1"]}')
    monkeypatch.setattr("rag.gates.genai.Client", lambda api_key: _FakeClient(models))
    gate = LLMRelevanceGate(api_key="k", model="m", temperature=0.12, top_p=0.34)

    decision = gate.judge_relevance(
        query="คำถาม",
        chunks=[RetrievedChunk("c1", "a.pdf", 1, "เนื้อหา", 0.9)],
    )

    assert decision.is_relevant is True
    assert models.calls[0]["config"] == {"temperature": 0.12, "top_p": 0.34}


def test_generator_passes_generation_config(monkeypatch) -> None:
    models = _FakeModels("คำตอบทดสอบ")
    monkeypatch.setattr("rag.generator.genai.Client", lambda api_key: _FakeClient(models))
    generator = GeminiAnswerGenerator(api_key="k", model="m", temperature=0.56, top_p=0.78)

    answer = generator.generate_answer(
        query="คำถาม",
        chunks=[RetrievedChunk("c1", "a.pdf", 1, "เนื้อหา", 0.9)],
        history=[],
    )

    assert answer.refusal is False
    assert models.calls[0]["config"] == {"temperature": 0.56, "top_p": 0.78}
