from rag.gates import apply_score_gate, parse_gate_response_json
from rag.models import RetrievedChunk



def test_apply_score_gate_filters_by_threshold() -> None:
    chunks = [
        RetrievedChunk("c1", "a.pdf", 1, "text", 0.9),
        RetrievedChunk("c2", "a.pdf", 2, "text", 0.4),
    ]
    kept = apply_score_gate(chunks, threshold=0.58)
    assert [c.chunk_id for c in kept] == ["c1"]



def test_parse_gate_response_json_success() -> None:
    text = '{"is_relevant": true, "confidence": 0.87, "reason": "ok", "kept_chunk_ids": ["c1", "c3"]}'
    decision = parse_gate_response_json(text)

    assert decision.is_relevant is True
    assert decision.confidence == 0.87
    assert decision.kept_chunk_ids == ["c1", "c3"]



def test_parse_gate_response_json_invalid() -> None:
    decision = parse_gate_response_json("not json")
    assert decision.is_relevant is False
    assert decision.confidence == 0.0
