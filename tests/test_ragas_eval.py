import csv
import json
from pathlib import Path

import pytest

from scripts.evaluate_ragas import (
    CSV_COLUMNS,
    build_ragas_samples,
    build_summary,
    collect_chatbot_outputs,
    load_dataset,
    merge_rows_with_scores,
    write_results_csv,
)


class FakeChunk:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeAnswer:
    def __init__(self, answer_text: str, citations: list[FakeChunk], refusal: bool) -> None:
        self.answer_text = answer_text
        self.citations = citations
        self.refusal = refusal


class FakePipeline:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def ask(self, query: str, history: list[dict]) -> FakeAnswer:
        if self.fail:
            raise RuntimeError("boom")
        return FakeAnswer("คำตอบจากบอท", [FakeChunk("บริบทที่ 1"), FakeChunk("บริบทที่ 2")], False)


def test_load_dataset_valid_shape(tmp_path: Path) -> None:
    dataset = tmp_path / "qa_prompt.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "source_document": "คู่มือนักศึกษา",
                    "category": "การลงทะเบียน",
                    "question": "ลงทะเบียนเมื่อไหร่",
                    "answer": "ตามปฏิทินการศึกษา",
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = load_dataset(dataset)
    assert len(rows) == 1
    assert rows[0]["id"] == "q1"
    assert rows[0]["reference_answer"] == "ตามปฏิทินการศึกษา"


def test_load_dataset_non_array_raises(tmp_path: Path) -> None:
    dataset = tmp_path / "qa_prompt.json"
    dataset.write_text(json.dumps({"id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(dataset)


def test_load_dataset_missing_required_field_raises(tmp_path: Path) -> None:
    dataset = tmp_path / "qa_prompt.json"
    dataset.write_text(
        json.dumps([{"id": "q1", "question": "", "answer": "x"}], ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_dataset(dataset)


def test_collect_chatbot_outputs_error_path() -> None:
    dataset = [
        {
            "id": "q1",
            "source_document": "doc",
            "category": "cat",
            "question": "x",
            "reference_answer": "ref",
        }
    ]
    rows, errors = collect_chatbot_outputs(dataset=dataset, pipeline=FakePipeline(fail=True))  # type: ignore[arg-type]
    assert errors["chatbot_answer_errors"] == 1
    assert rows[0]["chatbot_answer"].startswith("[ERROR]")
    assert rows[0]["retrieved_contexts"] == []


def test_merge_rows_and_summary_math() -> None:
    answer_rows = [
        {
            "id": "q1",
            "source_document": "doc1",
            "category": "cat1",
            "question": "q1",
            "reference_answer": "r1",
            "chatbot_answer": "a1",
            "refusal": False,
            "citation_count": 2,
        },
        {
            "id": "q2",
            "source_document": "doc2",
            "category": "cat2",
            "question": "q2",
            "reference_answer": "r2",
            "chatbot_answer": "a2",
            "refusal": True,
            "citation_count": 0,
        },
    ]
    score_rows = [
        {
            "faithfulness": 0.8,
            "answer_relevancy": 0.9,
            "context_precision": 0.7,
            "context_recall": 0.6,
        },
        {
            "faithfulness": None,
            "answer_relevancy": 0.5,
            "context_precision": None,
            "context_recall": 0.4,
        },
    ]
    merged = merge_rows_with_scores(answer_rows, score_rows)

    summary = build_summary(
        rows=merged,
        dataset_path=Path("data/json/qa_prompt.json"),
        ragas_llm_model="gemini-2.5-flash",
        ragas_embed_model="gemini-embedding-001",
        chatbot_model_desc="RAGPipeline(gen=x, embed=y)",
        errors={"chatbot_answer_errors": 1},
    )

    assert summary["metrics"]["faithfulness"] == 0.8
    assert summary["metrics"]["answer_relevancy"] == 0.7
    assert summary["metrics"]["context_precision"] == 0.7
    assert summary["metrics"]["context_recall"] == 0.5
    assert summary["metric_non_null_counts"]["faithfulness"] == 1
    assert summary["errors"]["ragas_metric_null_rows"] == 1
    assert summary["errors"]["refusal_count"] == 1


def test_write_results_csv_has_expected_columns(tmp_path: Path) -> None:
    output = tmp_path / "ragas_results.csv"
    rows = [
        {
            "id": "q1",
            "source_document": "doc1",
            "category": "cat1",
            "question": "q1",
            "reference_answer": "r1",
            "chatbot_answer": "a1",
            "refusal": False,
            "citation_count": 2,
            "faithfulness": 0.91,
            "answer_relevancy": 0.88,
            "context_precision": 0.76,
            "context_recall": 0.69,
        }
    ]
    write_results_csv(output, rows)

    with output.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_COLUMNS
        content = list(reader)
        assert len(content) == 1
        assert content[0]["id"] == "q1"


def test_build_ragas_samples_maps_expected_fields() -> None:
    rows = [
        {
            "question": "q1",
            "chatbot_answer": "a1",
            "reference_answer": "r1",
            "retrieved_contexts": ["c1", "c2"],
        }
    ]
    samples = build_ragas_samples(rows)
    assert samples == [
        {
            "user_input": "q1",
            "response": "a1",
            "reference": "r1",
            "retrieved_contexts": ["c1", "c2"],
        }
    ]
