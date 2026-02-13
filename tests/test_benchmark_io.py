import csv
import json
from pathlib import Path

import pytest

from scripts.benchmark_chatbot_vs_gemini import (
    CSV_COLUMNS,
    load_dataset,
    parse_judge_response,
    write_results_csv,
)


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
    assert rows[0]["question"] == "ลงทะเบียนเมื่อไหร่"
    assert rows[0]["reference_answer"] == "ตามปฏิทินการศึกษา"


def test_load_dataset_missing_required_field_raises(tmp_path: Path) -> None:
    dataset = tmp_path / "qa_prompt.json"
    dataset.write_text(
        json.dumps([{"id": "q1", "question": "x", "answer": ""}], ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_dataset(dataset)


def test_parse_judge_response_invalid_json_fallback() -> None:
    result, is_error = parse_judge_response("not json")
    assert is_error is True
    assert result["is_correct"] is False
    assert result["match_score"] == 0.0
    assert "judge_error:" in result["reason"]


def test_parse_judge_response_valid_json() -> None:
    result, is_error = parse_judge_response(
        '{"is_correct": true, "match_score": 0.91, "reason": "ตรงกับสาระหลัก"}'
    )
    assert is_error is False
    assert result["is_correct"] is True
    assert result["match_score"] == 0.91
    assert result["reason"] == "ตรงกับสาระหลัก"


def test_write_results_csv_has_expected_columns(tmp_path: Path) -> None:
    path = tmp_path / "result.csv"
    rows = [
        {
            "id": "q1",
            "source_document": "คู่มือนักศึกษา",
            "category": "การลงทะเบียน",
            "question": "ลงทะเบียนเมื่อไหร่",
            "reference_answer": "ตามปฏิทิน",
            "chatbot_answer": "ตอบจากแชตบอต",
            "gemini_answer": "ตอบจากเจมินี",
            "chatbot_is_correct": True,
            "chatbot_match_score": 1.0,
            "chatbot_judge_reason": "ok",
            "gemini_is_correct": False,
            "gemini_match_score": 0.2,
            "gemini_judge_reason": "missed key detail",
        }
    ]
    write_results_csv(path, rows)

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_COLUMNS
        content = list(reader)
        assert len(content) == 1
        assert content[0]["id"] == "q1"
