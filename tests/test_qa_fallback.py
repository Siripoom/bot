from pathlib import Path

from rag.qa_fallback import load_qa_entries


def test_load_qa_entries_skips_empty_answers(tmp_path: Path) -> None:
    qa_json = tmp_path / "qa_prompt.json"
    qa_json.write_text(
        """
[
  {
    "id": "q1",
    "source_document": "คู่มือนักศึกษา",
    "category": "การลงทะเบียน",
    "question": "ลงทะเบียนเมื่อไหร่",
    "answer": "ลงทะเบียนได้ตามปฏิทินการศึกษา"
  },
  {
    "id": "q2",
    "source_document": "คู่มือนักศึกษา",
    "category": "การลา",
    "question": "ลาป่วยอย่างไร",
    "answer": ""
  }
]
        """.strip(),
        encoding="utf-8",
    )

    entries = load_qa_entries(qa_json)
    assert len(entries) == 1
    assert entries[0].id == "qa:q1"
    assert entries[0].metadata["type"] == "qa_fallback"
    assert entries[0].metadata["source_document"] == "คู่มือนักศึกษา"


def test_load_qa_entries_returns_empty_for_invalid_json(tmp_path: Path) -> None:
    qa_json = tmp_path / "qa_prompt.json"
    qa_json.write_text("{invalid", encoding="utf-8")

    entries = load_qa_entries(qa_json)
    assert entries == []


def test_load_qa_entries_returns_empty_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    entries = load_qa_entries(missing)
    assert entries == []
