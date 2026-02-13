#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import load_settings
from rag.gates import apply_score_gate
from rag.direct_gemini import DirectGeminiClient
from rag.pipeline import RAGPipeline

CSV_COLUMNS = [
    "id",
    "source_document",
    "category",
    "question",
    "reference_answer",
    "chatbot_answer",
    "gemini_answer",
    "chatbot_is_correct",
    "chatbot_match_score",
    "chatbot_judge_reason",
    "gemini_is_correct",
    "gemini_match_score",
    "gemini_judge_reason",
]

REQUIRED_DATASET_FIELDS = ("id", "question", "answer")


class GeminiJudge:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_retries: int = 3,
        initial_backoff_sec: float = 1.0,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max(1, max_retries)
        self.initial_backoff_sec = max(0.0, initial_backoff_sec)

    def judge(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
    ) -> tuple[dict[str, Any], bool]:
        prompt = build_judge_prompt(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": 0, "top_p": 1},
                )
                result, judge_error = parse_judge_response(_response_text(response))
                return result, judge_error
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = self.initial_backoff_sec * (2 ** (attempt - 1))
                if delay > 0:
                    time.sleep(delay)

        return {
            "is_correct": False,
            "match_score": 0.0,
            "reason": f"judge_error:{last_error}",
        }, True


def build_judge_prompt(question: str, reference_answer: str, candidate_answer: str) -> str:
    return (
        "คุณเป็นกรรมการตรวจคำตอบแบบเข้มงวด\n"
        "เป้าหมาย: เปรียบเทียบคำตอบผู้สมัครกับคำตอบอ้างอิงแล้วตัดสินถูก/ผิดเท่านั้น\n"
        "กติกา:\n"
        "1) ถ้าคำตอบใกล้เคียงสาระหลักของคำตอบอ้างอิง ให้ is_correct=true\n"
        "2) ถ้าคลาดเคลื่อน ขาดสาระหลัก หรือปฏิเสธผิดบริบท ให้ is_correct=false\n"
        "3) คำตอบที่ถูกเพียงบางส่วน ให้ตัดเป็น is_correct=false\n"
        "4) ส่งออก JSON เท่านั้น ตาม schema นี้พอดี:\n"
        '{"is_correct": bool, "match_score": float, "reason": str}\n'
        "5) match_score ต้องอยู่ในช่วง 0 ถึง 1\n\n"
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE_ANSWER:\n{reference_answer}\n\n"
        f"CANDIDATE_ANSWER:\n{candidate_answer}\n"
    )


def parse_judge_response(text: str) -> tuple[dict[str, Any], bool]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {
            "is_correct": False,
            "match_score": 0.0,
            "reason": "judge_error:invalid_json",
        }, True

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "is_correct": False,
            "match_score": 0.0,
            "reason": "judge_error:json_decode_error",
        }, True

    raw_is_correct = parsed.get("is_correct", False)
    if isinstance(raw_is_correct, bool):
        is_correct = raw_is_correct
    elif isinstance(raw_is_correct, str):
        is_correct = raw_is_correct.strip().lower() in {"true", "1", "yes"}
    else:
        is_correct = bool(raw_is_correct)

    try:
        score = float(parsed.get("match_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0

    reason = str(parsed.get("reason", "")).strip() or "judge_error:missing_reason"
    score = max(0.0, min(1.0, score))
    return {
        "is_correct": is_correct,
        "match_score": score,
        "reason": reason,
    }, False


def load_dataset(dataset_path: Path) -> list[dict[str, str]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Dataset must be a JSON array")

    rows: list[dict[str, str]] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Record {idx}: item must be an object")

        missing = [
            field for field in REQUIRED_DATASET_FIELDS
            if not isinstance(item.get(field), str) or not item.get(field).strip()
        ]
        if missing:
            raise ValueError(f"Record {idx}: missing required fields {missing}")

        rows.append(
            {
                "id": str(item["id"]).strip(),
                "source_document": str(item.get("source_document", "")).strip() or "unknown",
                "category": str(item.get("category", "")).strip() or "unknown",
                "question": str(item["question"]).strip(),
                "reference_answer": str(item["answer"]).strip(),
            }
        )
    return rows


def compute_binary_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    total = len(rows)
    correct_count = sum(1 for row in rows if bool(row.get(f"{prefix}_is_correct")))
    incorrect_count = total - correct_count
    correct_pct = (correct_count * 100.0 / total) if total else 0.0
    incorrect_pct = (incorrect_count * 100.0 / total) if total else 0.0
    return {
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "correct_pct": round(correct_pct, 2),
        "incorrect_pct": round(incorrect_pct, 2),
    }


def compute_breakdown(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field, "unknown")) or "unknown"
        grouped.setdefault(key, []).append(row)

    summary: dict[str, Any] = {}
    for key in sorted(grouped):
        subset = grouped[key]
        summary[key] = {
            "total": len(subset),
            "chatbot_metrics": compute_binary_metrics(subset, prefix="chatbot"),
            "gemini_metrics": compute_binary_metrics(subset, prefix="gemini"),
        }
    return summary


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


def build_summary(
    rows: list[dict[str, Any]],
    dataset_path: Path,
    model_chatbot: str,
    model_gemini_direct: str,
    model_judge: str,
    errors: dict[str, int],
) -> dict[str, Any]:
    chatbot_metrics = compute_binary_metrics(rows, prefix="chatbot")
    gemini_metrics = compute_binary_metrics(rows, prefix="gemini")
    return {
        "dataset_path": str(dataset_path),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_chatbot": model_chatbot,
        "model_gemini_direct": model_gemini_direct,
        "model_judge": model_judge,
        "total_questions": len(rows),
        "chatbot_metrics": chatbot_metrics,
        "gemini_metrics": gemini_metrics,
        "delta_correct_pct": round(chatbot_metrics["correct_pct"] - gemini_metrics["correct_pct"], 2),
        "by_source_document": compute_breakdown(rows, field="source_document"),
        "by_category": compute_breakdown(rows, field="category"),
        "errors": errors,
    }


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark chatbot vs direct Gemini with qa_prompt dataset")
    parser.add_argument("--dataset", type=str, default="data/json/qa_prompt.json")
    parser.add_argument("--out-csv", type=str, default="reports/benchmark_results.csv")
    parser.add_argument("--out-json", type=str, default="reports/benchmark_summary.json")
    parser.add_argument("--gemini-model", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument(
        "--gemini-context-mode",
        type=str,
        choices=("none", "qa_fallback", "pdf"),
        default="pdf",
        help="Context mode for direct Gemini baseline",
    )
    parser.add_argument(
        "--gemini-context-top-k",
        type=int,
        default=3,
        help="Number of QA fallback chunks to pass to direct Gemini when context mode is enabled",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only first N records for debug")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()

    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be > 0")
        dataset = dataset[: args.limit]

    pipeline = RAGPipeline(settings=settings)
    direct_model = args.gemini_model or settings.gen_model
    judge_model = args.judge_model or settings.gen_model
    direct_client = DirectGeminiClient(api_key=settings.gemini_api_key, model=direct_model)
    judge = GeminiJudge(api_key=settings.gemini_api_key, model=judge_model)

    rows: list[dict[str, Any]] = []
    errors = {
        "chatbot_answer_errors": 0,
        "gemini_answer_errors": 0,
        "chatbot_judge_errors": 0,
        "gemini_judge_errors": 0,
    }

    for idx, item in enumerate(dataset, start=1):
        question = item["question"]
        reference_answer = item["reference_answer"]

        try:
            chatbot_answer = pipeline.ask(question, history=[]).answer_text
        except Exception as exc:
            chatbot_answer = f"[ERROR] {exc}"
            errors["chatbot_answer_errors"] += 1

        gemini_context_chunks: list[str] = []
        if args.gemini_context_mode == "qa_fallback":
            retrieved_ctx = pipeline.retrieve_qa_fallback(
                question,
                top_k=max(1, args.gemini_context_top_k),
            )
            scored_ctx = apply_score_gate(retrieved_ctx, threshold=settings.qa_sim_threshold)
            picked = scored_ctx if scored_ctx else retrieved_ctx
            gemini_context_chunks = [
                f"[QA:{chunk.file_name}] {chunk.text}"
                for chunk in picked[: max(1, args.gemini_context_top_k)]
                if chunk.text
            ]
        elif args.gemini_context_mode == "pdf":
            retrieved_ctx = pipeline.retrieve(
                question,
                top_k=max(1, args.gemini_context_top_k),
            )
            scored_ctx = apply_score_gate(retrieved_ctx, threshold=settings.sim_threshold)
            picked = scored_ctx if scored_ctx else retrieved_ctx
            gemini_context_chunks = [
                f"[PDF:{chunk.file_name} หน้า {chunk.page}] {chunk.text}"
                for chunk in picked[: max(1, args.gemini_context_top_k)]
                if chunk.text
            ]

        try:
            gemini_answer = direct_client.answer_question(question, context_chunks=gemini_context_chunks)
        except Exception as exc:
            gemini_answer = f"[ERROR] {exc}"
            errors["gemini_answer_errors"] += 1

        chatbot_eval, chatbot_eval_error = judge.judge(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=chatbot_answer,
        )
        if chatbot_eval_error:
            errors["chatbot_judge_errors"] += 1

        gemini_eval, gemini_eval_error = judge.judge(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=gemini_answer,
        )
        if gemini_eval_error:
            errors["gemini_judge_errors"] += 1

        rows.append(
            {
                "id": item["id"],
                "source_document": item["source_document"],
                "category": item["category"],
                "question": question,
                "reference_answer": reference_answer,
                "chatbot_answer": chatbot_answer,
                "gemini_answer": gemini_answer,
                "chatbot_is_correct": chatbot_eval["is_correct"],
                "chatbot_match_score": chatbot_eval["match_score"],
                "chatbot_judge_reason": chatbot_eval["reason"],
                "gemini_is_correct": gemini_eval["is_correct"],
                "gemini_match_score": gemini_eval["match_score"],
                "gemini_judge_reason": gemini_eval["reason"],
            }
        )
        print(f"[{idx}/{len(dataset)}] done id={item['id']}")

    summary = build_summary(
        rows=rows,
        dataset_path=dataset_path,
        model_chatbot=f"RAGPipeline(gen={settings.gen_model}, embed={settings.embed_model})",
        model_gemini_direct=f"{direct_model} (context_mode={args.gemini_context_mode})",
        model_judge=judge_model,
        errors=errors,
    )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    write_results_csv(out_csv, rows)
    write_summary_json(out_json, summary)

    print("===== Benchmark Summary =====")
    print(f"Total questions: {summary['total_questions']}")
    print(f"Chatbot correct: {summary['chatbot_metrics']['correct_pct']}%")
    print(f"Gemini direct correct: {summary['gemini_metrics']['correct_pct']}%")
    print(f"Delta correct pct (chatbot-gemini): {summary['delta_correct_pct']}%")
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text

    if hasattr(response, "candidates"):
        parts: list[str] = []
        for candidate in response.candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                candidate_text = getattr(part, "text", None)
                if candidate_text:
                    parts.append(candidate_text)
        if parts:
            return "\n".join(parts)

    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        return json.dumps(dumped, ensure_ascii=False)

    return ""


if __name__ == "__main__":
    main()
