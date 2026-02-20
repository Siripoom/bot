#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import load_settings
from rag.pipeline import RAGPipeline

CSV_COLUMNS = [
    "id",
    "source_document",
    "category",
    "question",
    "reference_answer",
    "chatbot_answer",
    "refusal",
    "citation_count",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

REQUIRED_DATASET_FIELDS = ("id", "question", "answer")
METRIC_COLUMNS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "faithfulness": ("faithfulness",),
    "answer_relevancy": ("answer_relevancy", "response_relevancy", "answer_relevance"),
    "context_precision": (
        "context_precision",
        "llm_context_precision_with_reference",
        "context_precision_with_reference",
    ),
    "context_recall": ("context_recall", "llm_context_recall"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate chatbot quality with Ragas")
    parser.add_argument("--dataset", type=str, default="data/json/qa_prompt.json")
    parser.add_argument("--out-csv", type=str, default="reports/ragas_results.csv")
    parser.add_argument("--out-json", type=str, default="reports/ragas_summary.json")
    parser.add_argument(
        "--ragas-llm-model",
        type=str,
        default=None,
        help="Model used by Ragas LLM metrics (default: GEN_MODEL env or gemini-2.5-flash)",
    )
    parser.add_argument(
        "--ragas-embed-model",
        type=str,
        default=None,
        help="Model used by Ragas embedding metrics (default: EMBED_MODEL env or gemini-embedding-001)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only first N records for debug")
    return parser.parse_args()


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
            field
            for field in REQUIRED_DATASET_FIELDS
            if not isinstance(item.get(field), str) or not item.get(field, "").strip()
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


def collect_chatbot_outputs(
    dataset: list[dict[str, str]],
    pipeline: RAGPipeline,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    errors = {"chatbot_answer_errors": 0}

    for idx, item in enumerate(dataset, start=1):
        question = item["question"]
        try:
            result = pipeline.ask(question, history=[])
            chatbot_answer = result.answer_text
            retrieved_contexts = [chunk.text for chunk in result.citations if chunk.text]
            refusal = bool(result.refusal)
            citation_count = len(result.citations)
        except Exception as exc:
            chatbot_answer = f"[ERROR] {exc}"
            retrieved_contexts = []
            refusal = False
            citation_count = 0
            errors["chatbot_answer_errors"] += 1

        rows.append(
            {
                **item,
                "chatbot_answer": chatbot_answer,
                "retrieved_contexts": retrieved_contexts,
                "citation_count": citation_count,
                "refusal": refusal,
            }
        )
        print(f"[{idx}/{len(dataset)}] done id={item['id']}")
    return rows, errors


def build_ragas_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "user_input": row["question"],
            "response": row["chatbot_answer"],
            "reference": row["reference_answer"],
            "retrieved_contexts": list(row.get("retrieved_contexts", [])),
        }
        for row in rows
    ]


def run_ragas(
    samples: list[dict[str, Any]],
    api_key: str,
    llm_model: str,
    embed_model: str,
) -> list[dict[str, float | None]]:
    runtime = _import_ragas_runtime()

    dataset = runtime["EvaluationDataset"].from_list(samples)
    llm_client = runtime["ChatGoogleGenerativeAI"](
        model=llm_model,
        google_api_key=api_key,
        temperature=0,
    )
    embed_client = runtime["GoogleGenerativeAIEmbeddings"](
        model=embed_model,
        google_api_key=api_key,
    )
    llm_wrapper = runtime["LangchainLLMWrapper"](llm_client)
    embed_wrapper = runtime["LangchainEmbeddingsWrapper"](embed_client)

    metrics = [
        runtime["Faithfulness"](llm=llm_wrapper),
        runtime["ResponseRelevancy"](llm=llm_wrapper, embeddings=embed_wrapper),
        runtime["LLMContextPrecisionWithReference"](llm=llm_wrapper),
        runtime["LLMContextRecall"](llm=llm_wrapper),
    ]

    result = _evaluate_with_ragas(
        evaluate_fn=runtime["evaluate"],
        dataset=dataset,
        metrics=metrics,
        llm=llm_wrapper,
        embeddings=embed_wrapper,
    )
    score_rows = _extract_score_rows(result, sample_count=len(samples))
    return [_normalize_metric_row(row) for row in score_rows]


def merge_rows_with_scores(
    rows: list[dict[str, Any]],
    score_rows: list[dict[str, float | None]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        score_row = score_rows[idx] if idx < len(score_rows) else {}
        merged.append(
            {
                "id": row["id"],
                "source_document": row["source_document"],
                "category": row["category"],
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "chatbot_answer": row["chatbot_answer"],
                "refusal": bool(row["refusal"]),
                "citation_count": int(row["citation_count"]),
                "faithfulness": score_row.get("faithfulness"),
                "answer_relevancy": score_row.get("answer_relevancy"),
                "context_precision": score_row.get("context_precision"),
                "context_recall": score_row.get("context_recall"),
            }
        )
    return merged


def build_summary(
    rows: list[dict[str, Any]],
    dataset_path: Path,
    ragas_llm_model: str,
    ragas_embed_model: str,
    chatbot_model_desc: str,
    errors: dict[str, int],
) -> dict[str, Any]:
    metric_means: dict[str, float | None] = {}
    metric_non_null_counts: dict[str, int] = {}
    for metric in METRIC_COLUMNS:
        values = [
            _coerce_optional_float(row.get(metric))
            for row in rows
            if _coerce_optional_float(row.get(metric)) is not None
        ]
        metric_non_null_counts[metric] = len(values)
        metric_means[metric] = round(sum(values) / len(values), 4) if values else None

    ragas_metric_null_rows = sum(
        1 for row in rows if any(_coerce_optional_float(row.get(metric)) is None for metric in METRIC_COLUMNS)
    )
    refusal_count = sum(1 for row in rows if bool(row.get("refusal")))

    return {
        "dataset_path": str(dataset_path),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_chatbot": chatbot_model_desc,
        "model_ragas_llm": ragas_llm_model,
        "model_ragas_embeddings": ragas_embed_model,
        "total_questions": len(rows),
        "metrics": metric_means,
        "metric_non_null_counts": metric_non_null_counts,
        "errors": {
            "chatbot_answer_errors": int(errors.get("chatbot_answer_errors", 0)),
            "ragas_metric_null_rows": ragas_metric_null_rows,
            "refusal_count": refusal_count,
        },
    }


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _csv_value(row.get(col)) for col in CSV_COLUMNS})


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _import_ragas_runtime() -> dict[str, Any]:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Ragas runtime dependencies are missing. "
            "Install with: pip install -r requirements.txt"
        ) from exc

    return {
        "ChatGoogleGenerativeAI": ChatGoogleGenerativeAI,
        "GoogleGenerativeAIEmbeddings": GoogleGenerativeAIEmbeddings,
        "EvaluationDataset": EvaluationDataset,
        "evaluate": evaluate,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "Faithfulness": Faithfulness,
        "ResponseRelevancy": ResponseRelevancy,
        "LLMContextPrecisionWithReference": LLMContextPrecisionWithReference,
        "LLMContextRecall": LLMContextRecall,
    }


def _evaluate_with_ragas(
    evaluate_fn: Callable[..., Any],
    dataset: Any,
    metrics: list[Any],
    llm: Any,
    embeddings: Any,
) -> Any:
    # Keep this adapter isolated to simplify migration to @experiment later.
    return evaluate_fn(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )


def _extract_score_rows(result: Any, sample_count: int) -> list[dict[str, Any]]:
    raw_scores = getattr(result, "scores", None)
    if isinstance(raw_scores, list):
        rows = [row for row in raw_scores if isinstance(row, dict)]
    else:
        rows = []

    if not rows and hasattr(result, "to_pandas"):
        try:
            frame = result.to_pandas()
            records = frame.to_dict(orient="records")
            rows = [row for row in records if isinstance(row, dict)]
        except Exception:
            rows = []

    normalized: list[dict[str, Any]] = []
    for idx in range(sample_count):
        normalized.append(rows[idx] if idx < len(rows) else {})
    return normalized


def _normalize_metric_row(row: dict[str, Any]) -> dict[str, float | None]:
    normalized: dict[str, float | None] = {}
    for target_metric, aliases in METRIC_ALIASES.items():
        value = None
        for alias in aliases:
            if alias in row:
                value = _coerce_optional_float(row.get(alias))
                break
        normalized[target_metric] = value
    return normalized


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def main() -> None:
    args = parse_args()
    settings = load_settings()
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required")

    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be > 0")
        dataset = dataset[: args.limit]
    if not dataset:
        raise ValueError("Dataset is empty after applying filters")

    ragas_llm_model = args.ragas_llm_model or settings.gen_model
    ragas_embed_model = args.ragas_embed_model or settings.embed_model

    pipeline = RAGPipeline(settings=settings)
    answer_rows, errors = collect_chatbot_outputs(dataset, pipeline)
    samples = build_ragas_samples(answer_rows)
    score_rows = run_ragas(
        samples=samples,
        api_key=settings.gemini_api_key,
        llm_model=ragas_llm_model,
        embed_model=ragas_embed_model,
    )
    merged_rows = merge_rows_with_scores(answer_rows, score_rows)

    summary = build_summary(
        rows=merged_rows,
        dataset_path=dataset_path,
        ragas_llm_model=ragas_llm_model,
        ragas_embed_model=ragas_embed_model,
        chatbot_model_desc=f"RAGPipeline(gen={settings.gen_model}, embed={settings.embed_model})",
        errors=errors,
    )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    write_results_csv(out_csv, merged_rows)
    write_summary_json(out_json, summary)

    print("===== Ragas Summary =====")
    print(f"Total questions: {summary['total_questions']}")
    for metric_name in METRIC_COLUMNS:
        print(f"{metric_name}: {summary['metrics'][metric_name]}")
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
