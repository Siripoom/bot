from __future__ import annotations

import json
import time
from typing import Any

from google import genai


class DirectGeminiClient:
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

    def answer_question(self, question: str, context_chunks: list[str] | None = None) -> str:
        prompt = build_direct_baseline_prompt(question, context_chunks=context_chunks)
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": 0, "top_p": 1},
                )
                return _response_text(response).strip()
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = self.initial_backoff_sec * (2 ** (attempt - 1))
                if delay > 0:
                    time.sleep(delay)
        raise RuntimeError(f"direct_gemini_failed: {last_error}") from last_error


def build_direct_baseline_prompt(question: str, context_chunks: list[str] | None = None) -> str:
    clean_question = question.strip()
    context_text = _format_context_chunks(context_chunks or [])
    if context_text:
        context_instruction = (
            "คุณมี CONTEXT จากชุดข้อมูลองค์กร ให้ใช้ข้อมูลจาก CONTEXT เป็นหลัก\n"
            "ถ้า CONTEXT ไม่พอ ค่อยตอบเท่าที่ทราบโดยไม่แต่งข้อมูล\n\n"
            f"CONTEXT:\n{context_text}\n\n"
        )
    else:
        context_instruction = ""

    return (
        "คุณเป็นผู้ช่วยตอบคำถามทั่วไป\n"
        "ข้อกำหนด:\n"
        "1) ตอบเป็นภาษาไทยเท่านั้น\n"
        "2) ตอบให้กระชับและตรงคำถาม\n"
        "3) ห้ามใส่แหล่งอ้างอิง ลิงก์ หรือ citation\n"
        "4) ถ้าไม่แน่ใจ ให้ตอบเท่าที่รู้โดยไม่แต่งข้อมูลเกินจริง\n\n"
        f"{context_instruction}"
        f"คำถาม:\n{clean_question}"
    )


def _format_context_chunks(chunks: list[str]) -> str:
    normalized = [" ".join(str(chunk).split()) for chunk in chunks if str(chunk).strip()]
    if not normalized:
        return ""
    lines = [f"- {chunk}" for chunk in normalized[:8]]
    return "\n".join(lines)


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
