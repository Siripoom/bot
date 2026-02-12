from __future__ import annotations

import json
from typing import Any

from google import genai

from .config import REFUSAL_MESSAGE
from .models import RAGAnswer, RetrievedChunk


class GeminiAnswerGenerator:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[dict],
    ) -> RAGAnswer:
        if not chunks:
            return RAGAnswer(answer_text=REFUSAL_MESSAGE, citations=[], grounded=False, refusal=True)

        history_text = _format_history(history)
        context_text = "\n".join(
            f"[chunk_id={c.chunk_id}] file={c.file_name} page={c.page}\n{c.text}"
            for c in chunks
        )

        prompt = (
            "คุณคือผู้ช่วยตอบคำถามจากเอกสารเท่านั้น\n"
            "ข้อกำหนด:\n"
            "1) ตอบเป็นภาษาไทยเท่านั้น\n"
            "2) ใช้เฉพาะข้อมูลจาก CONTEXT\n"
            "3) ห้ามเดาหรือเพิ่มข้อมูลจากภายนอก\n"
            "4) ถ้าข้อมูลไม่พอ ให้ตอบข้อความนี้เท่านั้น: "
            f"\"{REFUSAL_MESSAGE}\"\n"
            "5) คำตอบต้องกระชับและชัดเจน\n\n"
            f"CHAT_HISTORY:\n{history_text}\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION:\n{query}\n"
        )

        response = self.client.models.generate_content(model=self.model, contents=prompt)
        answer_text = _response_text(response).strip() or REFUSAL_MESSAGE

        refusal = answer_text == REFUSAL_MESSAGE
        return RAGAnswer(
            answer_text=answer_text,
            citations=chunks,
            grounded=not refusal,
            refusal=refusal,
        )


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(empty)"
    lines: list[str] = []
    for item in history:
        role = item.get("role", "user")
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(empty)"


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
