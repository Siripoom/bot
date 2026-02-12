from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag.config import load_settings
from rag.models import RetrievedChunk
from rag.pipeline import RAGPipeline


st.set_page_config(page_title="Gemini PDF RAG Chatbot", layout="wide")


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()



def _recent_history(messages: list[dict], turns: int) -> list[dict]:
    max_messages = max(1, turns * 2)
    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if msg.get("role") in {"user", "assistant"}
    ]
    return history[-max_messages:]



def _snippet(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."



def _render_citations(citations: list[RetrievedChunk]) -> None:
    if not citations:
        return

    st.caption("แหล่งอ้างอิง")
    for c in citations:
        st.markdown(f"- `{c.file_name}` | หน้า {c.page} | {_snippet(c.text)}")


settings = load_settings()
st.title("RAG Chatbot (Gemini + PDF)")

with st.sidebar:
    st.subheader("System Status")
    st.write(f"GEN model: `{settings.gen_model}`")
    st.write(f"Embedding model: `{settings.embed_model}`")
    st.write(f"Vector DB: `{settings.chroma_dir}`")

    if not settings.gemini_api_key:
        st.error("ไม่พบ GEMINI_API_KEY ในไฟล์ .env")

    pdf_files = sorted(Path(settings.pdf_dir).glob("*.pdf"))
    st.write(f"PDF files: `{len(pdf_files)}`")

    if st.button("Re-index from data/pdfs", use_container_width=True):
        try:
            pipe = get_pipeline()
            with st.spinner("กำลังสร้างดัชนีจาก PDF..."):
                result = pipe.index_pdfs(pdf_dir=str(settings.pdf_dir), reset=True)
            st.success(
                f"Index เสร็จ: files={result['indexed_files']}, chunks={result['indexed_chunks']}"
            )
        except Exception as exc:
            st.error(f"Index failed: {exc}")

    if st.button("Clear session", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

try:
    pipeline = get_pipeline()
except Exception as exc:
    st.error(f"ไม่สามารถเริ่มระบบได้: {exc}")
    st.stop()

stats = pipeline.get_index_stats()
with st.sidebar:
    st.write(f"Indexed docs: `{stats['document_count']}`")
    st.write(f"Indexed chunks: `{stats['chunk_count']}`")

if not pdf_files:
    st.warning("โฟลเดอร์ data/pdfs ยังไม่มีไฟล์ PDF")

if not pipeline.has_index():
    st.warning("ยังไม่มี index กรุณากด Re-index from data/pdfs ก่อนใช้งาน")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_citations(msg.get("citations", []))

query = st.chat_input("พิมพ์คำถามจากเอกสาร PDF...")
if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if not settings.gemini_api_key:
        answer_text = "ไม่พบ GEMINI_API_KEY จึงไม่สามารถตอบคำถามได้"
        citations = []
    elif not pipeline.has_index():
        answer_text = "ยังไม่มี index ของเอกสาร กรุณากด Re-index from data/pdfs ก่อน"
        citations = []
    else:
        try:
            history = _recent_history(st.session_state["messages"][:-1], turns=settings.memory_turns)
            result = pipeline.ask(query=query, history=history)
            answer_text = result.answer_text
            citations = result.citations
        except Exception as exc:
            answer_text = f"เกิดข้อผิดพลาดระหว่างประมวลผล: {exc}"
            citations = []

    with st.chat_message("assistant"):
        st.markdown(answer_text)
        _render_citations(citations)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer_text,
            "citations": citations,
        }
    )
