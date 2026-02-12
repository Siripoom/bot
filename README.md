# Gemini PDF RAG Chatbot (Streamlit)

RAG chatbot ที่ตอบจากข้อมูลใน PDF เท่านั้น โดยใช้:
- `gemini-2.5-flash` สำหรับ generation และ relevance gate
- `gemini-embedding-001` สำหรับ embeddings
- `Chroma` แบบ local สำหรับ vector database
- `Streamlit` สำหรับ UI

## Features
- ถามคำถามได้อิสระจากชุด PDF
- ตอบภาษาไทยเท่านั้น
- มี 2-stage relevance gate: similarity score + LLM groundedness
- ถ้าหลักฐานไม่พอจะปฏิเสธด้วยข้อความมาตรฐาน
- แสดงแหล่งอ้างอิง (ไฟล์ | หน้า | snippet)

## โครงสร้าง
- `app.py` - Streamlit UI
- `rag/` - core modules
- `scripts/index_pdfs.py` - batch indexer
- `tests/` - unit tests

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

ใส่ API key ใน `.env`:
```env
GEMINI_API_KEY=...
```

## การใช้งาน
1. วางไฟล์ PDF ใน `data/pdfs/`
2. สร้างดัชนี:
```bash
python scripts/index_pdfs.py --pdf-dir data/pdfs --reset
```
3. รัน UI:
```bash
streamlit run app.py
```

## นโยบายการตอบ
ถ้าเอกสารไม่พอสำหรับตอบคำถาม ระบบจะตอบ:

`ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในเอกสารที่ให้มา กรุณาระบุคำถามให้เฉพาะเจาะจงขึ้น`

## Test
```bash
pytest -q
```
