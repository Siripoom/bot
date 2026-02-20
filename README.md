# Gemini PDF RAG Chatbot (Streamlit)

RAG chatbot ที่ตอบจากข้อมูลใน PDF เป็นหลัก และมี QA fallback โดยใช้:
- `gemini-2.5-flash` สำหรับ generation และ relevance gate
- `gemini-embedding-001` สำหรับ embeddings
- `Chroma` แบบ local สำหรับ vector database
- `Streamlit` สำหรับ UI

## Features
- ถามคำถามได้อิสระจากชุด PDF
- ตอบภาษาไทยเท่านั้น
- มี 2-stage relevance gate: similarity score + LLM groundedness
- มี `QA fallback` จาก `data/json/qa_prompt.json` เพื่อลด false refusal
- รองรับ `partial answer` เมื่อมีข้อมูลบางส่วน
- จัดรูปแบบคำตอบอัตโนมัติเป็นข้อ/ย่อหน้า/ตารางตามคำถาม

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

# retrieval / gate
TOP_K=12
SIM_THRESHOLD=0.55
GATE_CONF_THRESHOLD=0.65

# qa fallback
QA_JSON_PATH=data/json/qa_prompt.json
QA_COLLECTION=qa_prompt_chunks
QA_TOP_K=5
QA_SIM_THRESHOLD=0.60

# partial answer
PARTIAL_ENABLED=true
PARTIAL_MIN_SCORE=0.68
PARTIAL_MIN_CHUNKS=2

# llm determinism + refusal retry
LLM_TEMPERATURE=0.0
LLM_TOP_P=1.0
GEN_RETRY_ON_REFUSAL=1

# startup self-heal for index
AUTO_HEAL_INDEX=true
AUTO_HEAL_MIN_DOCS=1

# output style: auto | list | paragraph
ANSWER_STYLE_POLICY=auto
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
หมายเหตุ index:
- เมื่อเปิดแอป ระบบจะตรวจสุขภาพ index อัตโนมัติ และ re-index ให้เองถ้าดัชนีว่าง/ไม่ครบตามเกณฑ์
- ปุ่ม `🔄 อัปเดต Index เอกสาร` ใน sidebar เป็นการสั่ง forced rebuild

หมายเหตุฟอนต์:
- หากฟอนต์ Kanit ยังไม่เปลี่ยนทันที ให้หยุดแล้วรัน `streamlit run app.py` ใหม่
- จากนั้นทำ hard refresh ที่เบราว์เซอร์ด้วย `Ctrl+Shift+R`

## Benchmark: Chatbot vs Gemini Direct
รันเปรียบเทียบคำตอบจากคำถามใน `data/json/qa_prompt.json`:

```bash
python3 scripts/benchmark_chatbot_vs_gemini.py
```

ไฟล์ผลลัพธ์:
- `reports/benchmark_results.csv` (รายข้อ)
- `reports/benchmark_summary.json` (สรุปเปอร์เซ็นต์ถูก/ผิด)

ตัวเลือกเพิ่มเติม:
```bash
python3 scripts/benchmark_chatbot_vs_gemini.py --limit 10
python3 scripts/benchmark_chatbot_vs_gemini.py --gemini-model gemini-2.5-flash --judge-model gemini-2.5-flash
python3 scripts/benchmark_chatbot_vs_gemini.py --gemini-context-mode qa_fallback --gemini-context-top-k 3
python3 scripts/benchmark_chatbot_vs_gemini.py --gemini-context-mode pdf --gemini-context-top-k 3
```

หมายเหตุ:
- ค่าเริ่มต้นของ baseline ตอนนี้คือ `--gemini-context-mode pdf` (ดึง context จากไฟล์ PDF ที่ index แล้ว)
- ถ้าต้องการทดสอบแบบไม่เชื่อมชุดข้อมูล ให้ใช้ `--gemini-context-mode none`

## Ragas Evaluation (Chatbot Only)
รันประเมินคุณภาพคำตอบของ chatbot ด้วย Ragas:

```bash
python3 scripts/evaluate_ragas.py
```

ไฟล์ผลลัพธ์:
- `reports/ragas_results.csv` (รายข้อ)
- `reports/ragas_summary.json` (สรุปค่าเฉลี่ย metric)

ตัวเลือกเพิ่มเติม:
```bash
python3 scripts/evaluate_ragas.py --limit 10
python3 scripts/evaluate_ragas.py --ragas-llm-model gemini-2.5-flash --ragas-embed-model gemini-embedding-001
```

## นโยบายการตอบ
ถ้าเอกสารไม่พอสำหรับตอบคำถาม ระบบจะตอบ:

`ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในเอกสารที่ให้มา กรุณาระบุคำถามให้เฉพาะเจาะจงขึ้น`

## Test
```bash
pytest -q
```
