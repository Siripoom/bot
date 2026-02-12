from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REFUSAL_MESSAGE = (
    "ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในเอกสารที่ให้มา "
    "กรุณาระบุคำถามให้เฉพาะเจาะจงขึ้น"
)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gen_model: str = "gemini-2.5-flash"
    embed_model: str = "gemini-embedding-001"
    pdf_dir: Path = Path("data/pdfs")
    chroma_dir: Path = Path("storage/chroma")
    top_k: int = 8
    sim_threshold: float = 0.58
    gate_conf_threshold: float = 0.65
    collection_name: str = "pdf_rag_chunks"
    log_file: Path = Path("logs/app.log")
    memory_turns: int = 6

    def ensure_paths(self) -> None:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)



def load_settings() -> Settings:
    load_dotenv(override=False)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    settings = Settings(
        gemini_api_key=api_key,
        gen_model=os.getenv("GEN_MODEL", "gemini-2.5-flash").strip(),
        embed_model=os.getenv("EMBED_MODEL", "gemini-embedding-001").strip(),
        pdf_dir=Path(os.getenv("PDF_DIR", "data/pdfs")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", "storage/chroma")),
        top_k=int(os.getenv("TOP_K", "8")),
        sim_threshold=float(os.getenv("SIM_THRESHOLD", "0.58")),
        gate_conf_threshold=float(os.getenv("GATE_CONF_THRESHOLD", "0.65")),
        collection_name=os.getenv("CHROMA_COLLECTION", "pdf_rag_chunks").strip(),
        log_file=Path(os.getenv("LOG_FILE", "logs/app.log")),
        memory_turns=int(os.getenv("MEMORY_TURNS", "6")),
    )
    settings.ensure_paths()
    return settings
