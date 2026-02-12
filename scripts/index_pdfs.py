#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline import RAGPipeline



def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDF files into Chroma")
    parser.add_argument("--pdf-dir", type=str, default=None, help="Directory containing PDF files")
    parser.add_argument("--reset", action="store_true", help="Reset collection before indexing")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    result = pipeline.index_pdfs(pdf_dir=args.pdf_dir, reset=args.reset)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
