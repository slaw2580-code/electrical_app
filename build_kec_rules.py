# build_kec_rules.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

BASE = Path(__file__).resolve().parent
PDF_PATH = BASE / "KEC.pdf"         # 네가 올려 둔 KEC 파일 이름
PERSIST_DIR = BASE / "chroma_db"
COLL_NAME = "kec_rules"

load_dotenv(BASE / ".env", override=True)


def extract_pdf_paragraphs(pdf_path: Path):
    """KEC.pdf에서 페이지 번호와 문단 텍스트를 뽑는다."""
    doc = fitz.open(pdf_path)
    chunks = []

    for page_idx, page in enumerate(doc):
        page_no = page_idx + 1
        text = page.get_text("text").strip()
        if not text:
            continue

        # 줄 단위로 잘라서 너무 짧은 것만 버린다
        for line in text.split("\n"):
            line = line.strip()
            if len(line) < 30:
                continue
            # LLM이 바로 근거로 쓸 수 있게 앞에 [pXX]를 붙여준다
            chunks.append((page_no, line))

    return chunks


def build_rules():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"KEC.pdf를 찾을 수 없습니다: {PDF_PATH}")

    print(f"[1] PDF 텍스트 추출: {PDF_PATH}")
    paras = extract_pdf_paragraphs(PDF_PATH)
    print(f"   -> 문단 수: {len(paras)}")

    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    vs = Chroma(
        embedding_function=emb,
        persist_directory=str(PERSIST_DIR),
        collection_name=COLL_NAME,
    )

    docs = []
    for page_no, text in paras:
        docs.append(Document(
            page_content=f"[p{page_no}] {text}",
            metadata={
                "source": "KEC.pdf",
                "page": page_no,
            },
        ))

    print("[2] 벡터 DB(kec_rules 컬렉션)에 저장 중...")
    vs.add_documents(docs)
    print("[3] 완료! 이제 review.py에서 이 규정만 쓸 수 있습니다.")


if __name__ == "__main__":
    build_rules()
