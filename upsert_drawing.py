# upsert_drawing.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# .env 로드
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=str(env_path))

PERSIST_DIR = "chroma_db"
COLL_NAME   = "drawings"
FINAL_JSON  = Path("final_result.json")     # run_yolo_ocr_local.py 산출
SUMMARY_JSON= Path("final_summary.json")    # 규칙기반 요약 판정
DRAWING_ID  = "TEST-001"
PAGE_NO     = 1

# ------------------- [헬퍼 함수: None 방지] -------------------
def safe_str(val, default=""):
    return str(val) if val is not None else default

def safe_float(val, default=-1.0):
    try:
        return float(val)
    except Exception:
        return default

def safe_int(val, default=-1):
    try:
        return int(val)
    except Exception:
        return default

# ------------------- [Document 생성] -------------------
def to_docs(symbols, drawing_id, page=1):
    docs = []
    for s in symbols:
        bbox = s.get("bbox") or [0, 0, 0, 0]
        sx1, sy1, sx2, sy2 = [safe_int(x, 0) for x in bbox]

        # 심볼 문서
        docs.append(Document(
            page_content=(f"[PAGE {page}] [SYMBOL] label='{safe_str(s.get('class'))}' "
                          f"bbox=({sx1},{sy1},{sx2},{sy2}) conf={safe_float(s.get('confidence'),0.0)}"),
            metadata={
                "type": "symbol",
                "label": safe_str(s.get('class')),
                "bbox_str": f"{sx1},{sy1},{sx2},{sy2}",
                "sx1": safe_float(sx1), "sy1": safe_float(sy1),
                "sx2": safe_float(sx2), "sy2": safe_float(sy2),
                "confidence": safe_float(s.get('confidence'), 0.0),
                "drawing_id": drawing_id,
                "page": safe_int(page, 1)
            }
        ))

        # OCR 문서
        for t in s.get("ocr_texts", []):
            txt = (t.get("text") or "").strip()
            if not txt:
                continue
            tb = t.get("bbox") or [0, 0, 0, 0]
            tx1, ty1, tx2, ty2 = [safe_int(x, -1) for x in tb]
            dist = safe_float(t.get("distance"), -1.0)

            docs.append(Document(
                page_content=(f"[PAGE {page}] [OCR NEAR {safe_str(s.get('class'))}] '{txt}' "
                              f"near_bbox=({sx1},{sy1},{sx2},{sy2}) "
                              f"text_bbox=({tx1},{ty1},{tx2},{ty2}) dist={dist}"),
                metadata={
                    "type": "ocr",
                    "label": safe_str(s.get('class')),
                    "near_bbox_str": f"{sx1},{sy1},{sx2},{sy2}",
                    "text_bbox_str": f"{tx1},{ty1},{tx2},{ty2}",
                    "sx1": safe_float(sx1), "sy1": safe_float(sy1),
                    "sx2": safe_float(sx2), "sy2": safe_float(sy2),
                    "tx1": safe_float(tx1), "ty1": safe_float(ty1),
                    "tx2": safe_float(tx2), "ty2": safe_float(ty2),
                    "distance": dist,
                    "text": txt,
                    "drawing_id": drawing_id,
                    "page": safe_int(page, 1)
                }
            ))
    return docs

# ------------------- [메인] -------------------
def main():
    if not FINAL_JSON.exists():
        raise FileNotFoundError(f"final_result.json을 찾을 수 없습니다: {FINAL_JSON.resolve()}")
    symbols = json.loads(FINAL_JSON.read_text(encoding="utf-8"))

    # 규칙기반 요약 판정 로드
    judge = {}
    if SUMMARY_JSON.exists():
        judge = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(embedding_function=embeddings,
                persist_directory=PERSIST_DIR,
                collection_name=COLL_NAME)

    docs = to_docs(symbols, DRAWING_ID, PAGE_NO)

    # 키-값 형태 문서 추가
    if judge:
        pc = (
            f"[PAGE {PAGE_NO}] [DRAWING-KEYS]\n"
            f"종류={safe_str(judge.get('extracted',{}).get('kind'))}\n"
            f"정격전압={safe_str(judge.get('extracted',{}).get('rated_voltage'))}\n"
            f"정격전류={safe_str(judge.get('extracted',{}).get('rated_current'))}\n"
            f"정격차단전류={safe_str(judge.get('extracted',{}).get('breaking_capacity'))}\n"
            f"ALL_PRESENT={judge.get('all_present')} "
            f"MISSING={','.join(judge.get('missing', []))}"
        )
        docs.append(Document(
            page_content=pc,
            metadata={
                "type": "keys",
                "drawing_id": DRAWING_ID,

                "page": safe_int(PAGE_NO, 1),
                "kind": safe_str(judge.get('extracted', {}).get('kind')),
                "rated_voltage": safe_str(judge.get('extracted', {}).get('rated_voltage')),
                "rated_current": safe_str(judge.get('extracted', {}).get('rated_current')),
                "breaking_capacity": safe_str(judge.get('extracted', {}).get('breaking_capacity')),
                "all_present": bool(judge.get('all_present'))
            }
        ))

    if docs:
        vs.add_documents(docs)
    print(f"[완료] 업서트 문서 수: {len(docs)} | drawing_id={DRAWING_ID} | collection={COLL_NAME}")

if __name__ == "__main__":
    main()

