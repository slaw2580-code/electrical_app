from __future__ import annotations
from rules_engine import check_all_rules
import sys
import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from review import chat_about_drawing

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# === 기본 설정 ===
BASE = Path(__file__).resolve().parent

# review.py import를 위해 경로 추가
sys.path.append(str(BASE))
from review import review_all_rules  # 위에서 sys.path 추가한 뒤에 import

YOLO_WEIGHTS = BASE / "data" / "models" / "substation_yolo_v83" / "weights" / "best.pt"

RULES_DIR = BASE / "chroma_db"
DRAW_COLL = "drawings"

load_dotenv(BASE / ".env", override=True)


def _run_yolo_ocr(image_path: Path, out_dir: Path):
    """
    run_yolo_ocr_local.py 를 서브프로세스로 실행해서
    final_result.json / final_summary.json 을 out_dir 안에 생성한다.
    """
    out_json = out_dir / "final_result.json"
    summary_json = out_dir / "final_summary.json"

    cmd = [
        sys.executable,
        str(BASE / "run_yolo_ocr_local.py"),
        "--image", str(image_path),
        "--weights", str(YOLO_WEIGHTS),
        "--out", str(out_json),
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    cp = subprocess.run(
        cmd,
        cwd=str(BASE),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )

    if cp.returncode != 0:
        raise RuntimeError(
            "YOLO/OCR 실패\n"
            f"STDOUT:\n{cp.stdout}\n\n"
            f"STDERR:\n{cp.stderr}"
        )

    symbols = json.loads(out_json.read_text(encoding="utf-8"))
    return out_json, summary_json, symbols


def _to_docs(
    symbols: list,
    drawing_id: str,
    page: int = 1,
    judge: dict | None = None,
) -> List[Document]:
    """
    YOLO+OCR 결과(symbols, judge)를 Chroma Document 리스트로 변환.
    """
    docs: List[Document] = []

    def s_int(x, d=0):
        try:
            return int(x)
        except Exception:
            return d

    def s_flt(x, d=0.0):
        try:
            return float(x)
        except Exception:
            return d

    # 1) 심볼
    for s in symbols:
        bbox = s.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [s_int(v) for v in bbox]

        docs.append(Document(
            page_content=(
                f"[PAGE {page}] [SYMBOL] label='{s.get('class', '')}' "
                f"bbox=({x1},{y1},{x2},{y2}) conf={s_flt(s.get('confidence'))}"
            ),
            metadata={
                "type": "symbol",
                "label": s.get("class", ""),
                "bbox_str": f"{x1},{y1},{x2},{y2}",
                "sx1": x1, "sy1": y1, "sx2": x2, "sy2": y2,
                "confidence": s_flt(s.get('confidence')),
                "drawing_id": drawing_id,
                "page": page,
            }
        ))

        # 2) 심볼 근처 OCR 텍스트
        for t in s.get("ocr_texts", []):
            txt = (t.get("text") or "").strip()
            if not txt:
                continue
            tb = t.get("bbox") or [0, 0, 0, 0]
            tx1, ty1, tx2, ty2 = [s_int(v, -1) for v in tb]

            docs.append(Document(
                page_content=(
                    f"[PAGE {page}] [OCR NEAR {s.get('class','')}] '{txt}' "
                    f"near_bbox=({x1},{y1},{x2},{y2}) "
                    f"text_bbox=({tx1},{ty1},{tx2},{ty2})"
                ),
                metadata={
                    "type": "ocr",
                    "label": s.get("class", ""),
                    "near_bbox_str": f"{x1},{y1},{x2},{y2}",
                    "text_bbox_str": f"{tx1},{ty1},{tx2},{ty2}",
                    "sx1": x1, "sy1": y1, "sx2": x2, "sy2": y2,
                    "tx1": tx1, "ty1": ty1, "tx2": tx2, "ty2": ty2,
                    "text": txt,
                    "drawing_id": drawing_id,
                    "page": page,
                }
            ))

    # 3) final_summary.json 정보도 문서로 넣기 (있으면)
    if judge:
        ex = judge.get("extracted", {})
        pc = (
            f"[PAGE {page}] [DRAWING-KEYS]\n"
            f"종류={ex.get('kind') or ex.get('종류', '')}\n"
            f"정격전압={ex.get('rated_voltage') or ex.get('정격전압', '')}\n"
            f"정격전류={ex.get('rated_current') or ex.get('정격전류', '')}\n"
            f"정격차단전류={ex.get('breaking_capacity') or ex.get('정격차단전류', '')}"
        )
        docs.append(Document(
            page_content=pc,
            metadata={
                "type": "keys",
                "drawing_id": drawing_id,
                "page": page,
            }
        ))

    return docs


def _upsert_drawings(symbols: list, judge: dict, drawing_id: str):
    """
    도면 1장을 drawings 컬렉션에 업서트.
    """
    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    vs = Chroma(
        embedding_function=emb,
        persist_directory=str(RULES_DIR),
        collection_name=DRAW_COLL,
    )

    docs = _to_docs(symbols, drawing_id, page=1, judge=judge)
    if docs:
        vs.add_documents(docs)

def chat_answer(question: str, drawing_id: str) -> str:
    """
    Streamlit에서 채팅 질문이 들어왔을 때 호출할 함수.
    review.py의 chat_about_drawing()을 그대로 감싼다.
    """
    return chat_about_drawing(question, drawing_id)

def analyze_drawing(input_path: str) -> Dict[str, Any]:
    """
    Streamlit 에서 호출하는 진입점.

    1) YOLO+OCR 실행
    2) 도면 정보 Vector DB(drawings)에 업서트
    3) 규정 DB + 도면 정보를 바탕으로 review_all_rules() 호출
    4) rules_engine(4·5·6번 하드 규칙)까지 함께 검사
    5) 위반 요약/목록 반환
    """
    image = Path(input_path)

    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"YOLO 가중치를 찾을 수 없습니다: {YOLO_WEIGHTS}")

    # ── 1~4단계: YOLO + LLM 리뷰 ──────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # 1) YOLO + OCR 실행
        result_path, summary_path, symbols = _run_yolo_ocr(image, td_path)

        # 2) final_summary.json 로드 (있으면 judge로 사용)
        judge: dict = {}
        if summary_path.exists():
            try:
                judge = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                judge = {}

        # 3) 도면 ID 만들고 Vector DB에 업서트
        drawing_id = f"DEMO-{image.stem}"
        _upsert_drawings(symbols, judge, drawing_id)

        # 4) LLM으로 전체 규정 위반 리뷰 (KEC.pdf 기반)
        final_judge = review_all_rules(drawing_id)

    # ── 여기부터는 temp 폴더 밖: symbols, judge, final_judge 사용 ──

    # 4-1) LLM이 찾은 위반 목록
    violations: List[dict] = list(final_judge.get("violations", []))

    # 4-2) 4·5·6번 좌표 기반 하드 규칙 검사
    # meta에는 전압 같은 부가 정보가 들어갈 자리 (지금은 summary_json의 extracted 사용)
    meta = judge.get("extracted", {}) if isinstance(judge, dict) else {}
    hard_violations = check_all_rules(symbols, meta)

    # LLM 위반 + 하드 규칙 위반 합치기
    violations.extend(hard_violations)

    # 5) Streamlit 표시용 summary 계산
    summary = {
        "total_violations": len(violations),
        "high": sum(1 for v in violations if v.get("severity") == "HIGH"),
        "medium": sum(1 for v in violations if v.get("severity") == "MEDIUM"),
        "low": sum(1 for v in violations if v.get("severity") == "LOW"),
    }

    return {"summary": summary, "violations": violations}
