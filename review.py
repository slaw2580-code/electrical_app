# review.py
# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# === 기본 설정 ===
BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env", override=True)

PERSIST_DIR = BASE / "chroma_db"
RULES_COLL  = "kec_rules"
DRAW_COLL   = "drawings"

MODEL        = os.getenv("OPENAI_MODEL", "gpt-4o")
SUMMARY_JSON = BASE / "final_summary.json"   # run_yolo_ocr_local.py에서 만든 요약


def review_all_rules(drawing_id: str) -> dict:
    """
    이 도면(drawing_id)에 대해,
    'KEC.pdf에서 추출한 규정(kec_rules 컬렉션)' 과
    '도면 벡터 DB(drawings 컬렉션)'만을 사용해서
    위반/확인 필요 항목을 찾아낸다.

    반환 예:
    {
      "violations": [
        {
          "rule_id": "KEC-ACB-001",
          "desc": "[p23] ... 규정을 근거로 ACB 정격전압 표시가 누락되어 있음",
          "severity": "HIGH"
        },
        ...
      ]
    }
    """

    emb = OpenAIEmbeddings()

    # 1) 규정 벡터스토어 (KEC.pdf만 들어 있음)
    rules_vs = Chroma(
        embedding_function=emb,
        persist_directory=str(PERSIST_DIR),
        collection_name=RULES_COLL,
    )

    # 2) 도면 벡터스토어
    draw_vs = Chroma(
        embedding_function=emb,
        persist_directory=str(PERSIST_DIR),
        collection_name=DRAW_COLL,
    )

    # 3) 규정 텍스트 검색 (KEC.pdf 에서만)
    rules_query = (
        "수변전 설비, ACB, MCCB, 차단기, CT, PT, MOF, LA, PF, GND, ZCT, "
        "단선결선도, 보호계전, 정격전압, 정격전류, 정격차단전류, 접지, 절연거리"
    )
    rules_docs = rules_vs.similarity_search(
        rules_query,
        k=15,
        filter={"source": "KEC.pdf"},   # ★ KEC.pdf에서 온 문단만 사용
    )
    rules_text = "\n".join(f"- {d.page_content}" for d in rules_docs)

    # 4) 도면 텍스트/심볼 정보 (해당 drawing_id만)
    draw_query = (
        f"{drawing_id} 도면에 포함된 모든 심볼과 텍스트, "
        "ACB, MCCB, LA, CT, PT, MOF, PF, GND, "
        "정격전압, 정격전류, 정격차단전류, 계통전압, 변압기 용량"
    )
    draw_docs = draw_vs.similarity_search(
        draw_query,
        k=25,
        filter={"drawing_id": drawing_id},
    )
    drawing_text = "\n".join(f"- {d.page_content}" for d in draw_docs)

    # 5) YOLO+OCR 요약 결과 (있으면 보조 정보로)
    summary_json = {}
    if SUMMARY_JSON.exists():
        try:
            summary_json = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        except Exception:
            summary_json = {}

    # 6) LLM 호출
    llm = ChatOpenAI(model=MODEL, temperature=0)

    prompt = f"""
너는 전기설비 KEC/KS 규정 전문가이다.

아래 정보는 오직 두 가지 출처에서만 왔다:
- KEC.pdf에서 추출한 규정 텍스트 (각 줄은 [p페이지번호] 로 시작함)
- 단 한 장의 도면에서 추출한 심볼/텍스트 정보

1) YOLO+OCR 요약(JSON)
{json.dumps(summary_json, ensure_ascii=False)}

2) [KEC 규정 발췌] (KEC.pdf에서 추출, [pXX]는 실제 페이지 번호)
{rules_text}

3) [도면 텍스트/심볼 정보]
{drawing_text}

[해야 할 일]

- 위 규정(2)과 도면 정보(3)를 비교하여,
  KEC/KS 규정을 위반했거나, 최소한 사람이 반드시 확인해야 할 사항을 찾아라.
- 각 항목에 대해:
  - 어떤 기기/설비/표시와 관련된 문제인지
  - KEC.pdf의 어느 문장을 근거로 삼는지 (반드시 [pXX] 형태의 구절을 desc 안에 포함)
  - 왜 위험하거나 부적합한지
  를 한국어로 설명하라.

[출력 형식]

반드시 아래 JSON 형식 *만* 출력하라. JSON 바깥에 다른 글자는 쓰지 마라.

{{
  "violations": [
    {{
      "rule_id": "KEC-XXXX-001",  // 규정 번호 또는 임의 ID. 모르면 "KEC-UNSPEC" 사용
      "desc": "[p23] ... 와 같은 규정을 근거로, ACB 정격전압 표시가 누락되어 있어 규정 위반이다",
      "severity": "HIGH"          // HIGH / MEDIUM / LOW 중 하나
    }}
  ]
}}

[추가 규칙]

- 규정 번호를 정확히 모르면 rule_id는 "KEC-UNSPEC" 으로 둔다.
- desc 안에는 반드시 KEC.pdf에서 온 규정 문장을
  최소 한 번 이상 그대로 포함해야 한다. (예: "[p23] 차단기의 정격전압은 ...")
- 위반이 명확하지 않고 정보가 부족할 경우,
  "위반 여부는 불명확하지만, 다음 규정을 근거로 추가 확인이 필요하다"는 식으로
  LOW 또는 MEDIUM 등급으로 넣어라.
- violations 리스트가 비어 있으면 안 된다. 최소 1개 이상 생성하라.
"""

    resp = llm.invoke(prompt)
    text = resp.content

    try:
        data = json.loads(text)
    except Exception:
        data = {"violations": []}

    if "violations" not in data or not isinstance(data["violations"], list):
        data["violations"] = []

    # 형식 정리
    norm_violations = []
    for v in data["violations"]:
        if not isinstance(v, dict):
            continue
        norm_violations.append({
            "rule_id": str(v.get("rule_id", "KEC-UNSPEC")),
            "desc": str(v.get("desc", "")),
            "severity": str(v.get("severity", "MEDIUM")).upper(),
        })

    # 혹시 LLM이 또 0개 주면, 더미 하나라도 만들어 준다.
    if not norm_violations:
        norm_violations.append({
            "rule_id": "KEC-NO-RESULT",
            "desc": "KEC.pdf와 도면 정보를 비교했으나 명확한 위반을 찾지 못했습니다. "
                    "다만 차단기 정격값, 접지, 보호계전 설정 등이 KEC 규정을 만족하는지 "
                    "사람이 직접 확인해야 합니다.",
            "severity": "LOW",
        })

    data["violations"] = norm_violations
    return data
def chat_about_drawing(question: str, drawing_id: str) -> str:
    """
    사용자가 도면/위반에 대해 질문하면,
    KEC 규정 + 도면 벡터DB + 위반 규칙을 기반으로 자연어로 답해주는 함수.
    """

    emb = OpenAIEmbeddings()

    rules_vs = Chroma(
        embedding_function=emb,
        persist_directory=str(PERSIST_DIR),
        collection_name=RULES_COLL,
    )

    draw_vs = Chroma(
        embedding_function=emb,
        persist_directory=str(PERSIST_DIR),
        collection_name=DRAW_COLL,
    )

    # 1) 규정에서 질문과 관련된 내용 찾기
    rules_docs = rules_vs.similarity_search(question, k=5)
    rules_text = "\n".join(f"- {d.page_content}" for d in rules_docs)

    # 2) 해당 도면에서 질문과 관련된 내용 찾기
    draw_docs = draw_vs.similarity_search(
        question,
        k=10,
        filter={"drawing_id": drawing_id},
    )
    drawing_text = "\n".join(f"- {d.page_content}" for d in draw_docs)

    llm = ChatOpenAI(model=MODEL, temperature=0)

    prompt = f"""
너는 전기설비 KEC/KS 규정과 수변전 단선결선도에 대해 설명해주는 조언자이다.

[질문]
{question}

[KEC/KS 규정 관련 텍스트]
{rules_text}

[이 도면(drawing_id={drawing_id})에서 추출한 정보]
{drawing_text}

요구사항:
- 반드시 한국어로 답변한다.
- 규정에 근거해서 설명하되, 너무 길게 소설 쓰지 말고 핵심 위주로 설명한다.
- 사용자가 이해하기 쉽게 2~4개 bullet으로 정리해도 좋다.
"""

    resp = llm.invoke(prompt)
    return resp.content


if __name__ == "__main__":
    print(review_all_rules("TEST-001"))
