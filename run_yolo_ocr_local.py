

import os, json, uuid, time, math, argparse, re
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import requests
from dotenv import load_dotenv
import networkx as nx
from ultralytics import YOLO


# =========================
# ── 토폴로지(배선 그래프) 추출
CANNY1 = 80
CANNY2 = 160
HOUGH_THRESH = 60
HOUGH_MIN_LEN = 30
HOUGH_MAX_GAP = 5
MERGE_TOL = 6            # 배선 끝점 병합 거리(px)

# ── 토폴로지 기반 매칭
MAX_TEXT_WIRE_DIST = 10  # 텍스트 중심이 배선(선분)까지 ≤ 이 거리이면 그 회로(컴포넌트)로 귀속
SPATIAL_TIE = 80         # 같은 회로라도 이 이상 멀면 버림(타이브레이커)
TOPK_PER_SYMBOL = 5

# ── 로컬 윈도우 하드게이트(토폴로지 실패 시 백업)
LOCAL_PAD_X = 30         # 심볼 bbox 좌우 패딩
LOCAL_PAD_Y = 24         # 심볼 bbox 상하 패딩
BASE_MAX_DIST = 70       # 전역 거리 상한 (백업 모드)
HARD_MAX = 60            # 절대 거리 상한 (백업 모드)
DIST_PER_SIZE = 0.7
MIN_ALIGN = 0.30
ALIGN_BONUS = 0.8
MIN_EDGE_W = 0.20
REL_MARGIN = 1.07

CLASS_RADIUS_MUL = {"ACB":0.7,"MCCB":0.7,"VCB":0.7,"ELB":0.8,"CT":0.6,"VT":0.6,"TR":0.9,"PF":0.8,"LA":0.7,"MOF":0.8,"PT":0.7}

# ── 클래스별 허용/금지 토큰(멀리 있는 다른 장치명 텍스트 컷)
CLASS_ALLOW = {
    "LA":  ["LA", "KV", "KA", "DISC", "W/DISC"],
    "PF":  ["PF", "KV", "KA", "FUSE", "A"],
    "CT":  ["CT", "/5A", "A", "CL", "N>"],
    "VT":  ["VT", "PT", "V", "KV", "VA"],
    "PT":  ["PT", "V", "KV", "VA", "/"],
    "VCB": ["VCB", "KV", "A", "KA", "MVA"],
    "CB":  ["VCB", "CB", "KV", "A", "KA"],
    "MOF": ["MOF", "PT", "CT", "V", "A"],
}
CLASS_DENY = {
    "LA":  ["MOF","PT","PF","CT","VCB","CB"],
    "PF":  ["MOF","PT","CT","VCB","CB"],
    "CT":  ["MOF","PT","PF","VCB","CB"],
    "PT":  ["MOF","PF","CT","VCB","CB"],
    "VCB": ["MOF","PT","PF","CT"],
    "CB":  ["MOF","PT","PF","CT"],
    "MOF": ["VCB","CB","LA","PF"],
}

# =========================
# 유틸
# =========================
def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (0.5*(x1+x2), 0.5*(y1+y2))

def _size(b):
    x1,y1,x2,y2 = b
    return max(1.0, (x2-x1) + (y2-y1))

def _dist(c1, c2):
    dx, dy = (c1[0]-c2[0]), (c1[1]-c2[1])
    return (dx*dx + dy*dy) ** 0.5

def _h_align(sym_box, txt_box):
    sy = 0.5*(sym_box[1]+sym_box[3]); ty = 0.5*(txt_box[1]+txt_box[3])
    scale = max(sym_box[3]-sym_box[1], txt_box[3]-txt_box[1], 1.0)
    return max(0.0, 1.0 - abs(sy-ty)/(2.0*scale))

def _v_align(sym_box, txt_box):
    sx = 0.5*(sym_box[0]+sym_box[2]); tx = 0.5*(txt_box[0]+txt_box[2])
    scale = max(sym_box[2]-sym_box[0], txt_box[2]-txt_box[0], 1.0)
    return max(0.0, 1.0 - abs(sx-tx)/(2.0*scale))

# =========================
# 배선(전선/리더선) 추출 → 컴포넌트 그래프
# =========================
def _pt_seg_dist(px, py, x1, y1, x2, y2):
    # 점-선분 거리
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    denom = vx*vx + vy*vy
    if denom <= 1e-6:
        return math.hypot(px-x1, py-y1)
    t = max(0.0, min(1.0, (wx*vx + wy*vy) / denom))
    cx, cy = x1 + t*vx, y1 + t*vy
    return math.hypot(px-cx, py-cy)

def _bbox_intersects_segment(b, x1, y1, x2, y2, pad=2):
    # bbox와 선분 교차/근접 여부(패딩)
    bx1, by1, bx2, by2 = b
    bx1 -= pad; by1 -= pad; bx2 += pad; by2 += pad
    # 박스 안에 선분 끝점이 들어오면 true
    if (bx1 <= x1 <= bx2 and by1 <= y1 <= by2) or (bx1 <= x2 <= bx2 and by1 <= y2 <= by2):
        return True
    # 박스 중심과 선분 거리로 근접 판정
    cx, cy = (bx1+bx2)/2, (by1+by2)/2
    w, h = (bx2-bx1), (by2-by1)
    return _pt_seg_dist(cx, cy, x1, y1, x2, y2) <= 0.5*max(w, h)

def _build_wire_components(image_path: Path):
    """
    이미지에서 배선/리더선을 추출해 endpoint 병합 그래프를 만든다.
    거의 수평/수직(±10도) 선분만 채택.
    반환: segs(list[(x1,y1,x2,y2)]), seg_index→comp_id dict, point-graph WG
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], {}, nx.Graph()
    edges = cv2.Canny(img, CANNY1, CANNY2)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=HOUGH_THRESH,
                            minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP)
    segs = []
    if lines is not None:
        for l in lines[:,0,:]:
            x1,y1,x2,y2 = map(int, l)
            ang = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
            if (ang < 10) or (ang > 80):  # 0±10° or 90±10°
                segs.append((x1,y1,x2,y2))

    # 끝점 병합 그래프
    WG = nx.Graph()
    def _near(p,q,t=MERGE_TOL): return math.hypot(p[0]-q[0], p[1]-q[1]) <= t
    pts = []
    for (x1,y1,x2,y2) in segs:
        p1 = (x1,y1); p2 = (x2,y2)
        f1 = next((i for i,p in enumerate(pts) if _near(p,p1)), None)
        f2 = next((i for i,p in enumerate(pts) if _near(p,p2)), None)
        if f1 is None: pts.append(p1); f1 = len(pts)-1
        if f2 is None: pts.append(p2); f2 = len(pts)-1
        WG.add_node(f1, xy=pts[f1]); WG.add_node(f2, xy=pts[f2])
        WG.add_edge(f1, f2)

    comp = {}
    if WG.number_of_nodes() > 0:
        comps = list(nx.connected_components(WG))
        # 컴포넌트 ID 할당(양 끝점이 동일 컴포넌트에 있으면 그 선분은 그 컴포넌트 소속)
        for cid, nodes in enumerate(comps):
            nodes = list(nodes); node_set = set(nodes)
            for idx,(x1,y1,x2,y2) in enumerate(segs):
                # 가장 가까운 노드 찾기
                def _closest_node(x,y):
                    best = None; bd = 1e9
                    for n in nodes:
                        nx_, ny_ = WG.nodes[n]['xy']
                        d = (nx_-x)**2 + (ny_-y)**2
                        if d < bd: bd = d; best = n
                    return best
                n1 = _closest_node(x1,y1); n2 = _closest_node(x2,y2)
                if (n1 in node_set) and (n2 in node_set):
                    comp[idx] = cid
    return segs, comp, WG

# =========================
# 토폴로지 우선 매칭 + 로컬 윈도우 백업
# =========================
def _class_token_ok(text, cls):
    t  = (text or "").upper()
    c  = (cls or "").upper()
    allow = CLASS_ALLOW.get(c, ["V","KV","A","KA","HZ","P"])
    deny  = CLASS_DENY.get(c, [])
    if any(d in t for d in deny):
        return False
    return any(a in t for a in allow)

def _inside_local_window(sym_box, txt_box):
    sx1,sy1,sx2,sy2 = sym_box
    cx, cy = _center(txt_box)
    return (sx1-LOCAL_PAD_X) <= cx <= (sx2+LOCAL_PAD_X) and (sy1-LOCAL_PAD_Y) <= cy <= (sy2+LOCAL_PAD_Y)

def _adaptive_radius(sym_class, sym_box, cli_max):
    base = min(max(cli_max, 40), BASE_MAX_DIST)
    rad  = base + DIST_PER_SIZE * _size(sym_box)
    mul  = CLASS_RADIUS_MUL.get(str(sym_class).upper(), 1.0)
    return max(30.0, min(rad * mul, min(BASE_MAX_DIST, HARD_MAX)))

def match_symbols_with_texts_graph_TOPOFIRST(detected_symbols, ocr_results, image_path: Path,
                                             cli_max_link=120):
    """
    1) 이미지에서 배선 그래프 컴포넌트 생성
    2) '같은 회로' 텍스트만 후보(1순위)
    3) 회로가 불명확하면 로컬 윈도우 하드게이트 + 거리/정렬로 백업
    """
    segs, comp_map, WG = _build_wire_components(image_path)

    # ── 심볼 → comp 집합
    sym_comp_sets = []
    for s in detected_symbols:
        comps = set()
        for idx,(x1,y1,x2,y2) in enumerate(segs):
            if _bbox_intersects_segment(s["bbox"], x1,y1,x2,y2, pad=2):
                cid = comp_map.get(idx)
                if cid is not None:
                    comps.add(cid)
        sym_comp_sets.append(comps)

    # ── 텍스트 → comp (배선에 가까운 선분의 comp)
    txt_comp = []
    for t in ocr_results:
        cx, cy = _center(t["bbox"])
        best_d = 1e9; best_c = None
        for idx,(x1,y1,x2,y2) in enumerate(segs):
            d = _pt_seg_dist(cx, cy, x1,y1,x2,y2)
            if d < best_d:
                best_d = d; best_c = comp_map.get(idx)
        txt_comp.append(best_c if best_d <= MAX_TEXT_WIRE_DIST else None)

    # ── 후보 선택
    for i, s in enumerate(detected_symbols):
        sym_cls = s.get("class","")
        cand = []

        # [A] 같은 회로 텍스트
        if sym_comp_sets[i]:
            scx, scy = _center(s["bbox"])
            for j, t in enumerate(ocr_results):
                c = txt_comp[j]
                if (c is None) or (c not in sym_comp_sets[i]):  # 다른 회로면 제외
                    continue
                if not _class_token_ok(t.get("text"), sym_cls):
                    continue
                tcx, tcy = _center(t["bbox"])
                d = math.hypot(scx-tcx, scy-tcy)
                if d > SPATIAL_TIE:
                    continue
                cand.append((j, d, 1.0/(1.0+d)))  # 같은 회로면 weight는 거리 타이브레이커만

        # [B] 회로가 비어 있거나 후보가 0이면 → 로컬 윈도우 백업
        if not cand:
            rad = _adaptive_radius(sym_cls, s["bbox"], cli_max_link)
            for j, t in enumerate(ocr_results):
                txt = (t.get("text") or "").strip()
                if not txt or not _class_token_ok(txt, sym_cls):
                    continue
                if not _inside_local_window(s["bbox"], t["bbox"]):
                    continue
                ha = _h_align(s["bbox"], t["bbox"]); va = _v_align(s["bbox"], t["bbox"])
                if max(ha,va) < MIN_ALIGN:
                    continue
                d = _dist(_center(s["bbox"]), _center(t["bbox"]))
                if d > rad:
                    continue
                decay = math.exp(-d / max(1.0, rad*0.6))
                w = decay * (1.0 + ALIGN_BONUS*max(ha, va))
                if w < MIN_EDGE_W:
                    continue
                cand.append((j, d, w))

        # 정리 + topK
        cand.sort(key=lambda x: (-x[2], x[1]))  # weight 내림차순, 거리 오름차순
        chosen = []
        for j, d, w in cand[:TOPK_PER_SYMBOL]:
            t = ocr_results[j]
            chosen.append({
                "text": (t.get("text") or "").strip(),
                "bbox": t["bbox"],
                "distance": round(float(d), 2),
                "weight": round(float(w), 4)
            })
        s["ocr_texts"] = chosen

    return detected_symbols

# =========================
# 심볼별 속성 파싱
# =========================
VOLTAGE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(k?V)\b", re.I)  # 22kV, 380V
CURRENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*A\b", re.I)      # 630A
ICAP_RE    = re.compile(r"\b(\d+(?:\.\d+)?)\s*kA\b", re.I)     # 5kA
POLES_RE   = re.compile(r"\b(\d)\s*P\b", re.I)                 # 3P, 4P
FREQ_RE    = re.compile(r"\b(\d+(?:\.\d+)?)\s*Hz\b", re.I)     # 60Hz
KIND_RE    = re.compile(r"\b(ACB|MCCB|VCB|GCB|ELB|VC)\b", re.I)

def _norm(text: str) -> str:
    return (text or "").replace("4P600V","4P 600V").replace("P600V","P 600V")

def parse_attrs(text: str) -> dict:
    t = _norm(text)
    out = {}
    if m := KIND_RE.search(t):     out["kind"] = m.group(1).upper()
    if m := VOLTAGE_RE.search(t):  out["rated_voltage"] = f"{m.group(1)}{m.group(2).upper()}"
    if m := CURRENT_RE.search(t):  out["rated_current"] = f"{m.group(1)}A"
    if m := ICAP_RE.search(t):     out["breaking_capacity"] = f"{m.group(1)}kA"
    if m := POLES_RE.search(t):    out["poles"] = f"{m.group(1)}P"
    if m := FREQ_RE.search(t):     out["frequency"] = f"{m.group(1)}Hz"
    return out

def attach_symbol_attributes(detected_symbols: list) -> list:
    for s in detected_symbols:
        texts = s.get("ocr_texts", [])
        per_text = []
        merged = {}
        for t in texts:
            txt = (t.get("text") or "").strip()
            if not txt:
                continue
            parsed = parse_attrs(txt)
            per_text.append({"text": txt, "parsed": parsed})
            for k, v in parsed.items():
                merged.setdefault(k, v)
        s["text_parsed"] = per_text
        s["attributes"]  = merged
    return detected_symbols

# =========================
# 환경변수
# =========================
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=str(env_path))
OCR_SECRET_KEY = os.getenv("CLOVA_OCR_SECRET_KEY")
OCR_API_URL    = os.getenv("CLOVA_OCR_URL")

print("OCR_SECRET_KEY =", (OCR_SECRET_KEY or "None")[:6], "...")
print("OCR_API_URL    =", (OCR_API_URL or "None")[:30], "...")
if not OCR_SECRET_KEY or not OCR_API_URL:
    raise RuntimeError("환경변수(.env)에 CLOVA_OCR_SECRET_KEY / CLOVA_OCR_URL을 설정하세요.")

# =========================
# OCR / YOLO
# =========================
def run_clova_ocr(image_path: Path):
    req_json = {
        'images': [{'format': 'jpg', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }
    payload = {'message': json.dumps(req_json).encode('UTF-8')}
    headers = {'X-OCR-SECRET': OCR_SECRET_KEY}
    with open(str(image_path), 'rb') as f:
        files_payload = [('file', f)]
        resp = requests.post(OCR_API_URL, headers=headers, data=payload, files=files_payload, timeout=60)
    resp.raise_for_status()

    ocr_results = []
    ocr_data = resp.json()
    for image_result in ocr_data.get('images', []):
        for field in image_result.get('fields', []):
            text = field.get('inferText', '')
            vertices = field.get('boundingPoly', {}).get('vertices', [])
            if len(vertices) >= 3:
                x1, y1 = vertices[0]['x'], vertices[0]['y']
                x2, y2 = vertices[2]['x'], vertices[2]['y']
                ocr_results.append({'text': text, 'bbox': [x1, y1, x2, y2]})
    return ocr_results

def run_yolo_detect(image_path: Path, weights_path: Path):
    model = YOLO(str(weights_path))
    img_pil = Image.open(str(image_path)).convert("RGB")
    img = np.array(img_pil)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = model(img_bgr)
    print(results[0].verbose())

    detected_symbols = []
    for *box, conf, cls in results[0].boxes.data:
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[int(cls)]
        print(f"🔹 {class_name} | bbox=({x1},{y1})~({x2},{y2}) | conf={float(conf):.2f}")
        detected_symbols.append({
            'class': class_name,
            'bbox': [x1, y1, x2, y2],
            'confidence': float(conf),
            'ocr_texts': []
        })
    return detected_symbols

# =========================
# 요약 판정(데모)
# =========================
@dataclass
class BreakerInfo:
    kind: str|None = None
    rated_voltage: str|None = None
    rated_current: str|None = None
    breaking_capacity: str|None = None

BREAKER_KIND_RE = r'\b(ACB|MCCB|VCB|GCB|ELB)\b'
VOLT_RE         = r'(\d{3,4})\s*V\b'
AMP_RE          = r'(\d{2,4})\s*A\b'
KA_RE           = r'(\d{1,3})\s*kA\b'

def parse_breaker_from_text(text: str) -> BreakerInfo:
    text = text.replace('P600V', 'P 600V').replace('4P600V','4P 600V')
    info = BreakerInfo()
    if m:=re.search(BREAKER_KIND_RE, text, re.I): info.kind = m.group(1).upper()
    if m:=re.search(VOLT_RE, text, re.I):         info.rated_voltage = m.group(1)+'V'
    if m:=re.search(AMP_RE, text, re.I):          info.rated_current = m.group(1)+'A'
    if m:=re.search(KA_RE, text, re.I):           info.breaking_capacity = m.group(1)+'kA'
    return info

def validate_breaker(info: BreakerInfo):
    missing = []
    if not info.kind:              missing.append("종류")
    if not info.rated_voltage:     missing.append("정격전압")
    if not info.rated_current:     missing.append("정격전류")
    if not info.breaking_capacity: missing.append("정격차단전류")
    return {"all_present": len(missing)==0, "missing": missing, "extracted": asdict(info)}

# =========================
# 메인
# =========================
def main():
    parser = argparse.ArgumentParser(description="YOLO + CLOVA OCR (로컬) → final_result.json / final_summary.json")
    parser.add_argument("--image", required=True, help="도면 이미지 경로 (jpg/png)")
    parser.add_argument("--weights", required=True, help="YOLO 가중치 경로 (예: data/models/best.pt)")
    parser.add_argument("--out", default="final_result.json", help="출력 JSON 경로")
    parser.add_argument("--dist_thr", type=int, default=30, help="(백업모드) 그래프 엣지 최대 거리(px) 하한선")
    args = parser.parse_args()

    image_path = Path(args.image)
    weights_path = Path(args.weights)
    output_json_path = Path(args.out)

    if not image_path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO 가중치 파일을 찾을 수 없습니다: {weights_path}")

    # 1) YOLO
    detected_symbols = run_yolo_detect(image_path, weights_path)

    # 2) OCR (예외 방어: 실패해도 빈 리스트로 진행해 JSON 생성 보장)
    print("CLOVA OCR 호출 중...")
    try:
        ocr_results = run_clova_ocr(image_path)
        print(f"OCR 텍스트 개수: {len(ocr_results)}")
    except Exception as e:
        print("[경고] OCR 실패 -> 빈 결과로 진행:", e)
        ocr_results = []

    # 3) 토폴로지 우선 매칭 (같은 회로만 후보)
    print("토폴로지 기반 매칭 수행.")
    detected_symbols = match_symbols_with_texts_graph_TOPOFIRST(
        detected_symbols,
        ocr_results,
        image_path=image_path,
        cli_max_link=max(args.dist_thr, 60)  # 백업모드에서 쓰일 상한 하한
    )

    # 3.5) 심볼별 속성 부착
    detected_symbols = attach_symbol_attributes(detected_symbols)

    # 4) 결과 저장(final_result.json)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(detected_symbols, f, indent=4, ensure_ascii=False)
    print(f"[저장] {output_json_path.resolve()}")

    # 5) 전체 텍스트 기반 1차 요약판정(final_summary.json) — 데모
    all_ocr_texts = []
    for s in detected_symbols:
        for t in s.get("ocr_texts", []):
            if txt := (t.get("text") or "").strip():
                all_ocr_texts.append(txt)
    flat_text = " ".join(all_ocr_texts)
    bi = parse_breaker_from_text(flat_text)
    judge = validate_breaker(bi)
    summary_path = output_json_path.with_name("final_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(judge, f, indent=4, ensure_ascii=False)
    print("[요약판정]", judge)
    print(f"[저장] {summary_path.resolve()}")

    # 6) 콘솔 요약(심볼 단위 연결 확인용)
    for symbol in detected_symbols:
        cls = symbol.get("class", "UNKNOWN")
        texts = [t["text"] for t in symbol.get("ocr_texts", []) if (t.get("text") or "").strip()]
        attrs = symbol.get("attributes", {})
        if texts:
            joined = '", "'.join(texts)
            print(f'[{cls}] 근처 텍스트: "{joined}"  -> attributes={attrs}')
        else:
            print(f'[{cls}] 근처 텍스트: 없음  -> attributes={attrs}')

if __name__ == "__main__":
    main()
