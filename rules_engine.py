# rules_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Dict, Any, Tuple


# ---------------- 공통 보조 함수 ----------------

def _center(bbox) -> Tuple[float, float]:
    """bbox = [x1,y1,x2,y2] -> 중심 좌표 (cx, cy)"""
    x1, y1, x2, y2 = bbox
    return ( (x1 + x2) / 2.0, (y1 + y2) / 2.0 )


def _h_overlap_ratio(a, b) -> float:
    """가로 방향 겹침 비율 (0~1)"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    left  = max(ax1, bx1)
    right = min(ax2, bx2)
    if right <= left:
        return 0.0
    overlap = right - left
    width   = max(ax2 - ax1, bx2 - bx1, 1e-6)
    return overlap / width


def _filter(symbols: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    """특정 class 이름만 골라내기"""
    return [s for s in symbols if s.get("class") == label]


# ================= 규칙 4 =================
# 4. LA (피뢰기) 위에 DS(단로기)가 있으면 안 된다.

def check_la_ds(symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    violations = []
    las = _filter(symbols, "LA")
    dss = _filter(symbols, "DS")

    for la in las:
        la_bbox = la.get("bbox", [0, 0, 0, 0])
        la_cx, la_cy = _center(la_bbox)

        for ds in dss:
            ds_bbox = ds.get("bbox", [0, 0, 0, 0])
            ds_cx, ds_cy = _center(ds_bbox)

            # x 방향이 어느 정도 겹치고, DS가 LA보다 위에 있는 경우
            h_ov = _h_overlap_ratio(la_bbox, ds_bbox)
            if h_ov > 0.3 and ds_cy < la_cy:
                violations.append({
                    "rule_id": "R4-LA-DS",
                    "desc": "LA(피뢰기)의 위쪽에 DS(단로기)가 위치해 있습니다. "
                            "규정상 LA 위에 DS를 둘 수 없습니다.",
                    "severity": "HIGH",
                    "page": la.get("page", 1),
                    "bbox": la_bbox,
                })
    return violations


# ================= 규칙 6 =================
# 6. MOF 1차측에 PF가 반드시 위치해야 한다.

def check_mof_pf(symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    violations = []
    mofs = _filter(symbols, "MOF")
    pfs  = _filter(symbols, "PF")

    for mof in mofs:
        mof_bbox = mof.get("bbox", [0, 0, 0, 0])
        mof_cx, mof_cy = _center(mof_bbox)

        # MOF 위쪽(1차측)에 PF가 하나라도 있는지 확인
        found_pf = False
        for pf in pfs:
            pf_bbox = pf.get("bbox", [0, 0, 0, 0])
            pf_cx, pf_cy = _center(pf_bbox)

            same_column = _h_overlap_ratio(mof_bbox, pf_bbox) > 0.4
            if same_column and pf_cy < mof_cy:
                found_pf = True
                break

        if not found_pf:
            violations.append({
                "rule_id": "R6-MOF-PF",
                "desc": "MOF 1차측(위쪽)에 PF(퓨즈)가 없습니다. "
                        "규정상 MOF 1차측에는 PF가 반드시 위치해야 합니다.",
                "severity": "HIGH",
                "page": mof.get("page", 1),
                "bbox": mof_bbox,
            })
    return violations


# ================= 규칙 5 =================
# 5. 계통이 66kV 이상이면 DS 금지, LS 사용

def check_voltage_ds_ls(symbols: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    violations = []

    # meta 안에 system_voltage_kv 가 있다고 가정 (없으면 0으로)
    try:
        v_raw = str(meta.get("system_voltage_kv", "0")).replace("kV", "").strip()
        v_kv = float(v_raw)
    except Exception:
        v_kv = 0.0

    dss = _filter(symbols, "DS")
    lss = _filter(symbols, "LS")

    if v_kv >= 66.0:
        if dss:
            bbox = dss[0].get("bbox", [0, 0, 0, 0])
            violations.append({
                "rule_id": "R5-DS-FORBIDDEN",
                "desc": f"계통 전압이 약 {v_kv:.0f}kV로 추정됩니다. "
                        "66kV 이상에서는 DS(단로기)를 사용할 수 없고, "
                        "LS(선로개폐기)를 사용해야 합니다.",
                "severity": "HIGH",
                "page": dss[0].get("page", 1),
                "bbox": bbox,
            })
        if not lss:
            violations.append({
                "rule_id": "R5-LS-MISSING",
                "desc": f"계통 전압이 약 {v_kv:.0f}kV로 추정됩니다. "
                        "66kV 이상에서는 LS(선로개폐기)를 사용하는 것이 일반적이나 "
                        "도면에서 LS 심볼이 인식되지 않았습니다.",
                "severity": "MEDIUM",
                "page": 1,
                "bbox": [0, 0, 0, 0],
            })
    return violations


# ================= 규칙 1,2,3 =================
# 1·2·3. CB 1차/2차측에 CT/PT 위치 조합 + 1000kVA 간이수전

def _classify_ct_pt_pattern(cb, cts, pts):
    """
    CB 하나에 대해,
    - CT/ PT 가 CB 위/아래 어디에 있는지 보고 패턴 문자열 리턴
    반환값 예: "T_CT_T_PT", "T_PT_B_CT", "T_CT_B_PT", "NONE"
    """
    cb_bbox = cb.get("bbox", [0, 0, 0, 0])
    cb_cx, cb_cy = _center(cb_bbox)

    def above(sym):
        cx, cy = _center(sym.get("bbox", [0, 0, 0, 0]))
        return cy < cb_cy and _h_overlap_ratio(cb_bbox, sym["bbox"]) > 0.3

    def below(sym):
        cx, cy = _center(sym.get("bbox", [0, 0, 0, 0]))
        return cy > cb_cy and _h_overlap_ratio(cb_bbox, sym["bbox"]) > 0.3

    top_ct    = any(above(ct) for ct in cts)
    bottom_ct = any(below(ct) for ct in cts)
    top_pt    = any(above(pt) for pt in pts)
    bottom_pt = any(below(pt) for pt in pts)

    if top_ct and top_pt:
        return "T_CT_T_PT"
    if top_pt and bottom_ct:
        return "T_PT_B_CT"
    if top_ct and bottom_pt:
        return "T_CT_B_PT"
    if not (top_ct or bottom_ct or top_pt or bottom_pt):
        return "NONE"
    return "INVALID"


def check_ct_pt_cb_patterns(symbols: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []

    cbs = _filter(symbols, "CB")
    cts = _filter(symbols, "CT")
    pts = _filter(symbols, "PT")

    # 변압기 용량(meta['transformer_kva'])가 1000kVA 이하이면
    # PT/CT는 "필수 아님"이라 없어도 위반 X
    try:
        kva_raw = str(meta.get("transformer_kva", "0")).replace("kVA", "").strip()
        kva = float(kva_raw)
    except Exception:
        kva = 0.0

    for cb in cbs:
        pattern = _classify_ct_pt_pattern(cb, cts, pts)

        # 3. 1000kVA 이하인 경우: PT/CT 없어도 허용
        if pattern == "NONE":
            if kva > 1000:  # 1000kVA 초과인데 CT/PT가 전혀 없으면 위반
                violations.append({
                    "rule_id": "R3-CTPT-MISSING",
                    "desc": f"변압기 용량이 약 {kva:.0f}kVA 이상으로 추정되지만, "
                            "CB 주변에서 CT/PT 심볼이 인식되지 않았습니다. "
                            "1000kVA 초과 수전설비에서는 CT/PT가 요구됩니다.",
                    "severity": "HIGH",
                    "page": cb.get("page", 1),
                    "bbox": cb.get("bbox", [0, 0, 0, 0]),
                })
            # 1000kVA 이하인 경우는 괜찮으므로 그냥 넘어감
            continue

        # 1·2·3. 허용되는 세 가지 패턴 검사
        if pattern not in ("T_CT_T_PT", "T_PT_B_CT", "T_CT_B_PT"):
            violations.append({
                "rule_id": "R1-3-CTPT-ARRANGE",
                "desc": "CB 주변 CT/PT의 배치가 규정된 세 가지 패턴 중 어느 것에도 맞지 않습니다. "
                        "1) CB 1차측에 CT와 PT, 2) CB 1차측 PT·2차측 CT, "
                        "3) CB 1차측 CT·2차측 PT 패턴을 다시 확인해야 합니다.",
                "severity": "HIGH",
                "page": cb.get("page", 1),
                "bbox": cb.get("bbox", [0, 0, 0, 0]),
            })

        # 2. CB 2차측에 PT를 시설하는 경우, PT 옆 PF 불필요
        if pattern in ("T_CT_B_PT", "T_PT_B_CT"):  # PT가 2차측(아래)에 존재
            # PT(아래쪽)에 붙어 있는 PF가 있는지 찾기
            pfs = _filter(symbols, "PF")
            for pt in pts:
                pt_bbox = pt.get("bbox", [0, 0, 0, 0])
                pt_cx, pt_cy = _center(pt_bbox)
                if pt_cy <= _center(cb.get("bbox", [0,0,0,0]))[1]:
                    continue  # 위에 있는 PT는 무시 (여기서는 2차측만 보고 싶음)

                for pf in pfs:
                    pf_bbox = pf.get("bbox", [0, 0, 0, 0])
                    # PT 옆에 PF가 붙어 있으면 위반
                    if _h_overlap_ratio(pt_bbox, pf_bbox) > 0.4:
                        violations.append({
                            "rule_id": "R2-PT-PF-SECONDARY",
                            "desc": "CB 2차측에 PT가 시설된 경우, MOF 전단에 차단기가 있어 "
                                    "PT 옆의 PF는 불필요합니다. PT 바로 옆에 PF가 인식되었습니다.",
                            "severity": "MEDIUM",
                            "page": pt.get("page", 1),
                            "bbox": pt_bbox,
                        })
                        break  # PT 하나당 한 번만 경고
    return violations


# ---------------- 모든 규칙 한 번에 실행 ----------------

def check_all_rules(symbols: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    YOLO+OCR에서 얻은 symbols 리스트 + 도면 메타정보(meta)를 받아,
    6개 규정을 모두 체크하고 위반 리스트를 반환.
    """
    violations: List[Dict[str, Any]] = []

    # 4번 규칙: LA 위에 DS 금지
    violations += check_la_ds(symbols)

    # 6번 규칙: MOF 1차측 PF 필수
    violations += check_mof_pf(symbols)

    # 5번 규칙: 66kV 이상 DS 금지, LS 사용
    violations += check_voltage_ds_ls(symbols, meta)

    # 1,2,3번 규칙: CT/PT 배치 + 1000kVA 간이수전
    violations += check_ct_pt_cb_patterns(symbols, meta)

    return violations

