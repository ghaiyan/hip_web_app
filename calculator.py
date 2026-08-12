"""
临床参数计算模块
================
从 19 个关键点坐标计算五项临床参数:
  1. 臼杯外展角 (Abduction Angle)
  2. 臼杯前倾角 (Anteversion Angle)
  3. 颈干角 (Neck-Shaft Angle)
  4. 股骨偏心距 (Femoral Offset)
  5. 下肢长度差 (Leg Length Discrepancy)
"""

import math
import numpy as np

# 关键点索引定义
KEYPOINT_NAMES = {
    0: "P1_Teardrop_L",       1: "P2_Teardrop_R",       2: "P3_Symphysis",
    3: "P4_HeadCenter_L",      4: "P5_NeckCenter_L",     5: "P6_ShaftProx_L",
    6: "P7_ShaftDist_L",       7: "P8_CupOuterUp_L",     8: "P9_CupInnerDown_L",
    9: "P10_CupAnt_L",        10: "P11_CupPost_L",
    11: "P12_HeadCenter_R",   12: "P13_NeckCenter_R",    13: "P14_ShaftProx_R",
    14: "P15_ShaftDist_R",    15: "P16_CupOuterUp_R",    16: "P17_CupInnerDown_R",
    17: "P18_CupAnt_R",       18: "P19_CupPost_R",
}

KPT_GROUPS = {
    "骨盆基准": [0, 1, 2],
    "股骨头/颈": [3, 4, 11, 12],
    "股骨干": [5, 6, 13, 14],
    "臼杯结构": [7, 8, 9, 10, 15, 16, 17, 18],
}

CLINICAL_REFERENCE = {
    "abduction_angle":    {"range": (30, 50),  "unit": "°",  "name_cn": "臼杯外展角",   "name_en": "Abduction Angle"},
    "anteversion_angle":  {"range": (5, 25),   "unit": "°",  "name_cn": "臼杯前倾角",   "name_en": "Anteversion Angle"},
    "neck_shaft_angle":   {"range": (120, 140),"unit": "°",  "name_cn": "颈干角",       "name_en": "Neck-Shaft Angle"},
    "femoral_offset":     {"range": None,       "unit": "px", "name_cn": "股骨偏心距",   "name_en": "Femoral Offset"},
    "leg_length_diff":    {"range": None,       "unit": "px", "name_cn": "下肢长度差",   "name_en": "Leg Length Discrepancy"},
}


def detect_implant_side(kpts):
    """自动判断假体侧"""
    has_L = not (np.allclose(kpts[7],  [0, 0]) and np.allclose(kpts[8],  [0, 0]))
    has_R = not (np.allclose(kpts[15], [0, 0]) and np.allclose(kpts[16], [0, 0]))
    if has_L and has_R:
        return "both"
    elif has_L:
        return "L"
    elif has_R:
        return "R"
    return None


def distance(a, b):
    """欧氏距离"""
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def angle_between(v1, v2):
    """两个向量的夹角 (度)"""
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-8:
        return None
    cos_val = np.clip(dot / norm, -1.0, 1.0)
    return math.degrees(math.acos(cos_val))


def point_to_line_dist(point, line_pt1, line_pt2):
    """点到直线的垂直距离"""
    x0, y0 = point
    x1, y1 = line_pt1
    x2, y2 = line_pt2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    norm = math.sqrt(a**2 + b**2)
    if norm < 1e-8:
        return 0.0
    return abs(a * x0 + b * y0 + c) / norm


def calc_abduction_angle(kpts, baseline_pt1, baseline_pt2, side):
    """臼杯外展角: 臼杯长轴(外上→内下) 与 基准线(泪滴连线) 夹角"""
    if side == "L":
        outer, inner = kpts[7], kpts[8]
    else:
        outer, inner = kpts[15], kpts[16]

    if np.allclose(outer, [0, 0]) or np.allclose(inner, [0, 0]):
        return None

    cup_axis = np.array(inner) - np.array(outer)
    baseline = np.array(baseline_pt2) - np.array(baseline_pt1)

    ang = angle_between(cup_axis, baseline)
    if ang is not None and ang > 90:
        ang = 180 - ang
    return round(ang, 1) if ang is not None else None


def calc_anteversion_angle(kpts, side):
    """臼杯前倾角: arcsin(短轴/长轴)"""
    if side == "L":
        outer, inner = kpts[7], kpts[8]
        ant, post = kpts[9], kpts[10]
    else:
        outer, inner = kpts[15], kpts[16]
        ant, post = kpts[17], kpts[18]

    if any(np.allclose(p, [0, 0]) for p in [outer, inner, ant, post]):
        return None

    long_len = distance(outer, inner)
    short_len = distance(ant, post)
    if long_len < 1e-3:
        return None

    ratio = min(short_len / long_len, 1.0)
    return round(math.degrees(math.asin(ratio)), 1)


def calc_neck_shaft_angle(kpts, side):
    """颈干角: 颈轴(头心→颈中点) 与 干轴(干近端→干远端) 的钝角"""
    if side == "L":
        head, neck, shaft_prox, shaft_dist = kpts[3], kpts[4], kpts[5], kpts[6]
    else:
        head, neck, shaft_prox, shaft_dist = kpts[11], kpts[12], kpts[13], kpts[14]

    if any(np.allclose(p, [0, 0]) for p in [head, neck, shaft_prox, shaft_dist]):
        return None

    neck_axis = np.array(neck) - np.array(head)
    shaft_axis = np.array(shaft_dist) - np.array(shaft_prox)

    ang = angle_between(neck_axis, shaft_axis)
    if ang is None:
        return None
    # 取钝角侧
    if ang < 90:
        ang = 180 - ang
    return round(ang, 1)


def calc_femoral_offset(kpts, side):
    """股骨偏心距: 头心到股骨干轴线的垂直距离"""
    if side == "L":
        head, shaft_prox, shaft_dist = kpts[3], kpts[5], kpts[6]
    else:
        head, shaft_prox, shaft_dist = kpts[11], kpts[13], kpts[14]

    if any(np.allclose(p, [0, 0]) for p in [head, shaft_prox, shaft_dist]):
        return None

    shaft_dir = np.array(shaft_dist) - np.array(shaft_prox)
    shaft_len = np.linalg.norm(shaft_dir)
    if shaft_len < 1e-8:
        return None
    shaft_unit = shaft_dir / shaft_len
    shaft_perp = np.array([-shaft_unit[1], shaft_unit[0]])
    vec_head = np.array(head) - np.array(shaft_prox)
    return round(float(abs(np.dot(vec_head, shaft_perp))), 1)


def calc_leg_length_diff(kpts):
    """下肢长度差: 双侧小转子到基准线的垂直距离之差"""
    p1, p2 = kpts[0], kpts[1]
    p6_L, p14_R = kpts[5], kpts[13]

    if np.allclose(p1, [0, 0]) or np.allclose(p2, [0, 0]):
        return None

    result = {}
    if not np.allclose(p6_L, [0, 0]):
        d_L = point_to_line_dist(p6_L, p1, p2)
        result["L"] = round(d_L, 1)
    if not np.allclose(p14_R, [0, 0]):
        d_R = point_to_line_dist(p14_R, p1, p2)
        result["R"] = round(d_R, 1)

    if "L" in result and "R" in result:
        result["diff"] = round(abs(result["L"] - result["R"]), 1)
    return result


def compute_all_parameters(kpts, img_size):
    """
    主计算函数

    Args:
        kpts: [19, 2] numpy array, 关键点像素坐标
        img_size: (width, height)

    Returns:
        {
            "side": "L" / "R" / "both",
            "parameters": { ... },
            "warnings": [ ... ],
            "reference": { ... },   # 临床参考范围
        }
    """
    warnings = []
    params = {}
    kpts = np.array(kpts, dtype=float)

    # ---- 基准线: 左右泪滴 ----
    p1 = kpts[0]  # P1_Teardrop_L
    p2 = kpts[1]  # P2_Teardrop_R
    if np.allclose(p1, [0, 0]) or np.allclose(p2, [0, 0]):
        warnings.append({"key": "baseline_missing", "args": []})
        return {"side": None, "parameters": params, "warnings": warnings, "reference": CLINICAL_REFERENCE}

    baseline_len = distance(p1, p2)
    params["baseline_length_px"] = round(baseline_len, 1)

    # ---- 判断假体侧 ----
    implant_side = detect_implant_side(kpts)
    if implant_side is None:
        warnings.append({"key": "implant_side_unknown", "args": []})
        return {"side": None, "parameters": params, "warnings": warnings, "reference": CLINICAL_REFERENCE}

    sides = ["L", "R"] if implant_side == "both" else [implant_side]

    for side in sides:
        suffix = f"_{side}"

        # 1. 臼杯外展角
        abd = calc_abduction_angle(kpts, p1, p2, side)
        if abd is not None:
            params[f"abduction_angle{suffix}"] = abd
        else:
            warnings.append({"key": "abduction_missing", "args": [side]})

        # 2. 臼杯前倾角
        ant = calc_anteversion_angle(kpts, side)
        if ant is not None:
            params[f"anteversion_angle{suffix}"] = ant
        else:
            if f"abduction_angle{suffix}" in params:
                warnings.append({"key": "anteversion_missing", "args": [side]})

        # 3. 颈干角
        nsa = calc_neck_shaft_angle(kpts, side)
        if nsa is not None:
            params[f"neck_shaft_angle{suffix}"] = nsa
        else:
            warnings.append({"key": "neck_shaft_missing", "args": [side]})

        # 4. 股骨偏心距
        off = calc_femoral_offset(kpts, side)
        if off is not None:
            params[f"femoral_offset{suffix}"] = off
        else:
            warnings.append({"key": "femoral_offset_missing", "args": [side]})

    # 5. 下肢长度差 (仅双侧)
    lld = calc_leg_length_diff(kpts)
    if lld and "diff" in lld:
        params["leg_length_L_px"] = lld.get("L", None)
        params["leg_length_R_px"] = lld.get("R", None)
        params["leg_length_diff_px"] = lld["diff"]
    elif lld:
        # 单侧也记录
        for k, v in lld.items():
            params[f"leg_length_{k}_px"] = v
    else:
        warnings.append({"key": "leg_length_missing", "args": []})

    # ---- 构建返回结果 ----
    return {
        "side": implant_side,
        "parameters": params,
        "warnings": warnings,
        "reference": CLINICAL_REFERENCE,
        "img_size": list(img_size),
        "keypoints_raw": {i: [float(kpts[i][0]), float(kpts[i][1])] for i in range(19)},
    }
