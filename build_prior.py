#!/usr/bin/env python3
"""
build_prior.py — 从 LabelMe 标注统计 P1-P19 先验坐标分布
==========================================================
输出: hip_web_app/keypoint_prior.json
格式:
{
  "all": {
    "P1": {"mean_x": 0.45, "mean_y": 0.18, "std_x": 0.03, "std_y": 0.02, "n": 250},
    ...
  },
  "bilateral": { ... },   # 双侧置换（含 P4-P11 和 P12-P19）
  "left_only": { ... },   # 仅左侧（含 P4-P11，不含 P12-P19）
  "right_only": { ... }   # 仅右侧（含 P12-P19，不含 P4-P11）
}
"""

import json
import glob
import os
import numpy as np
from collections import defaultdict

ANN_DIR = "G:/2-论文写作相关/44-髋关节人工关节关键点检测/影像数据标注/dataset/annotations"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "keypoint_prior.json")

# 19个关键点名称（按 0-indexed P1-P19）
KEYPOINT_LABELS = [
    "P1_Teardrop_L", "P2_Teardrop_R", "P3_Symphysis",
    "P4_HeadCenter_L", "P5_NeckCenter_L", "P6_ShaftProx_L", "P7_ShaftDist_L",
    "P8_CupOuterUp_L", "P9_CupInnerDown_L", "P10_CupAnt_L", "P11_CupPost_L",
    "P12_HeadCenter_R", "P13_NeckCenter_R", "P14_ShaftProx_R", "P15_ShaftDist_R",
    "P16_CupOuterUp_R", "P17_CupInnerDown_R", "P18_CupAnt_R", "P19_CupPost_R",
]
P_KEY = {name: f"P{i+1}" for i, name in enumerate(KEYPOINT_LABELS)}

# 左侧 / 右侧关键点 key
LEFT_KPTS  = {"P4","P5","P6","P7","P8","P9","P10","P11"}
RIGHT_KPTS = {"P12","P13","P14","P15","P16","P17","P18","P19"}


def load_annotation(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    w = data.get("imageWidth", 1)
    h = data.get("imageHeight", 1)
    kpts = {}  # key -> (norm_x, norm_y)
    for shape in data.get("shapes", []):
        label = shape["label"]
        if not label.startswith("P"):
            continue
        key = P_KEY.get(label)
        if key is None:
            # 兼容只有 "P1" 这种短标注
            parts = label.split("_")[0]
            key = parts if parts.startswith("P") else None
        if key is None:
            continue
        pts = shape["points"]
        if pts:
            x, y = pts[0]
            kpts[key] = (x / w, y / h)
    return kpts, w, h


def classify_side(kpts):
    """根据存在的左右关键点判断拍摄侧别"""
    has_left  = any(k in kpts for k in LEFT_KPTS)
    has_right = any(k in kpts for k in RIGHT_KPTS)
    if has_left and has_right:
        return "bilateral"
    elif has_left:
        return "left_only"
    elif has_right:
        return "right_only"
    else:
        return "unknown"


def collect_stats(coord_lists):
    """coord_lists: list of (norm_x, norm_y), 返回统计字典"""
    if not coord_lists:
        return None
    xs = [c[0] for c in coord_lists]
    ys = [c[1] for c in coord_lists]
    return {
        "mean_x": float(np.mean(xs)),
        "mean_y": float(np.mean(ys)),
        "std_x":  float(np.std(xs)),
        "std_y":  float(np.std(ys)),
        "min_x":  float(np.min(xs)),
        "max_x":  float(np.max(xs)),
        "min_y":  float(np.min(ys)),
        "max_y":  float(np.max(ys)),
        "n":      len(coord_lists),
    }


def main():
    ann_files = sorted(glob.glob(os.path.join(ANN_DIR, "*.json")))
    print(f"Found {len(ann_files)} annotation files")

    # 按 group 分组收集坐标
    # groups: "all", "bilateral", "left_only", "right_only"
    coords = {
        "all":        defaultdict(list),
        "bilateral":  defaultdict(list),
        "left_only":  defaultdict(list),
        "right_only": defaultdict(list),
    }

    side_counts = defaultdict(int)
    skip_count = 0

    for fp in ann_files:
        try:
            kpts, w, h = load_annotation(fp)
        except Exception as e:
            print(f"  SKIP {os.path.basename(fp)}: {e}")
            skip_count += 1
            continue

        if not kpts:
            skip_count += 1
            continue

        side = classify_side(kpts)
        side_counts[side] += 1

        for key, (nx, ny) in kpts.items():
            coords["all"][key].append((nx, ny))
            if side in coords:
                coords[side][key].append((nx, ny))

    print(f"\nSide distribution: {dict(side_counts)}")
    print(f"Skipped: {skip_count}")

    # 生成统计结果
    result = {}
    for group, kpt_dict in coords.items():
        result[group] = {}
        for pkey in [f"P{i}" for i in range(1, 20)]:
            stat = collect_stats(kpt_dict.get(pkey, []))
            if stat:
                result[group][pkey] = stat

    # 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {OUTPUT_FILE}")

    # 打印摘要
    print("\n=== ALL group summary (first 6 pts) ===")
    for pkey in [f"P{i}" for i in range(1, 7)]:
        s = result["all"].get(pkey)
        if s:
            print(f"  {pkey}: x={s['mean_x']:.3f}±{s['std_x']:.3f}  y={s['mean_y']:.3f}±{s['std_y']:.3f}  n={s['n']}")

    print("\n=== Bilateral group - P6 vs P14 (key troublemakers) ===")
    for pkey in ["P6", "P7", "P14", "P15"]:
        s = result["bilateral"].get(pkey)
        if s:
            print(f"  {pkey}: x_mean={s['mean_x']:.3f}  x_range=[{s['min_x']:.3f}, {s['max_x']:.3f}]  n={s['n']}")


if __name__ == "__main__":
    main()
