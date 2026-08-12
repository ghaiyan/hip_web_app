#!/usr/bin/env python3
"""
Labelme JSON → COCO + YOLO-Pose 双格式转换脚本
==============================================
将髋关节 X 光片 Labelme 标注同时转为:
  1. COCO keypoint JSON (供 mmpose / SimpleBaseline)
  2. YOLO pose txt 标签 (供 YOLOv8-pose)

输出结构:
  coco_format/
  ├── train2017/
  │   ├── images/          ← YOLO 图片
  │   └── labels/          ← YOLO .txt 标签
  ├── val2017/
  │   ├── images/
  │   └── labels/
  └── annotations/         ← COCO JSON (mmpose 用)
      ├── train2017.json
      └── val2017.json
"""

import json
import os
import glob
import random
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Tuple

# ====== 配置区 ======
DATA_ROOT = "/data/ghaiyan/髋关节影像检测/dataset/"
ANN_DIR = os.path.join(DATA_ROOT, "annotations")
IMG_DIR = os.path.join(DATA_ROOT, "images")
OUTPUT_DIR = os.path.join(DATA_ROOT, "coco_format")

TRAIN_RATIO = 0.85       # 训练集比例
RANDOM_SEED = 42
NUM_KPTS = 19             # 关键点数量

# ====== 19个关键点定义 ======
KEYPOINT_NAMES = [
    "P1_Teardrop_L",         #  0  左侧内泪滴
    "P2_Teardrop_R",         #  1  右侧内泪滴
    "P3_Symphysis",          #  2  耻骨联合
    "P4_HeadCenter_L",       #  3  左侧股骨头球心
    "P5_NeckCenter_L",       #  4  左侧股骨颈中点
    "P6_ShaftProx_L",        #  5  左侧股骨干近端(小转子)
    "P7_ShaftDist_L",        #  6  左侧股骨干远端
    "P8_CupOuterUp_L",       #  7  左侧臼杯外上缘
    "P9_CupInnerDown_L",     #  8  左侧臼杯内下缘
    "P10_CupAnt_L",          #  9  左侧臼杯前缘
    "P11_CupPost_L",         # 10  左侧臼杯后缘
    "P12_HeadCenter_R",      # 11  右侧股骨头球心
    "P13_NeckCenter_R",      # 12  右侧股骨颈中点
    "P14_ShaftProx_R",       # 13  右侧股骨干近端(小转子)
    "P15_ShaftDist_R",       # 14  右侧股骨干远端
    "P16_CupOuterUp_R",      # 15  右侧臼杯外上缘
    "P17_CupInnerDown_R",    # 16  右侧臼杯内下缘
    "P18_CupAnt_R",          # 17  右侧臼杯前缘
    "P19_CupPost_R",         # 18  右侧臼杯后缘
]

LABEL_TO_ID = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# ====== 骨架连接 ======
SKELETON = [
    [0, 1],   # 泪滴连线(基准线)
    [0, 2], [1, 2],  # 泪滴-耻骨
    [3, 4], [4, 5], [5, 6],  # 左侧: 头心→颈中→干近→干远
    [7, 8],  # 左侧臼杯长轴
    [8, 9], [8, 10],  # 左侧臼杯前缘/后缘
    [11, 12], [12, 13], [13, 14],  # 右侧: 头心→颈中→干近→干远
    [15, 16],  # 右侧臼杯长轴
    [16, 17], [16, 18],  # 右侧臼杯
]


def label_to_keypoint_index(label: str) -> Optional[int]:
    """将 Labelme 标注名映射到关键点索引"""
    return LABEL_TO_ID.get(label)


def find_image(json_path: str, ann_data: dict) -> Optional[str]:
    """根据 JSON 文件名和 imagePath 字段查找对应图片"""
    json_stem = os.path.splitext(os.path.basename(json_path))[0]
    img_rel = ann_data.get("imagePath", "")
    candidates = [
        img_rel,
        os.path.basename(img_rel),
        json_stem + os.path.splitext(img_rel)[1],
        json_stem + ".jpg",
        json_stem + ".png",
        json_stem + ".JPG",
    ]
    for cand in candidates:
        if cand and os.path.exists(os.path.join(IMG_DIR, cand)):
            return cand
    return None


def get_image_size(img_path: str, ann_data: dict) -> Tuple[int, int]:
    """读取图片尺寸"""
    try:
        from PIL import Image
        with Image.open(img_path) as pil_img:
            return pil_img.size
    except Exception:
        return ann_data.get("imageWidth", 512), ann_data.get("imageHeight", 512)


def extract_keypoints(shapes: list) -> Tuple[Dict[int, list], int]:
    """从 Labelme shapes 提取关键点, 返回 {idx: [x, y, v]} 和已标注数"""
    kpt_dict = {}
    for shape in shapes:
        lbl = shape.get("label", "")
        idx = label_to_keypoint_index(lbl)
        if idx is None:
            continue
        pts = shape.get("points", [[0, 0]])
        if len(pts) > 0:
            x, y = pts[0]
            kpt_dict[idx] = [float(x), float(y), 2]  # v=2: 可见已标注
    return kpt_dict, len(kpt_dict)


def load_annotations() -> list:
    """加载所有 Labelme JSON 文件"""
    json_files = sorted(glob.glob(os.path.join(ANN_DIR, "*.json")))
    items = []
    for f in json_files:
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        items.append((f, data))
    print(f"[INFO] 加载了 {len(items)} 个标注文件")
    return items


def make_split(items):
    """划分训练/验证集"""
    random.seed(RANDOM_SEED)
    random.shuffle(items)
    split_idx = int(len(items) * TRAIN_RATIO)
    return items[:split_idx], items[split_idx:]


def process_item(json_path, ann_data):
    """处理单个标注文件, 返回标准化数据或 None"""
    img_file = find_image(json_path, ann_data)
    if img_file is None:
        json_stem = os.path.splitext(os.path.basename(json_path))[0]
        print(f"  [SKIP] 找不到图片: {json_stem}")
        return None

    img_path = os.path.join(IMG_DIR, img_file)
    img_w, img_h = get_image_size(img_path, ann_data)
    kpt_dict, num_kpts = extract_keypoints(ann_data.get("shapes", []))

    # bbox (从可见关键点计算, 扩大15%)
    visible = [(v[0], v[1]) for v in kpt_dict.values() if v[2] == 2]
    if visible:
        xs = [p[0] for p in visible]
        ys = [p[1] for p in visible]
        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xs), max(ys)
        w = x_max - x_min
        h = y_max - y_min
        bbox = [max(0, x_min - w * 0.15), max(0, y_min - h * 0.15),
                min(img_w, w * 1.3), min(img_h, h * 1.3)]
    else:
        bbox = [0, 0, img_w, img_h]

    # 构建完整关键点数组 [x, y, v] × 19
    kpt_full = []
    for i in range(NUM_KPTS):
        if i in kpt_dict:
            kpt_full.extend(kpt_dict[i])
        else:
            kpt_full.extend([0.0, 0.0, 0])

    return {
        "img_file": img_file,
        "img_path": img_path,
        "img_w": img_w,
        "img_h": img_h,
        "bbox": bbox,
        "kpt": kpt_full,
        "kpt_dict": kpt_dict,
        "num_keypoints": num_kpts,
    }


def write_yolo_labels(items_data: list, label_dir: str):
    """生成 YOLO 格式的 .txt 标签文件

    YOLO pose 格式每行:
        class_id cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
    坐标为归一化值 (0~1), visibility: 0=未标注, 1=遮挡, 2=可见
    """
    os.makedirs(label_dir, exist_ok=True)
    written = 0
    for data in items_data:
        if data is None:
            continue

        # 归一化坐标
        img_w, img_h = data["img_w"], data["img_h"]
        kpt_norm = []
        for i in range(0, len(data["kpt"]), 3):
            x = data["kpt"][i] / img_w if img_w > 0 else 0.0
            y = data["kpt"][i + 1] / img_h if img_h > 0 else 0.0
            v = data["kpt"][i + 2]
            kpt_norm.extend([x, y, v])

        # 归一化 bbox: [cx, cy, w, h]
        bx, by, bw, bh = data["bbox"]
        cx = (bx + bw / 2) / img_w
        cy = (by + bh / 2) / img_h
        bw_n = bw / img_w
        bh_n = bh / img_h

        # 构建 YOLO 标签行
        parts = [0, cx, cy, bw_n, bh_n] + kpt_norm
        line = " ".join(f"{v:.6g}" for v in parts)

        # 写入 .txt (与图片同名)
        stem = os.path.splitext(data["img_file"])[0]
        txt_path = os.path.join(label_dir, stem + ".txt")
        with open(txt_path, "w") as f:
            f.write(line + "\n")
        written += 1

    return written


def copy_images(items_data: list, img_out_dir: str):
    """复制图片到目标目录"""
    os.makedirs(img_out_dir, exist_ok=True)
    copied = 0
    for data in items_data:
        if data is None:
            continue
        src = data["img_path"]
        dst = os.path.join(img_out_dir, data["img_file"])
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        copied += 1
    return copied


def build_coco_dict(items_data: list) -> dict:
    """将处理好的数据转为 COCO 格式 dict (供 mmpose)"""
    coco = {
        "info": {
            "description": "Hip Joint X-ray Keypoint Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "自建数据集",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [{"id": 1, "name": "CC BY-NC-SA 4.0", "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"}],
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "hip",
            "supercategory": "hip_joint",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": SKELETON,
        }],
    }

    image_id = 0
    annotation_id = 0

    for data in items_data:
        if data is None:
            continue

        image_id += 1
        coco["images"].append({
            "id": image_id,
            "file_name": data["img_file"],
            "width": data["img_w"],
            "height": data["img_h"],
        })

        annotation_id += 1
        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": data["bbox"],
            "area": data["bbox"][2] * data["bbox"][3],
            "iscrowd": 0,
            "keypoints": data["kpt"],
            "num_keypoints": data["num_keypoints"],
        })

    return coco


def main():
    items = load_annotations()
    train_items, val_items = make_split(items)
    print(f"[INFO] 训练集: {len(train_items)}, 验证集: {len(val_items)}")

    # ---- 处理所有标注 ----
    print("[INFO] 解析标注数据...")
    train_data = [process_item(jp, ad) for jp, ad in train_items]
    val_data = [process_item(jp, ad) for jp, ad in val_items]

    # ---- YOLO 格式输出 (YOLOv8-pose) ----
    # 目录结构: train2017/images/ + train2017/labels/
    train_img_dir = os.path.join(OUTPUT_DIR, "train2017", "images")
    train_lbl_dir = os.path.join(OUTPUT_DIR, "train2017", "labels")
    val_img_dir = os.path.join(OUTPUT_DIR, "val2017", "images")
    val_lbl_dir = os.path.join(OUTPUT_DIR, "val2017", "labels")

    print("[INFO] 复制训练集图片...")
    n1 = copy_images(train_data, train_img_dir)
    print(f"[INFO] 生成训练集 YOLO 标签...")
    n2 = write_yolo_labels(train_data, train_lbl_dir)

    print("[INFO] 复制验证集图片...")
    n3 = copy_images(val_data, val_img_dir)
    print(f"[INFO] 生成验证集 YOLO 标签...")
    n4 = write_yolo_labels(val_data, val_lbl_dir)

    print(f"[OK] YOLO 格式: 训练集 {n1} 图 + {n2} 标签, 验证集 {n3} 图 + {n4} 标签")

    # ---- COCO JSON 输出 (mmpose / SimpleBaseline) ----
    ann_output_dir = os.path.join(OUTPUT_DIR, "annotations")
    os.makedirs(ann_output_dir, exist_ok=True)

    print("[INFO] 生成训练集 COCO JSON...")
    train_coco = build_coco_dict(train_data)
    train_ann_path = os.path.join(ann_output_dir, "train2017.json")
    with open(train_ann_path, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=2)
    print(f"[OK] {train_ann_path}  ({len(train_coco['images'])} 图, {len(train_coco['annotations'])} 标注)")

    print("[INFO] 生成验证集 COCO JSON...")
    val_coco = build_coco_dict(val_data)
    val_ann_path = os.path.join(ann_output_dir, "val2017.json")
    with open(val_ann_path, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False, indent=2)
    print(f"[OK] {val_ann_path}  ({len(val_coco['images'])} 图, {len(val_coco['annotations'])} 标注)")

    # ---- 统计 ----
    print("\n====== 转换统计 ======")
    for name, coco in [("训练集", train_coco), ("验证集", val_coco)]:
        anns = coco["annotations"]
        kpt_counts = [a["num_keypoints"] for a in anns]
        avg_kpt = sum(kpt_counts) / len(kpt_counts) if kpt_counts else 0
        print(f"  {name}: {len(coco['images'])} 图, {len(anns)} 标注, 每图平均 {avg_kpt:.1f} 个关键点")

    print(f"\n[完成] 数据已输出到: {OUTPUT_DIR}")
    print("输出结构:")
    print("  ├── train2017/images/   (YOLOv8-pose 训练图片)")
    print("  ├── train2017/labels/   (YOLOv8-pose 训练标签 .txt)")
    print("  ├── val2017/images/     (YOLOv8-pose 验证图片)")
    print("  ├── val2017/labels/     (YOLOv8-pose 验证标签 .txt)")
    print("  └── annotations/        (COCO JSON: mmpose 用)")
    print("下一步:")
    print("  1. YOLOv8-pose: python train_yolov8_pose.py")
    print("  2. mmpose:       使用 annotations/ 下的 COCO JSON")
    print("  3. SimpleBaseline: python train_simplebaseline.py")


if __name__ == "__main__":
    main()
