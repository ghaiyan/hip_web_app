#!/usr/bin/env python3
"""
YOLOv8-Pose 训练脚本 (方案一: 最快上手)
========================================
依赖: pip install ultralytics

先运行 convert_to_coco.py，再运行本脚本。

训练策略:
  - 使用 yolov8n-pose.pt 预训练权重 (在 COCO-Pose 上预训练)
  - 输入尺寸 640x640
  - 强力数据增强 + 小样本策略
  - 多种模型尺寸可选

环境准备:
  pip install ultralytics torch torchvision
"""

import os
import sys

# ====== 训练配置 ======
DATA_YAML = r"/data/ghaiyan/髋关节影像检测/dataset/coco_format/hip_kpt.yaml"

# 模型选择 (从小到大):
#   yolov8n-pose.pt  - Nano     (3.2M 参数, 最快)
#   yolov8s-pose.pt  - Small    (11.2M)
#   yolov8m-pose.pt  - Medium   (26.4M)
#   yolov8l-pose.pt  - Large    (45.0M, 推荐)
#   yolov8x-pose.pt  - XLarge   (69.8M)
MODEL_NAME = "yolov8m-pose.pt"

# 训练超参数
IMG_SIZE = 640          # 输入分辨率
EPOCHS = 300            # 总轮数
BATCH_SIZE = 8          # batch size
DEVICE = "0"            # GPU ID, "cpu" 表示 CPU

# 数据增强 (髋关节 X 光专用策略)
# 核心原则：保留真实的骨盆几何结构，只做物理上合理的增强
AUG_HYPER = {
    # --- 色彩 (X光为灰度图，只调亮度/对比度，关闭色调/饱和度) ---
    "hsv_h": 0.0,       # 关闭色调变化（灰度图无效）
    "hsv_s": 0.0,       # 关闭饱和度变化（灰度图无效）
    "hsv_v": 0.3,       # 亮度±30%，模拟不同曝光条件

    # --- 几何变换（保守，符合临床拍摄范围）---
    "degrees": 5.0,     # 旋转±5°（摆位误差范围内）
    "translate": 0.05,  # 平移5%（轻微位置偏移）
    "scale": 0.15,      # 缩放±15%（0.85~1.15，不过度压缩）
    "shear": 0.0,       # 关闭剪切（X光不存在剪切失真）
    "perspective": 0.0, # 关闭透视（平片无透视变形）

    # --- 翻转 ---
    "flipud": 0.0,      # 禁止上下翻转（骨盆有明确上下方向）
    "fliplr": 0.5,      # 允许左右翻转（需 flip_idx 已配置）

    # --- 拼贴/混合（X光专用：全部关闭）---
    "mosaic": 0.0,      # 关闭！mosaic 破坏骨盆整体空间关系
    "mixup": 0.0,       # 关闭！混合图像产生不真实解剖结构
    "copy_paste": 0.0,  # 关闭

    # --- 遮挡增强（适度保留，模拟植入物/金属伪影遮挡）---
    "erasing": 0.1,     # 随机擦除10%（模拟术后金属伪影）
}


def generate_yaml():
    """生成 YOLOv8 训练所需的 YAML 配置文件"""
    coco_dir = r"/data/ghaiyan/髋关节影像检测/dataset/coco_format"

    # 19个关键点 (YOLO 格式要求每个关键点的 [x,y,visible])
    kpt_shape = [19, 3]  # 19 个关键点, 每个有 (x, y, visibility)

    # keypoint_names 用于可视化（必须与 convert_to_coco.py 的 KEYPOINT_NAMES 一致）
    keypoint_names = [
        "P1_Teardrop_L", "P2_Teardrop_R", "P3_Symphysis",
        "P4_HeadCenter_L", "P5_NeckCenter_L", "P6_ShaftProx_L", "P7_ShaftDist_L",
        "P8_CupOuterUp_L", "P9_CupInnerDown_L", "P10_CupAnt_L", "P11_CupPost_L",
        "P12_HeadCenter_R", "P13_NeckCenter_R", "P14_ShaftProx_R", "P15_ShaftDist_R",
        "P16_CupOuterUp_R", "P17_CupInnerDown_R", "P18_CupAnt_R", "P19_CupPost_R",
    ]

    # 骨架连接 (左右对称, 用于可视化)
    skeleton = [
        [0, 1],    # 泪滴连线
        [0, 2], [1, 2],
        [3, 4], [4, 5], [5, 6],   # 左侧股骨链
        [7, 8],                     # 左侧臼杯长轴
        [8, 9], [8, 10],           # 左侧臼杯前后缘
        [11, 12], [12, 13], [13, 14],  # 右侧股骨链
        [15, 16],                   # 右侧臼杯长轴
        [16, 17], [16, 18],        # 右侧臼杯前后缘
    ]

    yaml_content = f"""# YOLOv8-Pose 髋关节关键点检测配置文件
# 自动生成，请勿手动修改

path: {coco_dir}
train: train2017
val: val2017

# 关键点定义
kpt_shape: {kpt_shape}
flip_idx: [1, 0, 2, 11, 12, 13, 14, 15, 16, 17, 18, 3, 4, 5, 6, 7, 8, 9, 10]

# 类别
nc: 1
names:
  0: hip_joint

# 关键点名称 (仅用于可视化)
keypoint_names:
{chr(10).join(f'  - {n}' for n in keypoint_names)}

# 骨架连接 (仅用于可视化)
skeleton:
{chr(10).join(f'  - {s}' for s in skeleton)}
"""

    os.makedirs(os.path.dirname(DATA_YAML), exist_ok=True)
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[OK] YAML 配置文件已生成: {DATA_YAML}")


def train():
    """运行 YOLOv8-pose 训练"""
    from ultralytics import YOLO

    print(f"[INFO] 加载模型: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    print(f"[INFO] 开始训练...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        
        # 优化器
        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-4,
        weight_decay=1e-4,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # 数据增强
        **AUG_HYPER,
        
        # 训练策略
        cos_lr=True,         # 余弦退火学习率
        close_mosaic=10,     # 最后 10 个 epoch 关闭 mosaic
        dropout=0.1,         # 分类头 dropout
        
        # 验证
        val=True,
        save=True,
        save_period=10,
        
        # 日志
        project="hip_kpt_yolov8",
        name=f"hip_{MODEL_NAME.replace('.pt','')}",
        exist_ok=True,
        plots=False,         # 禁用绘图（headless 服务器兼容）
        
        # 预训练
        pretrained=True,
        
        # 早停
        patience=50,
        
        # 混合精度
        amp=True if DEVICE != "cpu" else False,
        
        # 其他
        workers=4,
        verbose=True,
    )

    print(f"\n[完成] 训练结束!")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    print(f"验证结果: {results.results_dict}")


def validate():
    """仅验证模式"""
    from ultralytics import YOLO
    
    best_pt = f"hip_kpt_yolov8/hip_{MODEL_NAME.replace('.pt','')}/weights/best.pt"
    if not os.path.exists(best_pt):
        print(f"[ERROR] 找不到模型: {best_pt}")
        return
    
    model = YOLO(best_pt)
    metrics = model.val(data=DATA_YAML, split="val")
    print(f"\n验证指标:")
    print(f"  mAP@0.5:0.95 = {metrics.box.map:.4f}")
    if hasattr(metrics, 'pose'):
        print(f"  关键点 mAP@0.5 = {metrics.pose.map50:.4f}")


def export_onnx():
    """导出 ONNX 格式 (可选)"""
    from ultralytics import YOLO
    
    best_pt = f"hip_kpt_yolov8/hip_{MODEL_NAME.replace('.pt','')}/weights/best.pt"
    model = YOLO(best_pt)
    path = model.export(format="onnx", imgsz=IMG_SIZE, simplify=True)
    print(f"[OK] ONNX 模型已导出: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "val", "export"], default="train")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--generate-yaml", action="store_true", default=True)
    args = parser.parse_args()

    # 更新全局变量
    MODEL_NAME = args.model
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    DEVICE = args.device

    generate_yaml()

    if args.mode == "train":
        train()
    elif args.mode == "val":
        validate()
    elif args.mode == "export":
        export_onnx()
