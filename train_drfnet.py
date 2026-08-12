#!/usr/bin/env python3
"""
DRFNet 消融实验训练脚本
=======================
Dual-Representation Fusion Network for Hip Joint Keypoint Detection

融合 YOLOv8-Pose 的"先检测后定位"范式与 SimpleBaseline 的全局热图回归范式，
通过检测引导空间注意力(DGSAM)、置信度引导检测精修(CGDR)、自适应融合模块(AFM)
实现双表征互补增强。

消融实验配置:
  C: python train_drfnet_ablation.py --ablation C   # 双分支, 无 DGSAM
  D: python train_drfnet_ablation.py --ablation D   # 双分支 + DGSAM, 无 CGDR
  E: python train_drfnet_ablation.py --ablation E   # 双分支 + DGSAM + CGDR, 等权融合
  F: python train_drfnet_ablation.py --ablation F   # 完整 DRFNet (AFM 自适应融合)

环境准备:
  pip install torch torchvision albumentations tensorboard
"""

import os
import sys
import math
import json
import copy
import random
import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import torchvision
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights

# 可选: albumentations
try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[WARN] albumentations 未安装, 将使用 torchvision 内置增强")
    print("[WARN] 安装命令: pip install albumentations")


# =====================================================================
#  消融配置
# =====================================================================
ABLATION_CONFIGS = {
    "C": {
        "use_dgsam": False, "use_cgdr": False, "use_afm": False,
        "desc": "双分支 + 无 DGSAM (验证空间注意力贡献)",
        "output_mode": "heatmap",  # 仅热图分支输出
    },
    "D": {
        "use_dgsam": True, "use_cgdr": False, "use_afm": False,
        "desc": "双分支 + DGSAM + 无 CGDR (验证置信度引导精修贡献)",
        "output_mode": "heatmap",  # 仅热图分支输出
    },
    "E": {
        "use_dgsam": True, "use_cgdr": True, "use_afm": False,
        "desc": "双分支 + DGSAM + CGDR + 等权融合 (验证自适应融合贡献)",
        "output_mode": "equal_fusion",  # 0.5 * refined + 0.5 * heatmap
    },
    "F": {
        "use_dgsam": True, "use_cgdr": True, "use_afm": True,
        "desc": "完整 DRFNet (DGSAM + CGDR + AFM)",
        "output_mode": "adaptive_fusion",  # alpha * refined + (1-alpha) * heatmap
    },
}


# =====================================================================
#  配置
# =====================================================================
class Config:
    # 路径
    COCO_ROOT = r"/data/ghaiyan/髋关节影像检测/dataset/coco_format/"
    TRAIN_ANN = os.path.join(COCO_ROOT, "annotations", "train2017.json")
    VAL_ANN = os.path.join(COCO_ROOT, "annotations", "val2017.json")
    TRAIN_IMG_DIR = os.path.join(COCO_ROOT, "train2017", "images")
    VAL_IMG_DIR = os.path.join(COCO_ROOT, "val2017", "images")
    CHECKPOINT_DIR = "./checkpoints_drfnet"
    LOG_DIR = "./logs_drfnet"

    # 模型
    NUM_KEYPOINTS = 19
    BACKBONE = "resnet50"
    INPUT_SIZE = (512, 512)
    HEATMAP_SIZE = 128       # 热图输出尺寸 (与 SimpleBaseline 一致)
    FPN_CHANNELS = 256       # FPN 统一通道数
    DET_HIDDEN = 256         # 检测头隐藏层维度

    # 训练
    EPOCHS = 300
    BATCH_SIZE = 8
    LR_BACKBONE = 1e-4       # 骨干 + FPN 学习率 (微调预训练)
    LR_HEAD = 1e-3           # 新增模块学习率 (从头学习)
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5
    LR_MIN = 1e-6
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 损失权重
    LAMBDA_BOX = 7.5
    LAMBDA_POSE = 12.0
    LAMBDA_KOBJ = 1.0
    LAMBDA_HEATMAP_BASE = 1.0
    LAMBDA_CONSIST = 1.0
    LAMBDA_FINAL = 1.0    # kpt_final 监督损失权重

    # 热图高斯核
    SIGMA_HEATMAP = 2.0
    SIGMA_DGSAM = 2.0

    # 阶段权重调度 (epoch 范围 -> heatmap_scale, consist_weight)
    PHASE_SCHEDULE = [
        (1, 30, 0.1, 0.0),      # 阶段一: 检测优先, 热图弱监督
        (31, 150, 1.0, 1.0),    # 阶段二: 联合训练
        (151, 300, 1.0, 2.0),   # 阶段三: 精修融合, 加强一致性
    ]

    # 其他
    NUM_WORKERS = 2
    SAVE_EVERY = 10
    VAL_EVERY = 5
    GRAD_CLIP = 5.0
    AMP = True                 # 混合精度训练

    # 关键点分组 (用于分组 PCK 分析)
    KPT_GROUPS = {
        "骨盆基准(泪滴/耻骨联合)": [0, 1, 2],               # P1, P2, P3
        "股骨头/颈": [3, 4, 11, 12],                         # P4, P5, P12, P13
        "股骨干": [5, 6, 13, 14],                             # P6, P7, P14, P15
        "臼杯结构": [7, 8, 9, 10, 15, 16, 17, 18],           # P8-P11, P16-P19
    }


cfg = Config()


# =====================================================================
#  数据集
# =====================================================================
class HipKeypointDataset(Dataset):
    """
    COCO 格式的髋关节关键点数据集
    返回: image, keypoints(归一化), bbox(归一化), heatmaps, visibility
    """

    def __init__(self, ann_path, img_dir, input_size=(512, 512),
                 heatmap_size=128, num_keypoints=19, sigma=2.0,
                 use_albumentations=False, is_train=True):
        self.img_dir = img_dir
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.sigma = sigma
        self.use_albumentations = use_albumentations and HAS_ALBUMENTATIONS
        self.is_train = is_train

        with open(ann_path, encoding="utf-8") as f:
            self.coco = json.load(f)

        self.images = self.coco["images"]
        self.annotations = self.coco["annotations"]
        self.img_to_anns = defaultdict(list)
        for ann in self.annotations:
            self.img_to_anns[ann["image_id"]].append(ann)

        # 基础图像变换 (仅归一化)
        self.basic_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 保守数据增强 (与 YOLOv8-Pose / SimpleBaseline 一致)
        if self.use_albumentations and is_train:
            self.albu_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.15, rotate_limit=5,
                    border_mode=0, p=0.5,
                ),
            ], keypoint_params=A.KeypointParams(
                format="xy", remove_invisible=False,
            ), bbox_params=A.BboxParams(
                format="coco", min_area=0, min_visibility=0,
                label_fields=["bbox_labels"],
            ))
        else:
            self.albu_transform = None

        print(f"[数据集] {ann_path}: {len(self.images)} 图, {len(self.annotations)} 标注")

    def __len__(self):
        return len(self.images)

    def _generate_heatmaps(self, kpts_norm, visibility, h, w):
        """从归一化关键点坐标生成热图"""
        heatmaps = np.zeros((self.num_keypoints, h, w), dtype=np.float32)

        for kpt_id in range(self.num_keypoints):
            if visibility[kpt_id] == 0:
                continue

            x_norm, y_norm = kpts_norm[kpt_id]
            hx = x_norm * w
            hy = y_norm * h

            if hx < 1 or hy < 1 or hx >= w - 1 or hy >= h - 1:
                continue

            # 修复: np.ogrid 索引顺序, 明确使用 [0] 和 [1] 避免混淆
            gy, gx = np.ogrid[0:h, 0:w]   # gy: (h,1) y坐标, gx: (1,w) x坐标
            dist2 = (gx - hx) ** 2 + (gy - hy) ** 2
            heatmap = np.exp(-dist2 / (2.0 * self.sigma ** 2))
            if heatmap.max() > 0:
                heatmap /= heatmap.max()
            heatmaps[kpt_id] = heatmap

        return heatmaps

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        # 读取图片
        try:
            from PIL import Image
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] 无法读取图片 {img_path}: {e}")
            image = Image.new("RGB", self.input_size, (0, 0, 0))

        orig_w, orig_h = image.size
        image = image.resize(self.input_size, Image.BILINEAR)

        # 标注
        anns = self.img_to_anns.get(img_id, [])
        if anns:
            ann = anns[0]
            keypoints_raw = list(ann["keypoints"])  # [x, y, v] × 19 (原始像素)
            bbox_raw = list(ann["bbox"])             # [x, y, w, h] (左上角, 原始像素)
        else:
            keypoints_raw = [0.0] * (self.num_keypoints * 3)
            bbox_raw = [0.0, 0.0, float(orig_w), float(orig_h)]

        # 缩放到输入像素空间
        scale_x = self.input_size[1] / orig_w
        scale_y = self.input_size[0] / orig_h
        kpts_pixel = []
        visibility = []
        for i in range(self.num_keypoints):
            x = keypoints_raw[i * 3] * scale_x
            y = keypoints_raw[i * 3 + 1] * scale_y
            v = min(int(keypoints_raw[i * 3 + 2]), 1)  # COCO v: 0/1/2 → clamp to 0/1
            kpts_pixel.append([x, y])
            visibility.append(v)

        # bbox 缩放到输入像素空间 + 边界裁剪
        bx = bbox_raw[0] * scale_x
        by = bbox_raw[1] * scale_y
        bw = bbox_raw[2] * scale_x
        bh = bbox_raw[3] * scale_y

        # 确保 bbox 在图像范围内 (防止 Albumentations 校验失败)
        bx = max(0.0, min(bx, self.input_size[1]))
        by = max(0.0, min(by, self.input_size[0]))
        bw = max(1.0, min(bw, self.input_size[1] - bx))
        bh = max(1.0, min(bh, self.input_size[0] - by))

        # 数据增强 (仅在训练时, 在像素空间操作)
        if self.albu_transform and self.is_train:
            img_np = np.array(image)
            transformed = self.albu_transform(
                image=img_np,
                keypoints=kpts_pixel,
                bboxes=[[bx, by, bw, bh]],
                bbox_labels=[0],
            )
            image = Image.fromarray(transformed["image"])
            new_kpts = transformed["keypoints"]
            new_bboxes = transformed["bboxes"]

            kpts_pixel = new_kpts
            if new_bboxes:
                bx, by, bw, bh = new_bboxes[0]

            # 更新 visibility (增强后越界的关键点标记为不可见)
            for i in range(self.num_keypoints):
                if kpts_pixel[i][0] < 0 or kpts_pixel[i][0] >= self.input_size[1] or \
                   kpts_pixel[i][1] < 0 or kpts_pixel[i][1] >= self.input_size[0]:
                    visibility[i] = 0

        # 归一化到 [0, 1]
        H, W = self.input_size
        kpts_norm = []
        for i in range(self.num_keypoints):
            kpts_norm.append([kpts_pixel[i][0] / W, kpts_pixel[i][1] / H])

        # bbox 归一化: [cx, cy, w, h] in [0, 1]
        bbox_norm = [
            (bx + bw / 2) / W,
            (by + bh / 2) / H,
            bw / W,
            bh / H,
        ]
        # 裁剪 bbox 到 [0, 1]
        bbox_norm = [max(0, min(1, v)) for v in bbox_norm]

        # visibility tensor
        vis_tensor = np.array(visibility, dtype=np.float32)  # 0 or 1

        # 生成热图
        heatmaps = self._generate_heatmaps(kpts_norm, visibility,
                                            self.heatmap_size, self.heatmap_size)

        # 转 tensor
        img_tensor = self.basic_transform(image)
        kpts_tensor = torch.tensor(kpts_norm, dtype=torch.float32)       # [19, 2]
        bbox_tensor = torch.tensor(bbox_norm, dtype=torch.float32)       # [4]
        vis_tensor = torch.tensor(vis_tensor, dtype=torch.float32)       # [19]
        heatmaps_tensor = torch.from_numpy(heatmaps)                     # [19, H_hm, W_hm]

        return {
            "image": img_tensor,
            "keypoints": kpts_tensor,          # [19, 2] 归一化坐标
            "bbox": bbox_tensor,               # [4] 归一化 [cx, cy, w, h]
            "visibility": vis_tensor,           # [19]
            "heatmaps": heatmaps_tensor,        # [19, H_hm, W_hm]
            "img_id": img_id,
        }


# =====================================================================
#  ResNet50 骨干 (提取多尺度特征)
# =====================================================================
class ResNetBackbone(nn.Module):
    """
    ResNet50 骨干, 提取 C3, C4, C5 多尺度特征
    输出:
      C3: [B, 512, 64, 64]  (stride 8)
      C4: [B, 1024, 32, 32] (stride 16)
      C5: [B, 2048, 16, 16] (stride 32)
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = resnet50(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1   # 256ch, stride 4
        self.layer2 = resnet.layer2   # C3: 512ch, stride 8
        self.layer3 = resnet.layer3   # C4: 1024ch, stride 16
        self.layer4 = resnet.layer4   # C5: 2048ch, stride 32

    def forward(self, x):
        x = self.stem(x)        # [B, 64, 128, 128]
        x = self.layer1(x)     # [B, 256, 128, 128]
        c3 = self.layer2(x)     # [B, 512, 64, 64]
        c4 = self.layer3(c3)   # [B, 1024, 32, 32]
        c5 = self.layer4(c4)   # [B, 2048, 16, 16]
        return c3, c4, c5


# =====================================================================
#  FPN 颈部 (特征金字塔网络)
# =====================================================================
class FPN(nn.Module):
    """
    自顶向下特征金字塔
    输入: C3(512ch), C4(1024ch), C5(2048ch)
    输出: P3(256ch, 64x64), P4(256ch, 32x32), P5(256ch, 16x16)
    """

    def __init__(self, in_channels=(512, 1024, 2048), out_channels=256):
        super().__init__()
        # 侧连接: 1x1 卷积降维
        self.lat5 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.lat4 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lat3 = nn.Conv2d(in_channels[0], out_channels, 1)
        # 平滑层: 3x3 卷积
        self.smooth5 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, c3, c4, c5):
        # 侧连接降维
        p5 = self.lat5(c5)    # [B, 256, 16, 16]
        p4 = self.lat4(c4)    # [B, 256, 32, 32]
        p3 = self.lat3(c3)    # [B, 256, 64, 64]

        # 自顶向下融合
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")

        # 3x3 平滑
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)

        return p3, p4, p5


# =====================================================================
#  检测分支头部 (解耦头: bbox + 关键点 + 可见性)
# =====================================================================
class DetectionHead(nn.Module):
    """
    单目标解耦检测头 (骨盆-股骨区域)
    输入: 多尺度 FPN 特征 P3, P4, P5 (GAP 后拼接)
    输出:
      bbox: [B, 4] 归一化 [cx, cy, w, h]
      keypoints: [B, 19, 2] 归一化坐标
      kobj: [B, 19] 可见性 logit
    """

    def __init__(self, fpn_channels=256, num_keypoints=19, hidden=256):
        super().__init__()
        total_in = fpn_channels * 3  # GAP(P3) + GAP(P4) + GAP(P5)

        self.shared = nn.Sequential(
            nn.Linear(total_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        self.box_head = nn.Linear(hidden, 4)
        self.kpt_head = nn.Linear(hidden, num_keypoints * 2)
        self.kobj_head = nn.Linear(hidden, num_keypoints)

        self._init_weights()

    def _init_weights(self):
        for m in [self.box_head, self.kpt_head, self.kobj_head]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        for m in self.shared:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, p3, p4, p5):
        B = p3.shape[0]
        # 多尺度全局平均池化
        g3 = F.adaptive_avg_pool2d(p3, 1).view(B, -1)   # [B, 256]
        g4 = F.adaptive_avg_pool2d(p4, 1).view(B, -1)   # [B, 256]
        g5 = F.adaptive_avg_pool2d(p5, 1).view(B, -1)   # [B, 256]

        feat = torch.cat([g3, g4, g5], dim=1)            # [B, 768]
        feat = self.shared(feat)                          # [B, 256]

        bbox = self.box_head(feat)                        # [B, 4]
        kpt = self.kpt_head(feat).view(B, -1, 2)        # [B, 19, 2]
        kobj = self.kobj_head(feat)                       # [B, 19]

        return bbox, kpt, kobj


# =====================================================================
#  热图解码器 (SimpleBaseline 风格, 3 层反卷积)
# =====================================================================
class HeatmapDecoder(nn.Module):
    """
    反卷积热图解码器
    输入: P5 [B, 256, 16, 16]
    输出: heatmap [B, 19, 128, 128]
    """

    def __init__(self, in_channels=256, num_keypoints=19, num_filters=256):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, num_filters, 4, 2, 1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, num_filters, 4, 2, 1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, num_filters, 4, 2, 1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(num_filters, num_keypoints, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.deconv.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.kaiming_normal_(self.final.weight, mode="fan_out", nonlinearity="relu")
        if self.final.bias is not None:
            nn.init.constant_(self.final.bias, 0)

    def forward(self, x):
        x = self.deconv(x)     # [B, 256, 128, 128]
        x = self.final(x)      # [B, 19, 128, 128]  raw logits
        x = torch.sigmoid(x)   # 约束到 [0, 1]，与 target 高斯热图同范围
        return x


# =====================================================================
#  DGSAM: 检测引导空间注意力
# =====================================================================
class DGSAM(nn.Module):
    """
    Detection-Guided Spatial Attention Module
    从 bbox 生成 2D 高斯空间注意力掩膜
    A(x,y) = exp( -((x-cx)^2/w^2 + (y-cy)^2/h^2) / (2*sigma^2) )
    """

    def __init__(self, sigma=2.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, bbox, feat_h, feat_w):
        """
        Args:
            bbox: [B, 4] 归一化 [cx, cy, w, h]
            feat_h, feat_w: 特征图空间尺寸
        Returns:
            A: [B, 1, feat_h, feat_w] 注意力掩膜
        """
        B = bbox.shape[0]
        device = bbox.device

        cx = bbox[:, 0].view(B, 1, 1, 1)
        cy = bbox[:, 1].view(B, 1, 1, 1)
        w = bbox[:, 2].view(B, 1, 1, 1)
        h = bbox[:, 3].view(B, 1, 1, 1)

        # 特征图坐标网格 (归一化到 [0, 1])
        grid_y = torch.arange(feat_h, device=device, dtype=torch.float32) / max(feat_h - 1, 1)
        grid_x = torch.arange(feat_w, device=device, dtype=torch.float32) / max(feat_w - 1, 1)
        gx, gy = torch.meshgrid(grid_x, grid_y)   # [feat_w, feat_h]
        gx = gx.unsqueeze(0).unsqueeze(0)            # [1, 1, feat_w, feat_h]
        gy = gy.unsqueeze(0).unsqueeze(0)

        dist2 = ((gx - cx) / (w + 1e-6)).pow(2) + \
                ((gy - cy) / (h + 1e-6)).pow(2)
        A = torch.exp(-dist2 / (2 * self.sigma ** 2))

        return A  # [B, 1, feat_h, feat_w]


# =====================================================================
#  CGDR: 置信度引导检测精修
# =====================================================================
class CGDR(nn.Module):
    """
    Confidence-Guided Detection Refinement
    热图置信度 → MLP嵌入 → 与粗略坐标拼接 → 卷积生成修正量
    K_refined = K_direct + Delta_K
    """

    def __init__(self, num_keypoints=19, embed_dim=64):
        super().__init__()
        self.num_keypoints = num_keypoints

        # 置信度嵌入: [19] → [19 × 64]
        self.conf_mlp = nn.Sequential(
            nn.Linear(num_keypoints, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_keypoints * embed_dim),
        )

        # 修正量卷积: 拼接 [64 + 2] → 2
        self.refine_conv = nn.Sequential(
            nn.Conv2d(embed_dim + 2, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, kpt_direct, confidence):
        """
        Args:
            kpt_direct: [B, 19, 2] 检测分支粗略坐标 (归一化)
            confidence: [B, 19] 热图峰值置信度
        Returns:
            kpt_refined: [B, 19, 2] 精修后坐标
        """
        B = kpt_direct.shape[0]

        # 置信度嵌入
        c_embed = self.conf_mlp(confidence)            # [B, 19*64]
        c_embed = c_embed.view(B, self.num_keypoints, 64, 1, 1)  # [B, 19, 64, 1, 1]

        # 与粗略坐标拼接: [B, 19, 64+2, 1, 1]
        kpt_reshaped = kpt_direct.unsqueeze(-1).unsqueeze(-1)     # [B, 19, 2, 1, 1]
        combined = torch.cat([c_embed, kpt_reshaped], dim=2)      # [B, 19, 66, 1, 1]

        # 修正量 (对每个关键点独立卷积)
        # [B, 19, 66, 1, 1] → reshape to [B*19, 66, 1, 1] → conv → [B*19, 2, 1, 1]
        combined_flat = combined.view(B * self.num_keypoints, 66, 1, 1)
        delta_k = self.refine_conv(combined_flat)                  # [B*19, 2, 1, 1]
        delta_k = delta_k.view(B, self.num_keypoints, 2)           # [B, 19, 2]

        kpt_refined = kpt_direct + delta_k

        return kpt_refined


# =====================================================================
#  AFM: 自适应融合模块
# =====================================================================
class AFM(nn.Module):
    """
    Adaptive Fusion Module
    逐关键点学习 alpha 权重:
      alpha = sigmoid( MLP( Concat(C, Delta) ) )
      K_final = alpha * K_refined + (1-alpha) * K_heatmap
    """

    def __init__(self, num_keypoints=19, hidden=32):
        super().__init__()
        self.alpha_mlp = nn.Sequential(
            nn.Linear(num_keypoints * 2, hidden),  # C(19) + Delta(19)
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_keypoints),
            nn.Sigmoid(),  # alpha in (0, 1)
        )
        self._init_weights()

    def _init_weights(self):
        # alpha MLP 偏置初始化为 0 → sigmoid(0)=0.5 → 初始等权融合
        for m in self.alpha_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, kpt_refined, kpt_heatmap, confidence):
        """
        Args:
            kpt_refined: [B, 19, 2] CGDR 精修后的检测坐标
            kpt_heatmap: [B, 19, 2] 热图解码坐标
            confidence: [B, 19] 热图峰值置信度
        Returns:
            kpt_final: [B, 19, 2] 自适应融合后坐标
            alpha: [B, 19] 融合权重
        """
        # 一致性度量
        delta = torch.norm(kpt_refined - kpt_heatmap, dim=-1)   # [B, 19]

        # 自适应权重
        alpha_input = torch.cat([confidence, delta], dim=-1)     # [B, 38]
        alpha = self.alpha_mlp(alpha_input)                       # [B, 19]

        # 加权融合
        alpha_w = alpha.unsqueeze(-1)                              # [B, 19, 1]
        kpt_final = alpha_w * kpt_refined + (1 - alpha_w) * kpt_heatmap

        return kpt_final, alpha


# =====================================================================
#  DRFNet: 完整模型
# =====================================================================
class DRFNet(nn.Module):
    """
    Dual-Representation Fusion Network
    共享骨干 + FPN + 双分支(Detection + Heatmap) + 交叉注意力(DGSAM/CGDR) + 自适应融合(AFM)
    """

    def __init__(self, ablation="F", num_keypoints=19, pretrained_backbone=True):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.ablation = ablation
        ab_cfg = ABLATION_CONFIGS[ablation]
        self.use_dgsam = ab_cfg["use_dgsam"]
        self.use_cgdr = ab_cfg["use_cgdr"]
        self.use_afm = ab_cfg["use_afm"]
        self.output_mode = ab_cfg["output_mode"]

        # 1. 共享骨干
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)

        # 2. FPN 颈部
        self.fpn = FPN(
            in_channels=(512, 1024, 2048),
            out_channels=cfg.FPN_CHANNELS,
        )

        # 3. 检测分支头部
        self.det_head = DetectionHead(
            fpn_channels=cfg.FPN_CHANNELS,
            num_keypoints=num_keypoints,
            hidden=cfg.DET_HIDDEN,
        )

        # 4. 热图解码器
        self.heatmap_decoder = HeatmapDecoder(
            in_channels=cfg.FPN_CHANNELS,
            num_keypoints=num_keypoints,
        )

        # 5. DGSAM (无参数模块)
        self.dgsam = DGSAM(sigma=cfg.SIGMA_DGSAM)

        # 6. CGDR (if enabled)
        if self.use_cgdr:
            self.cgdr = CGDR(num_keypoints=num_keypoints)

        # 7. AFM (if enabled)
        if self.use_afm:
            self.afm = AFM(num_keypoints=num_keypoints)

        print(f"[DRFNet] 消融配置: {ablation}")
        print(f"  DGSAM: {self.use_dgsam} | CGDR: {self.use_cgdr} | AFM: {self.use_afm}")
        print(f"  输出模式: {self.output_mode}")
        print(f"  总参数: {sum(p.numel() for p in self.parameters()) / 1e6:.1f}M")

    def _heatmap_to_coords(self, heatmap):
        """
        从热图提取关键点坐标和置信度
        Args:
            heatmap: [B, N, H, W]
        Returns:
            coords: [B, N, 2] 归一化坐标 [0, 1]
            confidence: [B, N] 峰值高度
        """
        B, N, H, W = heatmap.shape
        heat_flat = heatmap.view(B, N, -1)
        max_idx = heat_flat.argmax(dim=2)       # [B, N]
        max_val = heat_flat.max(dim=2).values    # [B, N]

        y = (max_idx // W).float() / max(H - 1, 1)   # 归一化到 [0, 1]
        x = (max_idx % W).float() / max(W - 1, 1)

        coords = torch.stack([x, y], dim=2)     # [B, N, 2]
        return coords, max_val

    def forward(self, x):
        B = x.shape[0]

        # 1. 共享骨干 + FPN
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.fpn(c3, c4, c5)

        # 2. 检测分支
        bbox_pred, kpt_direct, kobj_pred = self.det_head(p3, p4, p5)
        # bbox_pred: [B, 4], kpt_direct: [B, 19, 2], kobj_pred: [B, 19]

        # 3. DGSAM 空间注意力
        if self.use_dgsam:
            feat_h, feat_w = p5.shape[-2:]
            spatial_attn = self.dgsam(bbox_pred.detach(), feat_h, feat_w)
            p5_attended = p5 * spatial_attn
        else:
            p5_attended = p5

        # 4. 热图解码
        heatmap = self.heatmap_decoder(p5_attended)     # [B, 19, 128, 128]

        # 5. 从热图提取坐标和置信度
        kpt_heatmap, confidence = self._heatmap_to_coords(heatmap)
        # kpt_heatmap: [B, 19, 2], confidence: [B, 19]

        # 6. CGDR 置信度引导精修
        if self.use_cgdr:
            kpt_refined = self.cgdr(kpt_direct, confidence)
        else:
            kpt_refined = kpt_direct

        # 7. 融合输出
        if self.output_mode == "heatmap":
            # C, D: 直接使用热图分支输出
            kpt_final = kpt_heatmap
            alpha = torch.ones(B, self.num_keypoints, device=x.device) * 0.0
        elif self.output_mode == "equal_fusion":
            # E: 等权融合
            kpt_final = 0.5 * kpt_refined + 0.5 * kpt_heatmap
            alpha = torch.ones(B, self.num_keypoints, device=x.device) * 0.5
        elif self.output_mode == "adaptive_fusion":
            # F: AFM 自适应融合
            kpt_final, alpha = self.afm(kpt_refined, kpt_heatmap, confidence)
        else:
            kpt_final = kpt_heatmap
            alpha = torch.zeros(B, self.num_keypoints, device=x.device)

        return {
            "bbox": bbox_pred,
            "kpt_direct": kpt_direct,
            "kobj": kobj_pred,
            "heatmap": heatmap,
            "confidence": confidence,
            "kpt_heatmap": kpt_heatmap,
            "kpt_refined": kpt_refined,
            "kpt_final": kpt_final,
            "alpha": alpha,
        }


# =====================================================================
#  损失函数
# =====================================================================
class BoxLoss(nn.Module):
    """CIoU 边界框损失"""

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 4] (cx, cy, w, h) 归一化
            target: [B, 4] (cx, cy, w, h) 归一化
        """
        # 转为 (x1, y1, x2, y2)
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        tgt_x1 = target[:, 0] - target[:, 2] / 2
        tgt_y1 = target[:, 1] - target[:, 3] / 2
        tgt_x2 = target[:, 0] + target[:, 2] / 2
        tgt_y2 = target[:, 1] + target[:, 3] / 2

        # IoU
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        tgt_area = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)
        union_area = (pred_area + tgt_area - inter_area).clamp(min=1e-6)
        iou = (inter_area / union_area).clamp(0, 1)

        # 中心距离
        cx_dist = (pred[:, 0] - target[:, 0]) ** 2
        cy_dist = (pred[:, 1] - target[:, 1]) ** 2
        c2 = (torch.max(pred_x2, tgt_x2) - torch.min(pred_x1, tgt_x1)) ** 2 + \
             (torch.max(pred_y2, tgt_y2) - torch.min(pred_y1, tgt_y1)) ** 2 + 1e-6
        rho2 = cx_dist + cy_dist
        diou = rho2 / c2

        # 宽高比一致性
        v = (4 / (math.pi ** 2)) * \
            (torch.atan(target[:, 2] / (target[:, 3] + 1e-6)) -
             torch.atan(pred[:, 2] / (pred[:, 3] + 1e-6))) ** 2
        with torch.no_grad():
            alpha_ciou = v / (1 - iou + v + 1e-6)

        ciou = iou - diou - alpha_ciou * v
        loss = 1 - ciou
        return loss.mean()


class KeypointLoss(nn.Module):
    """SmoothL1 关键点坐标损失 (带可见性掩码)"""

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, visibility):
        """
        Args:
            pred: [B, 19, 2] 归一化
            target: [B, 19, 2] 归一化
            visibility: [B, 19] 0 or 1
        """
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta,
                            0.5 * diff ** 2 / self.beta,
                            diff - 0.5 * self.beta)
        loss = loss.sum(dim=-1)                          # [B, 19]
        loss = (loss * visibility).sum() / (visibility.sum() + 1e-6)
        return loss


class KobjLoss(nn.Module):
    """BCE 关键点可见性损失"""

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 19] logits
            target: [B, 19] 0 or 1
        """
        return F.binary_cross_entropy_with_logits(pred, target)


class HeatmapLoss(nn.Module):
    """MSE 热图损失 (带可见性掩码)"""

    def forward(self, pred, target, visibility):
        """
        Args:
            pred: [B, 19, H, W]
            target: [B, 19, H, W]
            visibility: [B, 19]
        """
        B, N, H, W = pred.shape
        vis_mask = visibility.view(B, N, 1, 1)     # [B, 19, 1, 1]
        loss = ((pred - target) ** 2 * vis_mask).sum() / (vis_mask.sum() * H * W + 1e-6)
        return loss


class ConsistencyLoss(nn.Module):
    """SmoothL1 两分支一致性正则损失"""

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, kpt_refined, kpt_heatmap, visibility):
        """
        Args:
            kpt_refined: [B, 19, 2]
            kpt_heatmap: [B, 19, 2]
            visibility: [B, 19]
        """
        diff = torch.abs(kpt_refined - kpt_heatmap)
        loss = torch.where(diff < self.beta,
                            0.5 * diff ** 2 / self.beta,
                            diff - 0.5 * self.beta)
        loss = loss.sum(dim=-1)                           # [B, 19]
        loss = (loss * visibility).sum() / (visibility.sum() + 1e-6)
        return loss


class MultiTaskLoss(nn.Module):
    """多任务联合损失 (含阶段性权重调度)"""

    def __init__(self, cfg):
        super().__init__()
        self.box_loss = BoxLoss()
        self.kpt_loss = KeypointLoss()
        self.kobj_loss = KobjLoss()
        self.heatmap_loss = HeatmapLoss()
        self.consist_loss = ConsistencyLoss()
        self.cfg = cfg

    def get_weights(self, epoch):
        """根据训练阶段返回损失权重"""
        for start, end, hm_scale, consist_w in self.cfg.PHASE_SCHEDULE:
            if start <= epoch <= end:
                return {
                    "lambda_box": self.cfg.LAMBDA_BOX,
                    "lambda_pose": self.cfg.LAMBDA_POSE,
                    "lambda_kobj": self.cfg.LAMBDA_KOBJ,
                    "lambda_heatmap": self.cfg.LAMBDA_HEATMAP_BASE * hm_scale,
                    "lambda_consist": self.cfg.LAMBDA_CONSIST * consist_w,
                    "lambda_final": self.cfg.LAMBDA_FINAL,
                }
        # 默认: 最后阶段
        return {
            "lambda_box": self.cfg.LAMBDA_BOX,
            "lambda_pose": self.cfg.LAMBDA_POSE,
            "lambda_kobj": self.cfg.LAMBDA_KOBJ,
            "lambda_heatmap": self.cfg.LAMBDA_HEATMAP_BASE,
            "lambda_consist": self.cfg.LAMBDA_CONSIST * 2.0,
            "lambda_final": self.cfg.LAMBDA_FINAL,
        }

    def forward(self, outputs, targets, epoch):
        """
        Args:
            outputs: DRFNet forward 输出
            targets: batch dict (keypoints, bbox, visibility, heatmaps)
            epoch: 当前轮次
        Returns:
            total_loss, loss_dict
        """
        weights = self.get_weights(epoch)

        bbox_pred = outputs["bbox"]
        kpt_direct = outputs["kpt_direct"]
        kobj_pred = outputs["kobj"]
        heatmap_pred = outputs["heatmap"]
        kpt_refined = outputs["kpt_refined"]
        kpt_heatmap = outputs["kpt_heatmap"]

        gt_kpts = targets["keypoints"]        # [B, 19, 2]
        gt_bbox = targets["bbox"]             # [B, 4]
        gt_vis = targets["visibility"]         # [B, 19]
        gt_heatmaps = targets["heatmaps"]       # [B, 19, H, W]

        # 各项损失
        l_box = self.box_loss(bbox_pred, gt_bbox)
        l_pose = self.kpt_loss(kpt_direct, gt_kpts, gt_vis)
        l_kobj = self.kobj_loss(kobj_pred, gt_vis)
        l_heatmap = self.heatmap_loss(heatmap_pred, gt_heatmaps, gt_vis)
        l_consist = self.consist_loss(kpt_refined, kpt_heatmap, gt_vis)
        l_final = self.kpt_loss(outputs["kpt_final"], gt_kpts, gt_vis)  # kpt_final 监督损失

        # 加权求和
        total_loss = (
            weights["lambda_box"] * l_box
            + weights["lambda_pose"] * l_pose
            + weights["lambda_kobj"] * l_kobj
            + weights["lambda_heatmap"] * l_heatmap
            + weights["lambda_consist"] * l_consist
            + weights["lambda_final"] * l_final
        )

        loss_dict = {
            "total": total_loss.item(),
            "box": l_box.item(),
            "pose": l_pose.item(),
            "kobj": l_kobj.item(),
            "heatmap": l_heatmap.item(),
            "consist": l_consist.item(),
            "final": l_final.item(),
            "w_hm": weights["lambda_heatmap"],
            "w_cons": weights["lambda_consist"],
        }

        return total_loss, loss_dict


# =====================================================================
#  评估指标
# =====================================================================
def compute_pckh(pred_kpts, gt_kpts, valid_mask, threshold=0.5):
    """
    PCK@threshold 指标
    以外接矩形对角线为归一化参考 (髋关节无头部结构)
    Args:
        pred_kpts: [B, N, 2] 像素坐标
        gt_kpts: [B, N, 2] 像素坐标
        valid_mask: [B, N] bool
        threshold: 归一化阈值
    Returns:
        pck: float [0, 1]
    """
    B, N, _ = gt_kpts.shape
    total_correct = 0
    total_valid = 0

    for b in range(B):
        mask_b = valid_mask[b]
        if mask_b.sum() == 0:
            continue

        gt_vis = gt_kpts[b][mask_b]
        pred_vis = pred_kpts[b][mask_b]

        x_min, y_min = gt_vis.min(dim=0).values
        x_max, y_max = gt_vis.max(dim=0).values
        ref_size = max(torch.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2).item(), 1e-6)

        dist = torch.norm(pred_vis - gt_vis, dim=1)
        correct = (dist < threshold * ref_size).float()
        total_correct += correct.sum().item()
        total_valid += mask_b.sum().item()

    return total_correct / total_valid if total_valid > 0 else 0.0


def compute_epe(pred_kpts, gt_kpts, valid_mask):
    """
    EPE (End Point Error): 平均端点误差 (像素)
    """
    B, N, _ = gt_kpts.shape
    total_epe = 0.0
    total_valid = 0

    for b in range(B):
        mask_b = valid_mask[b]
        if mask_b.sum() == 0:
            continue
        gt_vis = gt_kpts[b][mask_b]
        pred_vis = pred_kpts[b][mask_b]
        total_epe += torch.norm(pred_vis - gt_vis, dim=1).sum().item()
        total_valid += mask_b.sum().item()

    return total_epe / total_valid if total_valid > 0 else 0.0


def evaluate(model, dataloader, device, input_size, num_keypoints=19, kpt_groups=None):
    """
    综合评估: PCK@0.5, PCK@0.1, EPE, 分组 PCK
    """
    model.eval()
    H, W = input_size

    all_pred = []
    all_gt = []
    all_vis = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            gt_kpts = batch["keypoints"].to(device)         # [B, 19, 2] 归一化
            visibility = batch["visibility"].to(device)      # [B, 19]

            outputs = model(images)
            kpt_final = outputs["kpt_final"]                 # [B, 19, 2] 归一化

            # 反归一化到像素空间
            kpt_pred_px = kpt_final.clone()
            kpt_pred_px[:, :, 0] *= W
            kpt_pred_px[:, :, 1] *= H

            kpt_gt_px = gt_kpts.clone()
            kpt_gt_px[:, :, 0] *= W
            kpt_gt_px[:, :, 1] *= H

            all_pred.append(kpt_pred_px.cpu())
            all_gt.append(kpt_gt_px.cpu())
            all_vis.append(visibility.cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)
    all_vis = torch.cat(all_vis, dim=0)

    valid_mask = all_vis > 0

    metrics = {}
    metrics["PCK@0.5"] = compute_pckh(all_pred, all_gt, valid_mask, 0.5)
    metrics["PCK@0.1"] = compute_pckh(all_pred, all_gt, valid_mask, 0.1)
    metrics["EPE"] = compute_epe(all_pred, all_gt, valid_mask)

    # 分组 PCK 分析
    if kpt_groups:
        for group_name, indices in kpt_groups.items():
            group_mask = torch.zeros(all_vis.shape[1], dtype=torch.bool)
            for idx in indices:
                group_mask[idx] = True
            group_vis = valid_mask & group_mask.unsqueeze(0)
            if group_vis.sum() > 0:
                metrics[f"PCK@0.5_{group_name}"] = compute_pckh(
                    all_pred, all_gt, group_vis, 0.5
                )

    return metrics


# =====================================================================
#  权重加载工具
# =====================================================================
def warm_start_load(model, checkpoint_path, device):
    """
    跨消融配置 Warm-Start 加载权重
    只加载匹配的权重，跳过不匹配的（新增模块保持随机初始化）
    """
    print(f"\n{'='*60}")
    print(f"  Warm-Start 加载: {checkpoint_path}")
    print(f"{'='*60}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    src_state = checkpoint["model"]   # 源消融配置的权重
    dst_state = model.state_dict()     # 当前模型权重

    matched = 0
    unused  = 0
    new_mod  = 0

    # 1) 从源权重加载所有匹配的 key
    for k, v in src_state.items():
        if k in dst_state and v.shape == dst_state[k].shape:
            dst_state[k] = v
            matched += 1
        else:
            unused += 1

    # 2) 统计当前模型中有多少 key 没有从源权重中加载
    for k in dst_state.keys():
        if k not in src_state:
            new_mod += 1

    model.load_state_dict(dst_state, strict=False)

    print(f"  匹配加载: {matched} 层")
    print(f"  源多余 (形状不匹配/不存在): {unused} 层")
    print(f"  新增模块 (保持随机初始化): {new_mod} 层")
    print(f"{'='*60}\n")

    # 返回 checkpoint 中的 best_pck（如果有）
    return checkpoint.get("best_pck", 0.0), checkpoint.get("epoch", 0)


# =====================================================================
#  训练循环
# =====================================================================
def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, writer):
    model.train()
    total_loss = 0.0
    loss_accum = defaultdict(float)

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        targets = {
            "keypoints": batch["keypoints"].to(device),
            "bbox": batch["bbox"].to(device),
            "visibility": batch["visibility"].to(device),
            "heatmaps": batch["heatmaps"].to(device),
        }

        optimizer.zero_grad()

        if cfg.AMP:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss, loss_dict = criterion(outputs, targets, epoch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            if k not in ("w_hm", "w_cons"):
                loss_accum[k] += v

        if batch_idx % 20 == 0:
            phase = "检测优先" if epoch <= 30 else ("联合训练" if epoch <= 150 else "精修融合")
            print(f"  [Ep{epoch}][{batch_idx}/{len(dataloader)}] "
                  f"Loss={loss.item():.4f} ({phase}) "
                  f"box={loss_dict['box']:.2f} pose={loss_dict['pose']:.2f} "
                  f"kobj={loss_dict['kobj']:.2f} hm={loss_dict['heatmap']:.2f} "
                  f"cons={loss_dict['consist']:.2f} final={loss_dict['final']:.2f}")

    n_batch = len(dataloader)
    avg_loss = total_loss / n_batch
    for k in loss_accum:
        loss_accum[k] /= n_batch

    if writer:
        writer.add_scalar("train/total_loss", avg_loss, epoch)
        for k, v in loss_accum.items():
            if k not in ("w_hm", "w_cons"):
                writer.add_scalar(f"train/{k}", v, epoch)

    return avg_loss, loss_accum


def val_epoch(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = {
                "keypoints": batch["keypoints"].to(device),
                "bbox": batch["bbox"].to(device),
                "visibility": batch["visibility"].to(device),
                "heatmaps": batch["heatmaps"].to(device),
            }

            if cfg.AMP:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss, loss_dict = criterion(outputs, targets, epoch)
            else:
                outputs = model(images)
                loss, loss_dict = criterion(outputs, targets, epoch)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # 评估 PCK / EPE / 分组
    metrics = evaluate(model, dataloader, device, cfg.INPUT_SIZE,
                      num_keypoints=cfg.NUM_KEYPOINTS,
                      kpt_groups=cfg.KPT_GROUPS)

    if writer:
        writer.add_scalar("val/loss", avg_loss, epoch)
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

    return avg_loss, metrics


# =====================================================================
#  主入口
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="DRFNet 消融实验训练")
    parser.add_argument("--ablation", type=str, default="F", choices=["C", "D", "E", "F"],
                        help="消融配置: C(无DGSAM) D(+DGSAM) E(+CGDR等权) F(完整)")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr_backbone", type=float, default=cfg.LR_BACKBONE)
    parser.add_argument("--lr_head", type=float, default=cfg.LR_HEAD)
    parser.add_argument("--device", type=str, default=cfg.DEVICE)
    parser.add_argument("--amp", type=bool, default=cfg.AMP)
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练 (同消融配置)")
    parser.add_argument("--warm_start_from", type=str, default=None,
                        help="Warm-start: 从另一个消融配置的best.pth加载权重 (跨配置)")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch
    cfg.AMP = args.amp

    ablation = args.ablation
    ab_desc = ABLATION_CONFIGS[ablation]["desc"]

    # 创建消融专属目录
    ckpt_dir = os.path.join(cfg.CHECKPOINT_DIR, f"ablation_{ablation}")
    log_dir = os.path.join(cfg.LOG_DIR, f"ablation_{ablation}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"  DRFNet 消融实验: {ablation} - {ab_desc}")
    print(f"{'='*70}")
    print(f"  设备: {device}")
    print(f"  批量: {cfg.BATCH_SIZE} | 轮数: {cfg.EPOCHS}")
    print(f"  混合精度: {cfg.AMP}")
    print(f"  骨干LR: {args.lr_backbone} | 头部LR: {args.lr_head}")
    print(f"  检查点: {ckpt_dir}")
    print(f"  日志: {log_dir}")
    print(f"{'='*70}")

    # 加载数据
    train_dataset = HipKeypointDataset(
        cfg.TRAIN_ANN, cfg.TRAIN_IMG_DIR,
        input_size=cfg.INPUT_SIZE,
        heatmap_size=cfg.HEATMAP_SIZE,
        num_keypoints=cfg.NUM_KEYPOINTS,
        sigma=cfg.SIGMA_HEATMAP,
        use_albumentations=HAS_ALBUMENTATIONS,
        is_train=True,
    )
    val_dataset = HipKeypointDataset(
        cfg.VAL_ANN, cfg.VAL_IMG_DIR,
        input_size=cfg.INPUT_SIZE,
        heatmap_size=cfg.HEATMAP_SIZE,
        num_keypoints=cfg.NUM_KEYPOINTS,
        sigma=cfg.SIGMA_HEATMAP,
        use_albumentations=False,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )

    # 创建模型
    model = DRFNet(
        ablation=ablation,
        num_keypoints=cfg.NUM_KEYPOINTS,
        pretrained_backbone=True,
    ).to(device)

    # 分层学习率
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name or "fpn" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)

    # 学习率调度: warmup → cosine
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=cfg.WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS - cfg.WARMUP_EPOCHS,
                                eta_min=cfg.LR_MIN)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.WARMUP_EPOCHS])

    # 损失函数
    criterion = MultiTaskLoss(cfg)

    # GradScaler (AMP)
    scaler = torch.amp.GradScaler("cuda") if cfg.AMP else None

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    start_epoch = 1
    best_pck = 0.0

    if args.resume:
        # 模式1: 完全恢复 (同消融配置)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_pck = checkpoint.get("best_pck", 0.0)
        print(f"[恢复] epoch {start_epoch}, 最佳 PCK={best_pck:.4f}")
    elif args.warm_start_from:
        # 模式2: Warm-Start (跨消融配置)
        best_pck, src_epoch = warm_start_load(model, args.warm_start_from, device)
        start_epoch = 1   # 从头计 epoch
        print(f"[Warm-Start] 从 {args.warm_start_from}")
        print(f"  源模型 epoch={src_epoch}, best_PCK={best_pck:.4f}")
        print(f"  将从 epoch 1 开始重新训练 (不恢复 optimizer)")
        print(f"  提示: 建议设置 --lr_head=5e-4 小幅微调新模块\n")

    # 训练循环
    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        lr_str = f"backbone_lr={scheduler.get_last_lr()[0]:.2e}"
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.EPOCHS}, {lr_str}")
        if epoch <= 30:
            print("阶段: 检测优先 (hm_scale=0.1, consist=0)")
        elif epoch <= 150:
            print("阶段: 联合训练 (hm_scale=1.0, consist=1.0)")
        else:
            print("阶段: 精修融合 (hm_scale=1.0, consist=2.0)")
        print(f"{'='*60}")

        # 训练
        train_loss, train_loss_dict = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, writer
        )

        # 验证
        if epoch % cfg.VAL_EVERY == 0 or epoch == cfg.EPOCHS:
            val_loss, val_metrics = val_epoch(
                model, val_loader, criterion, device, epoch, writer
            )

            pck_05 = val_metrics.get("PCK@0.5", 0.0)
            pck_01 = val_metrics.get("PCK@0.1", 0.0)
            epe = val_metrics.get("EPE", 0.0)

            print(f"\n[VAL] Loss={val_loss:.4f} | "
                  f"PCK@0.5={pck_05:.4f} | PCK@0.1={pck_01:.4f} | EPE={epe:.1f}px")

            # 分组 PCK
            for k, v in val_metrics.items():
                if k.startswith("PCK@0.5_"):
                    print(f"  {k}: {v:.4f}")

            # 保存最佳模型 (跳过异常情况)
            if math.isnan(val_loss) or math.isnan(pck_05):
                print(f"[SKIP] val_loss 或 PCK 为 NaN，跳过保存")
            elif pck_05 > best_pck:
                best_pck = pck_05
                torch.save({
                    "epoch": epoch,
                    "ablation": ablation,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_pck": best_pck,
                    "metrics": val_metrics,
                    "config": {
                        "ablation": ablation,
                        "use_dgsam": ABLATION_CONFIGS[ablation]["use_dgsam"],
                        "use_cgdr": ABLATION_CONFIGS[ablation]["use_cgdr"],
                        "use_afm": ABLATION_CONFIGS[ablation]["use_afm"],
                        "output_mode": ABLATION_CONFIGS[ablation]["output_mode"],
                    },
                }, os.path.join(ckpt_dir, "best.pth"))
                print(f"[SAVE] 新最佳! PCK@0.5={best_pck:.4f} -> {ckpt_dir}/best.pth")

        # 定期保存
        if epoch % cfg.SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "ablation": ablation,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_pck": best_pck,
            }, os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

        scheduler.step()

    writer.close()
    print(f"\n{'='*70}")
    print(f"  消融 {ablation} 训练完成!")
    print(f"  最佳 PCK@0.5 = {best_pck:.4f}")
    print(f"  模型保存于: {ckpt_dir}/best.pth")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
