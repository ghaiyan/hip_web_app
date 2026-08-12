#!/usr/bin/env python3
"""
SimpleBaseline 热图回归训练脚本 (方案三: 纯 PyTorch)
======================================================
基于 ResNet + 反卷积的经典关键点检测方法。
不依赖 mmpose/ultralytics，仅需 PyTorch。

论文: Simple Baselines for Human Pose Estimation (ECCV 2018)
架构: ResNet 骨干 → 三层转置卷积 → 热图输出

环境准备:
  pip install torch torchvision albumentations tensorboard pycocotools

训练命令:
  python train_simplebaseline.py --mode train
  python train_simplebaseline.py --mode val --resume checkpoints/best.pth
  python train_simplebaseline.py --mode predict --image path/to/img.jpg
"""

import os
import sys
import math
import json
import copy
import random
import argparse
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
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights

# 可选: albumentations 做更强的数据增强
try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[WARN] albumentations 未安装，将使用 torchvision 内置增强")
    print("[WARN] 安装命令: pip install albumentations")


# ====== 配置 ======
class Config:
    # 路径
    COCO_ROOT = r"/data/ghaiyan/髋关节影像检测/dataset/coco_format/"
    TRAIN_ANN = os.path.join(COCO_ROOT, "annotations", "train2017.json")
    VAL_ANN = os.path.join(COCO_ROOT, "annotations", "val2017.json")
    TRAIN_IMG_DIR = os.path.join(COCO_ROOT, "train2017", "images")
    VAL_IMG_DIR = os.path.join(COCO_ROOT, "val2017", "images")
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"

    # 模型
    BACKBONE = "resnet50"       # resnet50 或 resnet101
    PRETRAINED = True           # 使用 ImageNet 预训练
    NUM_KEYPOINTS = 19
    HEATMAP_SIZE = (128, 128)    # 输出热图尺寸 (input_size/4 = 512/4)
    NUM_DECONV_LAYERS = 3
    NUM_DECONV_FILTERS = 256
    NUM_DECONV_KERNELS = 4
    
    # 图像
    INPUT_SIZE = (512, 512)     # 统一缩放

    # 训练
    EPOCHS = 250
    BATCH_SIZE = 8
    LR = 1e-3
    LR_MIN = 1e-6
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 损失
    HEATMAP_LOSS_WEIGHT = 1000.0  # 热图损失缩放系数

    # 数据增强
    USE_ALBUMENTATIONS = HAS_ALBUMENTATIONS

    # 其他
    NUM_WORKERS = 2
    SAVE_EVERY = 10
    VAL_EVERY = 5
    SIGMA = 2.0             # 热图高斯核 sigma


cfg = Config()


# ====== 数据集 ======
class HipKeypointDataset(Dataset):
    """COCO 格式的髋关节关键点数据集"""

    def __init__(self, ann_path, img_dir, input_size=(512, 512), 
                 heatmap_size=(128, 128), num_keypoints=19, sigma=2.0,
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

        # 基础图像变换
        self.basic_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Albumentations 增强 (保守策略, 与 YOLOv8-Pose 一致)
        # 医学X光影像增强原则: 保留轻微亮度/对比度变化和水平翻转,
        # 收窄旋转/平移/缩放范围, 禁用弹性形变和噪声注入
        # 对应 YOLOv8: fliplr + hsv_v + translate + erasing, 关闭 shear/perspective/mosaic/mixup
        if self.use_albumentations and is_train:
            self.albu_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.15, rotate_limit=5,
                    border_mode=0, p=0.5,
                ),
                # 注意: ElasticTransform 和 GaussNoise 已移除
                # 弹性形变会扭曲骨骼解剖结构, 高斯噪声干扰X光固有噪声模式
            ], keypoint_params=A.KeypointParams(
                format="xy", remove_invisible=False,
            ))
        else:
            self.albu_transform = None

        print(f"[数据集] {ann_path}: {len(self.images)} 图, {len(self.annotations)} 标注")

    def __len__(self):
        return len(self.images)

    def _generate_heatmaps(self, keypoints, h, w):
        """从关键点生成热图 (高斯核)"""
        heatmaps = np.zeros((self.num_keypoints, h, w), dtype=np.float32)
        
        for kpt_id in range(self.num_keypoints):
            x = keypoints[kpt_id * 3]
            y = keypoints[kpt_id * 3 + 1]
            v = keypoints[kpt_id * 3 + 2]

            if v == 0:
                continue  # 未标注

            # 坐标映射到热图空间
            hx = x * w / self.input_size[1]
            hy = y * h / self.input_size[0]
            
            if hx < 1 or hy < 1 or hx >= w - 1 or hy >= h - 1:
                continue

            # 生成 2D 高斯
            grid_y, grid_x = np.ogrid[0:h, 0:w]
            dist2 = (grid_x - hx) ** 2 + (grid_y - hy) ** 2
            heatmap = np.exp(-dist2 / (2 * self.sigma ** 2))
            
            # 归一化到 [0, 1]
            if heatmap.max() > 0:
                heatmap /= heatmap.max()
            heatmaps[kpt_id] = heatmap

        return heatmaps

    def _augment_keypoints(self, kpts, transform, orig_w, orig_h):
        """应用 albumentations 增强，同时变换关键点"""
        # kpts: [x1,y1, x2,y2, ...] × 19 (不带 visibility)
        kpt_list = []
        kpt_vis = []
        for i in range(self.num_keypoints):
            v = kpts[i * 3 + 2]
            kpt_list.append([kpts[i * 3], kpts[i * 3 + 1]])
            kpt_vis.append(v)

        # 注意: albumentations 要求 kpt 在 [0, w/h] 范围内, 且是列表
        # 先缩放到输入尺寸
        scale_x = self.input_size[1] / orig_w
        scale_y = self.input_size[0] / orig_h
        kpt_scaled = [[p[0] * scale_x, p[1] * scale_y] for p in kpt_list]

        transformed = transform(
            image=np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8),
            keypoints=kpt_scaled,
        )
        
        new_kpts = transformed["keypoints"]
        kpt_flat = []
        for i, p in enumerate(new_kpts):
            kpt_flat.extend([p[0], p[1], kpt_vis[i] if p[0] > 0 and p[1] > 0 else 0])
        return kpt_flat

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
            keypoints = list(ann["keypoints"])  # [x,y,v] × 19 (原始像素坐标)
        else:
            keypoints = [0.0] * (self.num_keypoints * 3)

        # 将关键点从原始像素坐标缩放到 input_size 空间 (训练和验证都需要)
        scale_x = self.input_size[1] / orig_w
        scale_y = self.input_size[0] / orig_h
        for i in range(self.num_keypoints):
            if keypoints[i * 3 + 2] > 0:  # 仅缩放有标注的关键点
                keypoints[i * 3] *= scale_x
                keypoints[i * 3 + 1] *= scale_y

        # 数据增强 (仅在训练时)
        if self.albu_transform and self.is_train:
            kpt_list = []
            kpt_vis = []
            for i in range(self.num_keypoints):
                kpt_list.append([keypoints[i * 3], keypoints[i * 3 + 1]])
                kpt_vis.append(keypoints[i * 3 + 2])

            img_np = np.array(image)
            transformed = self.albu_transform(
                image=img_np,
                keypoints=kpt_list,
            )
            image = Image.fromarray(transformed["image"])
            new_kpts = transformed["keypoints"]
            keypoints = []
            for i, p in enumerate(new_kpts):
                keypoints.extend([p[0], p[1], kpt_vis[i] if p[0] > 0 and p[1] > 0 else 0])

        # 生成热图
        h, w = self.heatmap_size
        heatmaps = self._generate_heatmaps(keypoints, h, w)

        # 转 tensor
        img_tensor = self.basic_transform(image)
        heatmaps_tensor = torch.from_numpy(heatmaps)
        kpt_tensor = torch.tensor(keypoints, dtype=torch.float32)

        return {
            "image": img_tensor,
            "heatmaps": heatmaps_tensor,
            "keypoints": kpt_tensor,
            "img_id": img_id,
        }


# ====== SimpleBaseline 模型 ======
class SimpleBaseline(nn.Module):
    """
    ResNet 骨干 + 三层反卷积 → 热图
    
    架构:
      ResNet (去尾) → 1×1 conv(降维) → 3× Deconv → 1×1 conv → N个热图通道
    """

    def __init__(self, backbone="resnet50", num_keypoints=19,
                 num_deconv_layers=3, num_deconv_filters=256,
                 num_deconv_kernels=4, pretrained=True):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_deconv_layers = num_deconv_layers
        self.num_deconv_filters = num_deconv_filters

        # 骨干网络 (去掉最后的 fc 和 avgpool)
        if backbone == "resnet50":
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.backbone_channels = 2048
        elif backbone == "resnet101":
            resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)
            self.backbone_channels = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # 1×1 卷积降维
        self.conv1x1 = nn.Conv2d(self.backbone_channels, num_deconv_filters, 1)

        # 反卷积层
        deconv_layers = []
        for i in range(num_deconv_layers):
            deconv_layers.extend([
                nn.ConvTranspose2d(
                    num_deconv_filters, num_deconv_filters,
                    kernel_size=num_deconv_kernels, stride=2, padding=1,
                ),
                nn.BatchNorm2d(num_deconv_filters),
                nn.ReLU(inplace=True),
            ])
        self.deconv = nn.Sequential(*deconv_layers)

        # 最终 1×1 卷积 → N 个热图通道
        self.final_layer = nn.Conv2d(num_deconv_filters, num_keypoints, 1)

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

        nn.init.kaiming_normal_(self.final_layer.weight, mode="fan_out", nonlinearity="relu")
        if self.final_layer.bias is not None:
            nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1x1(x)
        x = self.deconv(x)
        x = self.final_layer(x)
        return x


# ====== 损失函数 ======
class JointsMSELoss(nn.Module):
    """关键点热图 MSE 损失"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target, target_weight=None):
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        heatmap_pred = output.reshape((batch_size, num_joints, -1))
        heatmap_gt = target.reshape((batch_size, num_joints, -1))

        loss = ((heatmap_pred - heatmap_gt) ** 2).mean(dim=2)
        
        if target_weight is not None:
            loss = loss * target_weight
            loss = loss.sum() / (target_weight.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss


# ====== 评估指标 ======
def compute_pckh(pred_kpts, gt_kpts, valid_mask, threshold=0.5):
    """
    PCK@0.5 指标 (Percentage of Correct Keypoints)
    以所有可见关键点的外接矩形对角线长度的比例作为归一化参考。
    (髋关节影像无头部结构, 无法使用标准 PCKh 的头部长度参考)

    Args:
        pred_kpts: [B, num_kpts, 2] 预测关键点 (输入空间坐标)
        gt_kpts: [B, num_kpts, 2] 真实关键点 (输入空间坐标, 已缩放至 input_size)
        valid_mask: [B, num_kpts] 可见性掩码 (True=有标注)
        threshold: 0.5 表示预测点在参考距离50%以内即视为正确

    Returns:
        pckh: float, 范围 [0, 1]
    """
    B, N, _ = gt_kpts.shape
    total_correct = 0
    total_valid = 0

    for b in range(B):
        mask_b = valid_mask[b]  # [N]
        if mask_b.sum() == 0:
            continue

        gt_vis = gt_kpts[b][mask_b]  # [M, 2]
        pred_vis = pred_kpts[b][mask_b]  # [M, 2]

        # 归一化参考: 可见关键点外接矩形对角线
        x_min, y_min = gt_vis.min(dim=0).values
        x_max, y_max = gt_vis.max(dim=0).values
        ref_size = max(torch.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2).item(), 1e-6)

        dist = torch.norm(pred_vis - gt_vis, dim=1)  # [M]
        correct = (dist < threshold * ref_size).float()
        total_correct += correct.sum().item()
        total_valid += mask_b.sum().item()

    if total_valid == 0:
        return 0.0
    return total_correct / total_valid


# ====== 训练函数 ======
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0.0
    total_mse = 0.0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        heatmaps_gt = batch["heatmaps"].to(device)

        optimizer.zero_grad()
        outputs = model(images)  # [B, 19, 64, 64]
        loss = criterion(outputs, heatmaps_gt) * cfg.HEATMAP_LOSS_WEIGHT
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()

        total_loss += loss.item()
        total_mse += (loss.item() / cfg.HEATMAP_LOSS_WEIGHT)

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} (MSE: {loss.item()/cfg.HEATMAP_LOSS_WEIGHT:.6f})")

    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)

    if writer:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/mse", avg_mse, epoch)

    return avg_loss


def val_epoch(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    all_pckh = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            heatmaps_gt = batch["heatmaps"].to(device)
            keypoints = batch["keypoints"].to(device)

            outputs = model(images)
            loss = criterion(outputs, heatmaps_gt) * cfg.HEATMAP_LOSS_WEIGHT
            total_loss += loss.item()

            # 从热图提取关键点坐标 (输入空间 512×512)
            pred_kpts = _heatmaps_to_keypoints(outputs, cfg.INPUT_SIZE)
            # GT 关键点已缩放至 input_size 空间 (在 __getitem__ 中处理)
            gt_kpts = keypoints.view(-1, cfg.NUM_KEYPOINTS, 3)[:, :, :2]

            # PCK@0.5 (使用外接矩形对角线归一化)
            valid_mask = keypoints.view(-1, cfg.NUM_KEYPOINTS, 3)[:, :, 2] > 0
            avg_pckh = compute_pckh(pred_kpts, gt_kpts, valid_mask, threshold=0.5)
            if avg_pckh > 0:
                all_pckh.append(avg_pckh)

    avg_loss = total_loss / len(dataloader)
    avg_pckh = np.mean(all_pckh) if all_pckh else 0.0

    if writer:
        writer.add_scalar("val/loss", avg_loss, epoch)
        writer.add_scalar("val/PCKh@0.5", avg_pckh, epoch)

    return avg_loss, avg_pckh


def _heatmaps_to_keypoints(heatmaps, original_size):
    """从热图提取关键点坐标 (取最大值位置)"""
    B, K, H, W = heatmaps.shape
    hw = heatmaps.view(B, K, -1)
    max_idx = hw.argmax(dim=2)

    y = (max_idx // W).float()
    x = (max_idx % W).float()

    # 映射回输入坐标 (512×512), 不做原始尺寸映射
    x = x * original_size[1] / W
    y = y * original_size[0] / H

    kpts = torch.stack([x, y], dim=2)  # [B, K, 2], 输入空间坐标
    return kpts


# ====== 推理 ======
def predict_single(model, image_path, device):
    """单张图片预测"""
    from PIL import Image
    
    model.eval()
    transform = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)  # [1, 19, H', W']

    # 提取关键点
    kpts = _heatmaps_to_keypoints(outputs, cfg.INPUT_SIZE)[0]  # [19, 2]
    return kpts.cpu().numpy()


# ====== 主入口 ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "val", "predict"], default="train")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--device", type=str, default=cfg.DEVICE)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch
    cfg.LR = args.lr

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    print(f"[INFO] 设备: {device}")
    print(f"[INFO] 数据集: {cfg.COCO_ROOT}")

    # 加载数据
    train_dataset = HipKeypointDataset(
        cfg.TRAIN_ANN, cfg.TRAIN_IMG_DIR,
        input_size=cfg.INPUT_SIZE, heatmap_size=cfg.HEATMAP_SIZE,
        num_keypoints=cfg.NUM_KEYPOINTS, sigma=cfg.SIGMA,
        use_albumentations=cfg.USE_ALBUMENTATIONS, is_train=True,
    )
    val_dataset = HipKeypointDataset(
        cfg.VAL_ANN, cfg.VAL_IMG_DIR,
        input_size=cfg.INPUT_SIZE, heatmap_size=cfg.HEATMAP_SIZE,
        num_keypoints=cfg.NUM_KEYPOINTS, sigma=cfg.SIGMA,
        use_albumentations=False, is_train=False,
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
    model = SimpleBaseline(
        backbone=cfg.BACKBONE, num_keypoints=cfg.NUM_KEYPOINTS,
        pretrained=cfg.PRETRAINED,
    ).to(device)

    print(f"[模型] {cfg.BACKBONE}, 参数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    criterion = JointsMSELoss()

    if args.mode == "predict":
        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location=device))
            print(f"[INFO] 加载模型: {args.resume}")
        if args.image:
            kpts = predict_single(model, args.image, device)
            print(f"\n预测关键点 (19个):")
            for i, (x, y) in enumerate(kpts):
                print(f"  {i:2d}: ({x:7.1f}, {y:7.1f})")
        return

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    
    # 学习率调度: warmup → cosine
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=cfg.WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS - cfg.WARMUP_EPOCHS, eta_min=cfg.LR_MIN)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[cfg.WARMUP_EPOCHS])

    writer = SummaryWriter(log_dir=cfg.LOG_DIR)

    start_epoch = 1
    best_pckh = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_pckh = checkpoint.get("best_pckh", 0.0)
        print(f"[INFO] 从 epoch {start_epoch} 恢复训练, 最佳 PCKh={best_pckh:.4f}")

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.EPOCHS}, LR={scheduler.get_last_lr()[0]:.2e}")
        print(f"{'='*60}")

        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        
        # 验证
        if epoch % cfg.VAL_EVERY == 0 or epoch == cfg.EPOCHS:
            val_loss, val_pckh = val_epoch(model, val_loader, criterion, device, epoch, writer)
            print(f"\n[VAL] Loss={val_loss:.4f}, PCKh@0.5={val_pckh:.4f}")

            # 保存最佳模型
            if val_pckh > best_pckh:
                best_pckh = val_pckh
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_pckh": best_pckh,
                    "config": vars(cfg),
                }, os.path.join(cfg.CHECKPOINT_DIR, "best.pth"))
                print(f"[SAVE] 新最佳模型! PCKh@0.5={best_pckh:.4f}")

        # 定期保存
        if epoch % cfg.SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_pckh": best_pckh,
                "config": vars(cfg),
            }, os.path.join(cfg.CHECKPOINT_DIR, f"epoch_{epoch}.pth"))

        scheduler.step()

    writer.close()
    print(f"\n[完成] 训练结束! 最佳 PCKh@0.5 = {best_pckh:.4f}")
    print(f"模型保存在: {cfg.CHECKPOINT_DIR}/best.pth")


if __name__ == "__main__":
    main()
