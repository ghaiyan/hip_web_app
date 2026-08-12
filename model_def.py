"""
DRFNet 模型架构定义 (消融实验 C)
==================================
从 train_drfnet_ablation.py 提取的纯推理用模型定义。
支持加载训练好的 checkpoint 进行推理。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# =====================================================================
#  ResNet50 骨干
# =====================================================================
class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = resnet50(weights=weights)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2   # C3: 512ch, stride 8
        self.layer3 = resnet.layer3   # C4: 1024ch, stride 16
        self.layer4 = resnet.layer4   # C5: 2048ch, stride 32

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


# =====================================================================
#  FPN 颈部
# =====================================================================
class FPN(nn.Module):
    def __init__(self, in_channels=(512, 1024, 2048), out_channels=256):
        super().__init__()
        self.lat5 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.lat4 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lat3 = nn.Conv2d(in_channels[0], out_channels, 1)
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
        p5 = self.lat5(c5)
        p4 = self.lat4(c4)
        p3 = self.lat3(c3)
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        return p3, p4, p5


# =====================================================================
#  检测分支头部 (仅辅助监督, 不输出)
# =====================================================================
class DetectionHead(nn.Module):
    def __init__(self, fpn_channels=256, num_keypoints=19, hidden=256):
        super().__init__()
        total_in = fpn_channels * 3
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
        g3 = F.adaptive_avg_pool2d(p3, 1).view(B, -1)
        g4 = F.adaptive_avg_pool2d(p4, 1).view(B, -1)
        g5 = F.adaptive_avg_pool2d(p5, 1).view(B, -1)
        feat = torch.cat([g3, g4, g5], dim=1)
        feat = self.shared(feat)
        bbox = self.box_head(feat)
        kpt = self.kpt_head(feat).view(B, -1, 2)
        kobj = self.kobj_head(feat)
        return bbox, kpt, kobj


# =====================================================================
#  热图解码器
# =====================================================================
class HeatmapDecoder(nn.Module):
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
        x = self.deconv(x)
        x = self.final(x)
        x = torch.sigmoid(x)
        return x


# =====================================================================
#  DRFNet (消融 C: 仅热图输出, 无 DGSAM/CGDR/AFM)
# =====================================================================
class DRFNet(nn.Module):
    """
    DRFNet 消融实验 C:
    - 共享 ResNet50 骨干 + FPN
    - 检测分支: 仅提供辅助监督信号, 不参与最终输出
    - 热图解码器: 唯一输出分支 → K_final = K_heatmap
    - 无 DGSAM / CGDR / AFM
    """

    def __init__(self, num_keypoints=19, pretrained_backbone=True):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)
        self.fpn = FPN(in_channels=(512, 1024, 2048), out_channels=256)
        self.det_head = DetectionHead(fpn_channels=256, num_keypoints=num_keypoints)
        self.heatmap_decoder = HeatmapDecoder(in_channels=256, num_keypoints=num_keypoints)

    def _heatmap_to_coords(self, heatmap):
        B, N, H, W = heatmap.shape
        heat_flat = heatmap.view(B, N, -1)
        max_idx = heat_flat.argmax(dim=2)
        max_val = heat_flat.max(dim=2).values
        y = (max_idx // W).float() / max(H - 1, 1)
        x = (max_idx % W).float() / max(W - 1, 1)
        coords = torch.stack([x, y], dim=2)
        return coords, max_val

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.fpn(c3, c4, c5)
        # 检测分支 (仅用于前向传播, 不用于输出)
        _bbox_pred, _kpt_direct, _kobj_pred = self.det_head(p3, p4, p5)
        # 热图解码 (唯一输出)
        heatmap = self.heatmap_decoder(p5)
        kpt_heatmap, confidence = self._heatmap_to_coords(heatmap)
        return {
            "heatmap": heatmap,
            "confidence": confidence,
            "kpt_final": kpt_heatmap,    # K_final = K_heatmap (消融 C)
        }


def load_checkpoint(model: DRFNet, checkpoint_path: str, device: str = "cpu") -> DRFNet:
    """加载训练好的 checkpoint 到 DRFNet 模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 移除 DataParallel 的 "module." 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model
