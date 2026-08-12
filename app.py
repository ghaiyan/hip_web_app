"""
髋关节关键点检测与参数计算 Web 系统 - 后端
==========================================
FastAPI 后端: 图片上传 → 模型推理 → 参数计算 → 可视化 → 返回结果

启动:
  python app.py               # 默认 0.0.0.0:8000
  python app.py --port 8080   # 指定端口
  python app.py --device cpu  # CPU 模式
"""

import os
import sys
import io
import base64
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ---- 路径设置 ----
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from calculator import compute_all_parameters, KEYPOINT_NAMES, KPT_GROUPS, CLINICAL_REFERENCE


# =====================================================================
#  常量定义
# =====================================================================
MODEL_TYPES = ["drfnet", "yolov8", "simplebaseline"]
DEFAULT_DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
INPUT_SIZE = (512, 512)
HEATMAP_SIZE = 128

# 关键点颜色 (BGR → RGB)
KPT_COLORS_RGB = {
    0:  (255, 255, 0),   1:  (255, 255, 0),   2:  (200, 200, 255),   # 骨盆
    3:  (255, 128, 0),   4:  (255, 128, 0),   5:  (255, 80, 80),   6:  (255, 80, 80),   # 左股骨
    7:  (0, 255, 128),   8:  (0, 255, 128),   9:  (0, 200, 255),  10:  (0, 200, 255),   # 左臼杯
    11: (255, 160, 80), 12: (255, 160, 80), 13: (255, 130, 130), 14: (255, 130, 130),  # 右股骨
    15: (100, 255, 180), 16: (100, 255, 180), 17: (100, 220, 255), 18: (100, 220, 255), # 右臼杯
}

# 连线定义 (关键点索引对)
CONNECTIONS = [
    # 骨盆基准
    (0, 1, (0, 255, 0), "基准线"),
    # 左股骨链
    (3, 4, (255, 128, 0), ""), (4, 5, (255, 128, 0), ""), (5, 6, (255, 128, 0), ""),
    # 右股骨链
    (11, 12, (255, 160, 80), ""), (12, 13, (255, 160, 80), ""), (13, 14, (255, 160, 80), ""),
    # 左臼杯
    (7, 8, (0, 255, 128), ""), (9, 10, (0, 200, 255), ""),
    # 右臼杯
    (15, 16, (100, 255, 180), ""), (17, 18, (100, 220, 255), ""),
]


# =====================================================================
#  模型管理
# =====================================================================
class ModelManager:
    """统一的模型管理器; 支持 DRFNet / YOLOv8-Pose / SimpleBaseline"""

    def __init__(self):
        self.model_type = None
        self.model = None
        self.device = DEFAULT_DEVICE
        self.transform = None
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    def load_drfnet(self, weights_path: str, device: str = None):
        """加载 DRFNet 模型 (消融实验 C)"""
        import torch
        from model_def import DRFNet, load_checkpoint

        dev = device or self.device
        model = DRFNet(num_keypoints=19, pretrained_backbone=True)
        load_checkpoint(model, weights_path, dev)
        model.to(dev)
        model.eval()

        from torchvision import transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = model
        self.device = dev
        self.transform = transform
        self.model_type = "drfnet"
        self._loaded = True
        print(f"[模型] DRFNet 已加载 (设备: {dev})")

    def load_yolov8(self, weights_path: str, device: str = None):
        """加载 YOLOv8-Pose 模型"""
        import torch

        # PyTorch 2.6+ 默认 torch.load(weights_only=True),
        # 需要注册 ultralytics 模型类为安全全局变量，否则加载 .pt 会失败
        try:
            from ultralytics.nn.tasks import PoseModel, DetectionModel
            torch.serialization.add_safe_globals([PoseModel, DetectionModel])
        except (ImportError, AttributeError):
            pass  # 旧版 PyTorch 不需要此操作

        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.device = device or self.device
        self.model_type = "yolov8"
        self._loaded = True
        print(f"[模型] YOLOv8-Pose 已加载")

    def load_simplebaseline(self, weights_path: str, device: str = None):
        """加载 SimpleBaseline 模型"""
        import torch
        from model_def import ResNetBackbone, HeatmapDecoder

        dev = device or self.device

        class SimpleBaselineModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = ResNetBackbone(pretrained=False)
                self.decoder = HeatmapDecoder(in_channels=2048, num_keypoints=19)

            def forward(self, x):
                _, _, c5 = self.backbone(x)
                return self.decoder(c5)

        model = SimpleBaselineModel()
        checkpoint = torch.load(weights_path, map_location=dev, weights_only=False)
        if "model" in checkpoint:
            sd = checkpoint["model"]
        elif "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
        else:
            sd = checkpoint
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[7:]
            new_sd[k] = v
        model.load_state_dict(new_sd, strict=False)
        model.to(dev)
        model.eval()

        from torchvision import transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = model
        self.device = dev
        self.transform = transform
        self.model_type = "simplebaseline"
        self._loaded = True
        print(f"[模型] SimpleBaseline 已加载 (设备: {dev})")


# 全局模型管理器
model_manager = ModelManager()


# =====================================================================
#  推理函数
# =====================================================================
def infer_image(image: Image.Image):
    """对 PIL Image 执行关键点检测推理"""
    import torch

    if not model_manager.is_loaded:
        raise RuntimeError("模型未加载，请先加载模型")

    orig_w, orig_h = image.size
    img_resized = image.resize(INPUT_SIZE, Image.BILINEAR)

    if model_manager.model_type == "yolov8":
        return _infer_yolov8(image, orig_w, orig_h)
    else:
        return _infer_torch(img_resized, orig_w, orig_h)


def _infer_yolov8(image, orig_w, orig_h):
    """YOLOv8-Pose 推理"""
    import tempfile
    import os as _os

    # YOLOv8 需要文件路径
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image.save(f.name, format="JPEG", quality=95)
        tmp_path = f.name

    try:
        results = model_manager.model(tmp_path, verbose=False)
        if len(results) == 0 or results[0].keypoints is None:
            return np.zeros((19, 2))

        kpts_tensor = results[0].keypoints.data
        if kpts_tensor.shape[0] == 0:
            return np.zeros((19, 2))

        confs = results[0].boxes.conf if results[0].boxes is not None else None
        if confs is not None and len(confs) > 0:
            best_idx = confs.argmax().item()
        else:
            best_idx = 0

        kpts = kpts_tensor[best_idx].cpu().numpy()
        return kpts[:, :2]
    finally:
        _os.unlink(tmp_path)


def _infer_torch(img_resized, orig_w, orig_h):
    """PyTorch 模型推理 (DRFNet / SimpleBaseline)"""
    import torch

    img_tensor = model_manager.transform(img_resized).unsqueeze(0).to(model_manager.device)

    with torch.no_grad():
        outputs = model_manager.model(img_tensor)

    # 提取热图分支输出的关键点坐标 (归一化坐标)
    if isinstance(outputs, dict):
        kpts_norm = outputs["kpt_final"][0].cpu().numpy()  # [19, 2], 归一化 [0,1]
    else:
        # SimpleBaseline 直接输出热图
        heatmap = outputs[0].cpu().numpy()
        kpts_norm = _heatmap_to_keypoints_np(heatmap)  # [19, 2]

    # 缩放回原图尺寸
    kpts_pixel = kpts_norm.copy()
    kpts_pixel[:, 0] *= orig_w
    kpts_pixel[:, 1] *= orig_h

    return kpts_pixel


def _heatmap_to_keypoints_np(heatmap):
    """从热图 numpy 数组提取关键点归一化坐标"""
    N, H, W = heatmap.shape
    kpts = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        hm = heatmap[i]
        max_idx = np.argmax(hm)
        y = (max_idx // W) / max(H - 1, 1)
        x = (max_idx % W) / max(W - 1, 1)
        kpts[i] = [x, y]
    return kpts


# =====================================================================
#  可视化
# =====================================================================
def visualize_result(image: Image.Image, kpts: np.ndarray, params: dict, side: str, lang: str = "en") -> str:
    """
    在原始图像上绘制关键点、连线、参数标注，返回 base64 编码的 PNG

    Args:
        lang: "zh" or "en", 控制叠加层文字语言

    Returns:
        str: data:image/png;base64,...
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    # 尝试加载字体
    try:
        font_lg = ImageFont.truetype("simhei.ttf", 16)
        font_md = ImageFont.truetype("simhei.ttf", 13)
        font_sm = ImageFont.truetype("simhei.ttf", 11)
    except Exception:
        font_lg = font_md = font_sm = ImageFont.load_default()

    r = 5  # 关键点半径

    # ---- 绘制连线 ----
    for a, b, color, label in CONNECTIONS:
        if not (kpts[a] == 0).all() and not (kpts[b] == 0).all():
            draw.line([tuple(kpts[a].astype(int)), tuple(kpts[b].astype(int))],
                      fill=color, width=2)

    # ---- 绘制关键点 ----
    for i in range(19):
        x, y = kpts[i]
        if x == 0 and y == 0:
            continue
        color = KPT_COLORS_RGB.get(i, (200, 200, 200))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(255, 255, 255), width=1)
        draw.text((x + r + 2, y - r - 2), str(i), fill=color, font=font_sm)

    # ---- 绘制结果半透明底框 ----
    img_w, img_h = img.size
    lines = []

    if side:
        side_label = {"L": "Left", "R": "Right", "both": "Bilateral"}.get(side, side)
        lines.append(f"Implant side: {side_label}")

    for key, ref in CLINICAL_REFERENCE.items():
        val = None
        for k in params:
            if key in k.lower().replace("_angle", ""):
                if side and side in k:
                    val = params[k]
                    break
                elif side is None:
                    val = params[k]
                    break

        if val is not None:
            ref_range = ref["range"]
            flag = ""
            if ref_range:
                lo, hi = ref_range
                if val < lo:
                    flag = " ↓ Low"
                elif val > hi:
                    flag = " ↑ High"
                else:
                    flag = " ✓"
            name = ref[f"name_{lang}"] if f"name_{lang}" in ref else ref["name_en"]
            unit = ref["unit"]
            range_str = f"[{ref_range[0]}~{ref_range[1]}{unit}]" if ref_range else ""
            lines.append(f'{name}: {val} {unit}{flag} {range_str}')

    if lines:
        box_h = len(lines) * 20 + 12
        box_w = min(520, img_w - 8)
        box_x, box_y = 4, img_h - box_h - 4
        # 半透明黑色底
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([box_x - 1, box_y - 1, box_x + box_w, img_h - 3],
                                fill=(0, 0, 0, 180))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        for i, line in enumerate(lines):
            draw.text((box_x + 4, box_y + 2 + i * 20), line, fill=(180, 255, 180), font=font_sm)

    # ---- 转 base64 ----
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# =====================================================================
#  FastAPI 应用
# =====================================================================
app = FastAPI(
    title="髋关节关键点检测与参数计算系统",
    description="DRFNet-based Hip Joint Keypoint Detection & Clinical Parameter Calculation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# =====================================================================
#  API 路由
# =====================================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    index_path = BASE_DIR / "templates" / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h2>index.html 未找到, 请检查 templates/ 目录</h2>")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def api_status():
    """系统状态"""
    return JSONResponse({
        "status": "ok",
        "model_loaded": model_manager.is_loaded,
        "model_type": model_manager.model_type,
        "device": model_manager.device,
        "available_models": MODEL_TYPES,
    })


@app.post("/api/load_model")
async def api_load_model(
    model_type: str = Form(...),
    weights_path: str = Form(...),
    device: str = Form(DEFAULT_DEVICE),
):
    """加载模型"""
    if model_type not in MODEL_TYPES:
        raise HTTPException(400, f"不支持的模型类型: {model_type}。可选: {MODEL_TYPES}")

    if not os.path.exists(weights_path):
        raise HTTPException(400, f"权重文件不存在: {weights_path}")

    try:
        if model_type == "drfnet":
            model_manager.load_drfnet(weights_path, device)
        elif model_type == "yolov8":
            model_manager.load_yolov8(weights_path, device)
        elif model_type == "simplebaseline":
            model_manager.load_simplebaseline(weights_path, device)
    except Exception as e:
        raise HTTPException(500, f"模型加载失败: {str(e)}")

    return JSONResponse({
        "status": "ok",
        "model_type": model_manager.model_type,
        "device": model_manager.device,
        "message": f"模型 {model_type} 加载成功",
    })


@app.post("/api/detect")
async def api_detect(file: UploadFile = File(...)):
    """
    核心接口: 上传图片 → 检测关键点 → 计算参数 → 返回结果

    Request:  multipart/form-data, file field
    Response: JSON with keypoints, parameters, visualized_base64
    """
    if not model_manager.is_loaded:
        raise HTTPException(400, "模型未加载。请先调用 /api/load_model")

    # 读取上传图片
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"图片读取失败: {str(e)}")

    orig_w, orig_h = image.size
    t0 = time.time()

    # 推理
    try:
        kpts_pixel = infer_image(image)
    except Exception as e:
        raise HTTPException(500, f"模型推理失败: {str(e)}")

    infer_time = round((time.time() - t0) * 1000)  # ms

    # 计算临床参数
    result = compute_all_parameters(kpts_pixel, (orig_w, orig_h))

    # 可视化 (叠加层使用英文)
    vis_b64 = visualize_result(image, kpts_pixel, result["parameters"], result["side"], lang="en")

    # 关键点分组 PCK (示例, 需要 GT 才有意义; 这里返回坐标信息)
    kpt_groups_info = {}
    for group_name, indices in KPT_GROUPS.items():
        kpt_groups_info[group_name] = {
            "indices": indices,
            "present": [i for i in indices if not (np.allclose(kpts_pixel[i], [0, 0]))],
            "missing": [i for i in indices if np.allclose(kpts_pixel[i], [0, 0])],
        }

    return JSONResponse({
        "status": "ok",
        "model_type": model_manager.model_type,
        "inference_time_ms": infer_time,
        "image_size": [orig_w, orig_h],
        "implant_side": result["side"],
        "parameters": result["parameters"],
        "warnings": result["warnings"],
        "reference": CLINICAL_REFERENCE,
        "keypoints": {str(i): [float(kpts_pixel[i][0]), float(kpts_pixel[i][1])] for i in range(19)},
        "keypoint_names": {str(i): KEYPOINT_NAMES[i] for i in range(19)},
        "keypoint_groups": kpt_groups_info,
        "visualized_base64": vis_b64,
    })


# =====================================================================
#  入口
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="髋关节关键点检测与参数计算 Web 系统")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                       help="推理设备 (cuda/cpu)")
    parser.add_argument("--weights", type=str, default=None,
                       help="自动加载的模型权重路径")
    parser.add_argument("--model", type=str, default="drfnet",
                       choices=MODEL_TYPES,
                       help="模型类型")
    args = parser.parse_args()

    model_manager.device = args.device

    # 自动加载模型
    if args.weights and os.path.exists(args.weights):
        print(f"[系统] 自动加载模型: {args.model} from {args.weights}")
        try:
            if args.model == "drfnet":
                model_manager.load_drfnet(args.weights, args.device)
            elif args.model == "yolov8":
                model_manager.load_yolov8(args.weights, args.device)
            elif args.model == "simplebaseline":
                model_manager.load_simplebaseline(args.weights, args.device)
        except Exception as e:
            print(f"[警告] 模型自动加载失败: {e}")
            print("[提示] 请通过 /api/load_model 接口手动加载模型")
    else:
        print("[提示] 未指定模型权重, 请在 Web 界面手动加载模型")

    print(f"\n{'='*60}")
    print(f"  髋关节关键点检测与参数计算系统")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  设备: {args.device}")
    print(f"  模型: {'已加载' if model_manager.is_loaded else '未加载'}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
