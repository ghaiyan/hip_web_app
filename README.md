# hip_web_app

## 项目简介

这是一个基于 FastAPI 的髋关节关键点检测与临床参数计算 Web 应用。项目支持上传 X 光片图像后端进行关键点检测、假体侧识别、五项临床参数计算，并返回可视化结果图。

核心功能：
- DRFNet / YOLOv8-Pose / SimpleBaseline 三种关键点检测模型加载
- 19 个髋关节关键点预测
- 计算臼杯外展角、臼杯前倾角、颈干角、股骨偏心距、下肢长度差
- 前端页面上传图片并查看可视化叠加结果
- 支持自动加载权重文件、GPU/CPU 推理模式

## 目录结构

- `app.py` - FastAPI 后端入口，负责模型加载、推理、参数计算与接口定义
- `calculator.py` - 临床参数计算逻辑：关键点解析与参数输出
- `model_def.py` - DRFNet 与 SimpleBaseline 模型定义与加载函数
- `templates/index.html` - Web 前端页面
- `static/` - 静态资源目录
- `weights/` - 模型权重文件存放目录（示例权重文件已存在）
- `requirements.txt` - Python 依赖列表
- `start.sh` - 启动脚本

## 依赖安装

建议使用 Python 3.10+。

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

如果使用 YOLOv8 模型，请安装 `ultralytics>=8.4.0`。

## 启动方式

### 直接运行

```bash
python app.py
```

默认启动后端服务并输出访问地址，默认监听 `0.0.0.0:8000`。

### 指定端口、设备和模型权重

```bash
python app.py --port 8080 --device cpu --model drfnet --weights ./weights/drfnet_best.pth
```

### 使用启动脚本

```bash
bash start.sh --port 8080 --gpu --model drfnet --weights ./weights/drfnet_best.pth
```

## 可访问页面

启动成功后，在浏览器中访问：

```text
http://127.0.0.1:8000/
```

前端页面支持：
- 模型类型选择：`drfnet` / `yolov8` / `simplebaseline`
- 权重文件路径输入
- 推理设备选择：`cuda` / `cpu`
- 图像上传并显示检测结果

## API 说明

### `/api/status`

检查服务状态与模型加载情况。

响应示例：
```json
{
  "status": "ok",
  "model_loaded": false,
  "model_type": null,
  "device": "cuda",
  "available_models": ["drfnet", "yolov8", "simplebaseline"]
}
```

### `/api/load_model` (POST)

加载模型权重。

表单字段：
- `model_type`：`drfnet` / `yolov8` / `simplebaseline`
- `weights_path`：权重文件路径
- `device`：`cuda` / `cpu`

### `/api/detect` (POST)

上传图像进行推理与参数计算。

请求：`multipart/form-data`，字段名 `file`。

响应包含：
- `image_size`：原始图像尺寸
- `implant_side`：假体侧（`L`、`R`、`both`）
- `parameters`：临床参数值
- `warnings`：缺失关键点或计算异常提示
- `keypoints`：19 个关键点像素坐标
- `visualized_base64`：可直接展示的 Base64 图像结果

## 模型与权重

项目支持三种模型：

- `drfnet`：基于 `model_def.py` 中 DRFNet 架构
- `yolov8`：基于 `ultralytics` YOLOv8-Pose
- `simplebaseline`：基于 ResNet50 + HeatmapDecoder

权重文件目录为 `weights/`，模板项目中包含多个示例权重文件。

## 关键点与参数说明

项目中使用 19 个关键点：
- `P1_Teardrop_L`, `P2_Teardrop_R`, `P3_Symphysis`
- 左右股骨头、颈、干、臼杯关键点

计算的临床参数包括：
- 臼杯外展角（abduction_angle）
- 臼杯前倾角（anteversion_angle）
- 颈干角（neck_shaft_angle）
- 股骨偏心距（femoral_offset）
- 下肢长度差（leg_length_diff）

## 运行提示

- 若使用 GPU，请确保 `torch` 能识别 CUDA
- YOLOv8 模型需要 `ultralytics` 库
- 若模型加载失败，可先检查 `weights_path` 是否正确
- 前端页面通过 `/` 访问，后台接口通过 `/api/load_model` 和 `/api/detect` 交互

## 扩展建议

如果你要继续开发，可以考虑：
- 添加模型训练入口与参数配置
- 支持更多图像格式与批量推理
- 增加更完整的前端结果展示与中文/英文切换
- 输出 CSV/JSON 报告

---

`hip_web_app` 是一个面向髋关节假体影像分析的轻量级可视化推理系统，适合用于模型验证、临床参数计算与快速演示。
