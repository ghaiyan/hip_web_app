#!/bin/bash
# 髋关节关键点检测与参数计算系统 - 启动脚本
# ============================================
# 用法:
#   bash start.sh                          # 默认 8000 端口 CPU 模式
#   bash start.sh --port 8080 --gpu        # GPU 模式 8080 端口
#   bash start.sh --weights ./weights/drfnet_C_best.pth

PORT=8000
DEVICE="cpu"
WEIGHTS=""
MODEL="drfnet"

while [[ $# -gt 0 ]]; do
  case $1 in
    --port)    PORT="$2"; shift 2 ;;
    --gpu)     DEVICE="cuda"; shift ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --model)   MODEL="$2"; shift 2 ;;
    *)         shift ;;
  esac
done

CMD="python app.py --host 0.0.0.0 --port $PORT --device $DEVICE --model $MODEL"
if [ -n "$WEIGHTS" ]; then
  CMD="$CMD --weights $WEIGHTS"
fi

echo "============================================"
echo "  髋关节关键点检测与参数计算系统"
echo "============================================"
echo "  端口: $PORT"
echo "  设备: $DEVICE"
echo "  模型: $MODEL"
if [ -n "$WEIGHTS" ]; then
  echo "  权重: $WEIGHTS"
fi
echo "============================================"
echo ""

$CMD
