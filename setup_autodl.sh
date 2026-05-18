#!/bin/bash
# AutoDL 环境一键配置脚本
# 用法: bash setup_autodl.sh

set -e

echo "=== 1. 创建 conda 环境 ==="
# AutoDL 通常已装好 conda
conda create -n yolov8 python=3.10 -y

echo "=== 2. 激活环境 ==="
source activate yolov8

echo "=== 3. 安装 PyTorch（AutoDL CUDA 版本）==="
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

echo "=== 4. 安装项目依赖 ==="
pip install -r requirements.txt

echo "=== 5. 下载数据集 ==="
python scripts/download_data.py

echo "=== 6. 验证 ==="
python -c "from ultralytics import YOLO; import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "环境配置完成！运行训练: python scripts/train.py"
