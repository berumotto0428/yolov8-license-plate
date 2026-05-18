# YOLOv8 车牌检测

基于 YOLOv8 的车牌检测项目，使用 Roboflow 开源数据集。

## 项目结构

```
├── configs/
│   ├── dataset.yaml          # 数据集配置
│   └── train.yaml            # 训练超参数
├── data/dataset/             # 数据集
│   ├── train/images/         # 训练集图片
│   ├── train/labels/         # 训练集标签
│   ├── valid/                # 验证集
│   └── test/                 # 测试集
├── scripts/
│   ├── train.py              # 训练
│   ├── predict.py            # 评估 + 推理
│   └── download_data.py      # 下载数据集
├── runs/detect/              # 输出
│   ├── exp/                  # 训练结果
│   ├── val/                  # 评估结果
│   └── infer/                # 推理结果
├── weights/                  # 模型权重
│   └── best.pt
├── setup_autodl.sh           # AutoDL 环境配置
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

```bash
python scripts/download_data.py
```

### 3. 训练

```bash
python scripts/train.py                   # 使用 configs/train.yaml 默认参数
python scripts/train.py --epochs 50       # 覆盖轮数
python scripts/train.py --batch 64 --lr0 0.001  # 覆盖 batch 和学习率
```

### 4. 评估

```bash
python scripts/predict.py                 # 在测试集上评估模型
```

### 5. 推理

```bash
python scripts/predict.py 图片.jpg         # 单张图片
python scripts/predict.py 目录/            # 批量推理目录下所有图片
```

结果保存到 `runs/detect/infer/`。
