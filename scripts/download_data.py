"""
从 Roboflow 下载车牌识别数据集

Roboflow 是一个数据集托管平台：
- 这个项目使用的数据集是 license-plate-recognition-rxg4e
- 版本 11，格式为 YOLOv8（即每张图片对应一个 .txt 标签文件）
- 标签格式：class x_center y_center width height（归一化坐标）

下载后自动分割为：
  data/dataset/
    ├── train/   images/ + labels/   ← 训练集（~70%）
    ├── valid/   images/ + labels/   ← 验证集（~20%，训练时评估用）
    └── test/    images/ + labels/   ← 测试集（~10%，最终评估用）

每个 .txt 标签文件内容示例（一行一个车牌）：
  0 0.543 0.678 0.123 0.045
  └  └───────┴───────┴───────┘
 类别   归一化的框坐标 (xywh)
"""
from roboflow import Roboflow

API_KEY = "h5MQcuqAwwpVtLrozKa9"

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
dataset = project.version(11).download("yolov8", location="data/dataset")

print(f"数据集已下载到: data/dataset/")
