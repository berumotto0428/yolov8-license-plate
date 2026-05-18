"""
YOLOv8 车牌检测训练脚本

整个训练过程：
1. 读取配置（模型类型、数据路径、超参数）
2. 加载预训练权重 yolov8n.pt（在 COCO 上训过的通用特征提取器）
3. 在自己的车牌数据集上继续训练（迁移学习）
4. 每轮结束在验证集上评估，保存效果最好的权重

YOLOv8 的工作流程：
- 输入图片 → 640×640 缩放 → 主干网络（CSPDarknet）提取特征
  → 特征金字塔（FPN+PAN）多尺度融合 → 检测头输出预测框
- 损失 = box_loss（框位置）+ cls_loss（分类）+ dfl_loss（框分布）
- 优化器通过反向传播调整权重，让损失越来越小
"""
import argparse
import yaml
from ultralytics import YOLO


def main():
    # argparse：从命令行读取参数，支持 --config 指定配置文件
    # 也支持 --epochs / --batch / --lr0 临时覆盖配置里的值
    parser = argparse.ArgumentParser(description="YOLOv8 车牌检测训练")
    parser.add_argument("--config", default="configs/train.yaml", help="训练配置文件")
    parser.add_argument("--epochs", type=int, help="覆盖训练轮数")
    parser.add_argument("--batch", type=int, help="覆盖 batch size")
    parser.add_argument("--lr0", type=float, help="覆盖初始学习率")
    args = parser.parse_args()

    # 读取 YAML 配置文件，把训练参数加载到字典 cfg 里
    # 包含：model, data, epochs, batch, imgsz, lr0, device 等
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 如果命令行指定了 epochs/batch/lr0，覆盖 YAML 里的值
    # 这样不用改配置文件也能临时调参
    for key in ("epochs", "batch", "lr0"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    # YOLO() 是 ultralytics 库的核心类：
    # 传 "yolov8n.pt" → 加载预训练权重
    # 传 "yolov8n.yaml" → 从零开始训练（需要更多数据）
    model = YOLO(cfg.pop("model"))  # pop 取出 "model" 键，剩下的传给 train()

    # model.train() 启动训练：
    # - 自动从 cfg['data'] 指定的路径加载数据集
    # - 每轮遍历所有 batch，前向传播 → 算损失 → 反向传播 → 更新权重
    # - 每轮结束在验证集上跑一次评估
    # - 训练完成后 weights/best.pt 和 last.pt 保存到 save_dir
    model.train(**cfg)


if __name__ == "__main__":
    main()
