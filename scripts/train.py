"""YOLOv8 车牌检测训练脚本"""
import argparse
import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 车牌检测训练")
    parser.add_argument("--config", default="configs/train.yaml", help="训练配置文件")
    parser.add_argument("--epochs", type=int, help="覆盖训练轮数")
    parser.add_argument("--batch", type=int, help="覆盖 batch size")
    parser.add_argument("--lr0", type=float, help="覆盖初始学习率")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for key in ("epochs", "batch", "lr0"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    model = YOLO(cfg.pop("model"))
    model.train(**cfg)


if __name__ == "__main__":
    main()
