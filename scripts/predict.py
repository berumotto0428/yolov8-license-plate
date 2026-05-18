"""使用训练好的模型进行推理和评估"""
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO


def find_model():
    best_pt = Path("weights/best.pt")
    if best_pt.exists():
        return best_pt
    exp_dirs = sorted(Path("runs/detect").glob("exp*"))
    if exp_dirs:
        return exp_dirs[-1] / "weights/best.pt"
    print("错误：未找到模型文件")
    sys.exit(1)


def run_eval(model):
    print("=== 测试集评估 ===\n")
    metrics = model.val(data="configs/dataset.yaml", split="test")
    print(f"\nmAP50:   {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p[0]:.4f}")
    print(f"Recall:    {metrics.box.r[0]:.4f}")


def run_infer(model, source_path):
    source_path = Path(source_path)
    if source_path.is_file():
        img_paths = [source_path]
    elif source_path.is_dir():
        img_paths = sorted(p for p in source_path.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    else:
        print(f"错误：找不到路径 {source_path}")
        return

    if not img_paths:
        print(f"没有找到图片: {source_path}")
        return

    save_dir = Path("runs/detect/infer")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n检测 {len(img_paths)} 张图片，结果保存到: {save_dir}")

    for img_path in img_paths:
        results = model(str(img_path))
        boxes = results[0].boxes

        img = cv2.imread(str(img_path))
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            conf = boxes.conf[i].item()
            cls = int(boxes.cls[i].item())
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out_path = save_dir / f"{img_path.stem}_result.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"  [{img_path.name}] {len(boxes)} 个车牌")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print("用法:")
        print("  python scripts/predict.py             测试集评估")
        print("  python scripts/predict.py <路径>      单张/批量推理")
        return

    model = YOLO(str(find_model()))

    if len(sys.argv) < 2:
        run_eval(model)
    else:
        run_infer(model, sys.argv[1])


if __name__ == "__main__":
    main()
