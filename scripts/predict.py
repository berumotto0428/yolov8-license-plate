"""
使用训练好的模型进行推理和评估

两种模式：
1. 不带参数运行 → 在测试集上评估（输出 mAP、Precision、Recall）
2. 带图片路径运行 → 检测单张/批量图片，画出框保存结果

推理时 YOLOv8 内部做了这些事：
1. 输入图片缩放到 640×640（保持宽高比，不足补灰边）
2. 通过 CNN 前向传播，输出特征图
3. 在特征图上解码出候选框（每个格子预测多个框）
4. NMS（非极大值抑制）：按置信度排序，去掉重叠过多的框
5. 输出最终检测框的 xyxy 坐标、置信度、类别
"""
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO


def find_model():
    """
    查找最佳模型权重，按优先级：
    1. weights/best.pt（项目根目录下的）
    2. runs/detect/exp*/weights/best.pt（最近一次训练的）
    """
    best_pt = Path("weights/best.pt")
    if best_pt.exists():
        return best_pt
    exp_dirs = sorted(Path("runs/detect").glob("exp*"))
    if exp_dirs:
        return exp_dirs[-1] / "weights/best.pt"
    print("错误：未找到模型文件")
    sys.exit(1)


def run_eval(model):
    """
    在测试集上评估模型精度

    model.val() 做的事情：
    - 遍历测试集所有图片
    - 对每张图片做推理，得到预测框
    - 将预测框和标注真值（ground truth）对比
    - 计算 mAP（Mean Average Precision）

    关键指标：
    - mAP50: IoU 阈值 0.5 时的 mAP（宽松标准）
    - mAP50-95: IoU 从 0.5 到 0.95 取平均（严格标准）
    - Precision: 预测的框中有多少是真的车牌
    - Recall: 真正的车牌有多少被检测出来了
    """
    print("=== 测试集评估 ===\n")
    # split="test" 指定用测试集评估，不传默认用验证集
    # save_dir 自动设为 runs/detect/val/，输出混淆矩阵等图
    metrics = model.val(data="configs/dataset.yaml", split="test")
    print(f"\nmAP50:   {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p[0]:.4f}")
    print(f"Recall:    {metrics.box.r[0]:.4f}")


def run_infer(model, source_path):
    """
    对单张图片或目录下所有图片做推理，画出检测框保存结果

    model() 推理返回的 Results 对象包含：
    - boxes.xyxy:   [N, 4] 检测框坐标（左上角 x,y, 右下角 x,y）
    - boxes.conf:   [N]    每个框的置信度（0~1）
    - boxes.cls:    [N]    每个框的类别索引（车牌 = 0）
    - names:         类别名映射 {0: 'License_Plate'}

    NMS 已经在 model() 内部自动执行了，返回的 boxes
    已经是过滤后的最终结果
    """
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
        # model() = 推理。内部流程：
        #   预处理 → 模型前向 → NMS → 输出 Results
        #  预处理：resize 到 640×640 + 归一化
        results = model(str(img_path))
        boxes = results[0].boxes  # batch=1，取第一张

        # 用 OpenCV 在原图上画检测框
        img = cv2.imread(str(img_path))
        for i in range(len(boxes)):
            # boxes.xyxy[i] → [x1, y1, x2, y2]，框的像素坐标
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            # 置信度，如 0.95 表示 95% 确定这是车牌
            conf = boxes.conf[i].item()
            # 类别索引（single_cls 所以永远是 0）
            cls = int(boxes.cls[i].item())
            label = f"{model.names[cls]} {conf:.2f}"

            # 画绿色框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 在框上方画黑色背景标签
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out_path = save_dir / f"{img_path.stem}_result.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"  [{img_path.name}] {len(boxes)} 个车牌")


def main():
    # 没有参数 → 评估模式；有参数 → 推理模式
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print("用法:")
        print("  python scripts/predict.py             测试集评估")
        print("  python scripts/predict.py <路径>      单张/批量推理")
        return

    # 加载训练好的模型权重
    model = YOLO(str(find_model()))

    if len(sys.argv) < 2:
        run_eval(model)
    else:
        run_infer(model, sys.argv[1])


if __name__ == "__main__":
    main()
