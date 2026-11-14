#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
装甲板检测：训练 / 验证 / 推理 / 导出
用法示例：
  python armor_yolo.py --mode train
  python armor_yolo.py --mode train --device cpu --epochs 30 --batch 8
  python armor_yolo.py --mode predict --source test_images
  python armor_yolo.py --mode predict --source test.mp4
  python armor_yolo.py --mode predict --source 0
  python armor_yolo.py --mode export --formats onnx openvino
"""

from pathlib import Path
import argparse
import csv
import torch
from ultralytics import YOLO

# ================== 路径配置 ==================
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "armor_dataset"
YAML_PATH = ROOT / "armor.yaml"
RUNS_DIR = ROOT / "runs" / "detect"
PREDICT_DIR = ROOT / "runs" / "predict"


def write_yaml():
    """自动生成 armor.yaml，使用绝对路径避免相对路径找不到数据集"""
    text = f'''path: "{DATASET_DIR.as_posix()}"
train: images/train
val: images/val
test: images/test
names:
  0: armor_red
  1: armor_blue
'''
    YAML_PATH.write_text(text, encoding="utf-8")
    print(f"[OK] 已生成 {YAML_PATH}")
    print(text)


def check_dataset():
    """检查关键目录是否存在"""
    required = [
        DATASET_DIR / "images" / "train",
        DATASET_DIR / "images" / "val",
        DATASET_DIR / "labels" / "train",
        DATASET_DIR / "labels" / "val",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("[警告] 以下目录不存在，请检查数据集结构：")
        for p in missing:
            print("  ", p)
        print("训练至少需要 images/train、images/val、labels/train、labels/val。")
    else:
        print("[OK] train/val 图片和标签目录存在。")
    if not (DATASET_DIR / "images" / "test").exists():
        print("[提示] images/test 不存在，不影响训练，但 test 集验证会跳过。")


def resolve_device(device_arg: str):
    """auto -> 有 GPU 用 0，否则 cpu；输入 0/1/cpu 等原样返回"""
    if device_arg == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if isinstance(device_arg, str) and device_arg.isdigit():
        return int(device_arg)
    return device_arg


def get_weights(args):
    """获取权重：优先 --weights，否则找 runs/detect 下最新的 best.pt"""
    if args.weights:
        p = Path(args.weights)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"权重不存在：{p}")
        return p

    candidates = list(RUNS_DIR.glob("*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "未找到 best.pt，请先训练，或者用 --weights 指定权重路径。"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def print_train_summary(save_dir: Path):
    """打印 results.csv 最后一行的核心指标"""
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        print(f"[提示] 未找到 {csv_path}")
        return

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    last = rows[-1]
    keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    print("\n[训练最终指标]")
    for k in keys:
        if k in last:
            print(f"  {k}: {last[k]}")

    print("\n[重点结果文件]")
    for name in ["results.csv", "results.png", "confusion_matrix.png", "PR_curve.png"]:
        p = save_dir / name
        if p.exists():
            print("  ", p)


def train(args):
    write_yaml()
    check_dataset()

    device = resolve_device(args.device)
    print(f"[INFO] 使用设备：{device}")
    print(f"[INFO] 模型：{args.model}")

    model = YOLO(args.model)

    model.train(
        data=str(YAML_PATH),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        seed=0,
    )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\n[OK] 训练结束，best.pt：{best}")

    print_train_summary(save_dir)

    # 训练完自动做一次 val 验证
    print("\n[INFO] 开始验证 best.pt ...")
    val_model = YOLO(str(best))
    val_model.val(
        data=str(YAML_PATH),
        imgsz=args.imgsz,
        device=device,
        project=str(RUNS_DIR),
        name=f"{args.name}_val",
    )
    print(f"[OK] 验证结果保存在：{RUNS_DIR / (args.name + '_val')}")


def predict(args):
    weights = get_weights(args)
    device = resolve_device(args.device)

    print(f"[INFO] 使用权重：{weights}")
    print(f"[INFO] 推理 source：{args.source}")

    model = YOLO(str(weights))

    source = args.source
    # 摄像头：source=0
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        device=device,
        project=str(PREDICT_DIR),
        name=args.name,
    )
    print(f"[OK] 推理结果请查看：{PREDICT_DIR} 下最新目录")


def export(args):
    weights = get_weights(args)
    model = YOLO(str(weights))

    for fmt in args.formats:
        print(f"\n[INFO] 开始导出 {fmt} ...")
        out = model.export(format=fmt, imgsz=args.imgsz)
        print(f"[OK] {fmt} 导出完成：{out}")


def build_parser():
    p = argparse.ArgumentParser(description="装甲板 YOLO 训练/推理/导出脚本")
    p.add_argument(
        "--mode",
        choices=["train", "predict", "export", "all"],
        default="train",
        help="运行模式",
    )
    p.add_argument(
        "--model",
        default="yolo11n.pt",
        help="预训练权重。文档中的 yolo26n.pt 若不存在，用 yolo11n.pt 或 yolov8n.pt",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--device",
        default="auto",
        help="auto / 0 / 1 / cpu。auto 表示有 GPU 用 GPU",
    )
    p.add_argument("--workers", type=int, default=4, help="Windows 报错可改 0")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--name", default="armor_train")
    p.add_argument("--weights", default=None, help="指定 best.pt 路径")
    p.add_argument("--source", default="test_images", help="图片文件夹/视频/0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--formats",
        nargs="+",
        default=["onnx", "openvino"],
        help="导出格式，例如 onnx openvino",
    )
    return p


def main():
    args = build_parser().parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
    elif args.mode == "export":
        export(args)
    elif args.mode == "all":
        train(args)
        predict(args)
        export(args)


if __name__ == "__main__":
    main()
