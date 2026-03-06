"""
train.py
--------
Fine-tunes model/best.pt on the prepared data (data/train, data/valid).
Run after prepare_data.py.

    python train.py --epochs 30 --batch 4 --imgsz 640
"""

import argparse
import os
import shutil
from ultralytics import YOLO

ROOT       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "model", "best.pt")
DATA_YAML  = os.path.join(ROOT, "data.yaml")


def count_train_images():
    img_dir = os.path.join(ROOT, "data", "train", "images")
    if not os.path.isdir(img_dir):
        return 0
    return len([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def train(epochs, batch, imgsz):
    n = count_train_images()
    if n > 0:
        epochs = min(epochs, max(5, n * 3))
        print(f"Dataset size: {n} image(s) → running {epochs} epoch(s)")

    model = YOLO(MODEL_PATH)           # start from existing weights — fine-tune
    model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        project=os.path.join(ROOT, "runs", "train"),
        name="finetune",
        exist_ok=True,
    )

    new_best = os.path.join(ROOT, "runs", "train", "finetune", "weights", "best.pt")
    if os.path.exists(new_best):
        shutil.copy(new_best, MODEL_PATH)
        print(f"model/best.pt updated from {new_best}")
    else:
        print("Warning: new best.pt not found — model/best.pt unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch",  type=int, default=4)
    parser.add_argument("--imgsz",  type=int, default=640)
    args = parser.parse_args()
    train(args.epochs, args.batch, args.imgsz)
