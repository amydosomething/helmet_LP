"""
prepare_data.py
---------------
Reads violation_certain.csv, takes all rows where Status == "Challan Issued",
copies those rider images into data/train/images/ (80%) and data/valid/images/ (20%),
then runs best.pt on each image to auto-generate YOLO label .txt files.

Run automatically by the CI pipeline, or manually:
    python prepare_data.py
"""

import os
import shutil
import csv
import random
import cv2
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = os.path.dirname(os.path.abspath(__file__))
CERTAIN_CSV   = os.path.join(ROOT, "violation_certain.csv")
MODEL_PATH    = os.path.join(ROOT, "model", "best.pt")
TRAIN_IMAGES  = os.path.join(ROOT, "data", "train", "images")
TRAIN_LABELS  = os.path.join(ROOT, "data", "train", "labels")
VALID_IMAGES  = os.path.join(ROOT, "data", "valid", "images")
VALID_LABELS  = os.path.join(ROOT, "data", "valid", "labels")

for d in [TRAIN_IMAGES, TRAIN_LABELS, VALID_IMAGES, VALID_LABELS]:
    os.makedirs(d, exist_ok=True)

VIOLATION_IMGS = os.path.join(ROOT, "violation_imgs")

def resolve_image_path(raw_path):
    """Try the stored path first; if not found (e.g. absolute Windows path on Linux runner),
    fall back to looking for the filename inside violation_imgs/."""
    if raw_path and os.path.exists(raw_path):
        return raw_path
    fname = os.path.basename(raw_path.replace("\\", "/"))
    fallback = os.path.join(VIOLATION_IMGS, fname)
    return fallback if os.path.exists(fallback) else None

# ── Read challan-issued rider images ──────────────────────────────────────────
rider_images = []
with open(CERTAIN_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("Status", "").strip() == "Challan Issued":
            img_path = resolve_image_path(row.get("RiderImage", "").strip())
            if img_path:
                rider_images.append(img_path)

if not rider_images:
    print("No challan-issued images found. Nothing to prepare.")
    raise SystemExit(0)

print(f"Found {len(rider_images)} challan-issued image(s).")

# ── 80/20 train/valid split ───────────────────────────────────────────────────
random.seed(42)
random.shuffle(rider_images)
split = max(1, int(len(rider_images) * 0.8))
splits = {
    "train": rider_images[:split],
    "valid": rider_images[split:] if len(rider_images) > 1 else rider_images[:1],
}

# ── Load model for auto-labelling ─────────────────────────────────────────────
model = YOLO(MODEL_PATH)

def generate_label(img_path, label_path):
    """Run inference and save YOLO-format .txt label file."""
    results = model(img_path, verbose=False)[0]
    h, w = results.orig_shape
    lines = []
    for box in results.boxes:
        cls  = int(box.cls[0])
        xc, yc, bw, bh = box.xywhn[0].tolist()   # already normalised
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(label_path, "w") as f:
        f.write("\n".join(lines))

# ── Copy images and generate labels ──────────────────────────────────────────
for split_name, paths in splits.items():
    img_dir   = TRAIN_IMAGES if split_name == "train" else VALID_IMAGES
    lbl_dir   = TRAIN_LABELS if split_name == "train" else VALID_LABELS
    for src in paths:
        fname     = os.path.basename(src)
        dst_img   = os.path.join(img_dir, fname)
        dst_lbl   = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        shutil.copy2(src, dst_img)
        generate_label(dst_img, dst_lbl)
        print(f"[{split_name}] {fname} → labelled")

print("Data preparation complete.")
print(f"  Train: {len(splits['train'])} image(s)")
print(f"  Valid: {len(splits['valid'])} image(s)")
