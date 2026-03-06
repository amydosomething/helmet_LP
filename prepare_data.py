"""
prepare_data.py
---------------
Two sources of training data:

1. violation_certain.csv  — rows with Status == "Challan Issued"
   → confirmed nohelmet violations; auto-labelled with best.pt

2. violation_rejected.csv — rows rejected by the admin (e.g. "helmet present")
   → false positives; nohelmet boxes are relabelled as helmet (corrective training)

Run automatically by the CI pipeline, or manually:
    python prepare_data.py
"""

import os
import shutil
import csv
import random
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = os.path.dirname(os.path.abspath(__file__))
CERTAIN_CSV   = os.path.join(ROOT, "violation_certain.csv")
REJECTED_CSV  = os.path.join(ROOT, "violation_rejected.csv")
MODEL_PATH    = os.path.join(ROOT, "model", "best.pt")
TRAIN_IMAGES  = os.path.join(ROOT, "data", "train", "images")
TRAIN_LABELS  = os.path.join(ROOT, "data", "train", "labels")
VALID_IMAGES  = os.path.join(ROOT, "data", "valid", "images")
VALID_LABELS  = os.path.join(ROOT, "data", "valid", "labels")

for d in [TRAIN_IMAGES, TRAIN_LABELS, VALID_IMAGES, VALID_LABELS]:
    os.makedirs(d, exist_ok=True)

VIOLATION_IMGS = os.path.join(ROOT, "violation_imgs")

CLS_NOHELMET = 2   # class indices from data.yaml: 0=rider,1=helmet,2=nohelmet,3=numberplate
CLS_HELMET   = 1

def resolve_image_path(raw_path):
    """Try the stored path first; if not found (e.g. absolute Windows path on Linux runner),
    fall back to looking for the filename inside violation_imgs/."""
    if raw_path and os.path.exists(raw_path):
        return raw_path
    fname = os.path.basename(raw_path.replace("\\", "/"))
    fallback = os.path.join(VIOLATION_IMGS, fname)
    return fallback if os.path.exists(fallback) else None

# ── Read challan-issued images ────────────────────────────────────────────────
rider_images = []
with open(CERTAIN_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("Status", "").strip() == "Challan Issued":
            img_path = resolve_image_path(row.get("RiderImage", "").strip())
            if img_path:
                rider_images.append(img_path)

# ── Read rejected (false-positive) images ─────────────────────────────────────
rejected_images = []
if os.path.exists(REJECTED_CSV):
    with open(REJECTED_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_path = resolve_image_path(row.get("RiderImage", "").strip())
            if img_path:
                rejected_images.append(img_path)

# ── Exit only if both sources are empty ───────────────────────────────────────
if not rider_images and not rejected_images:
    print("No training data found. Nothing to prepare.")
    raise SystemExit(0)

print(f"Found {len(rider_images)} challan-issued image(s).")
print(f"Found {len(rejected_images)} rejected (false-positive) image(s).")

# ── Load model ────────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)

def generate_label(img_path, label_path):
    """Auto-label with best.pt, keeping original class predictions."""
    results = model(img_path, verbose=False)[0]
    lines = []
    for box in results.boxes:
        cls = int(box.cls[0])
        xc, yc, bw, bh = box.xywhn[0].tolist()
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(label_path, "w") as f:
        f.write("\n".join(lines))

def generate_corrective_label(img_path, label_path):
    """Auto-label but flip nohelmet → helmet (human said it was actually a helmet)."""
    results = model(img_path, verbose=False)[0]
    lines = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == CLS_NOHELMET:
            cls = CLS_HELMET
        xc, yc, bw, bh = box.xywhn[0].tolist()
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(label_path, "w") as f:
        f.write("\n".join(lines))

# ── 80/20 split and label challan images ──────────────────────────────────────
splits = {"train": [], "valid": []}
if rider_images:
    random.seed(42)
    random.shuffle(rider_images)
    split = max(1, int(len(rider_images) * 0.8))
    splits["train"] = rider_images[:split]
    splits["valid"] = rider_images[split:] if len(rider_images) > 1 else rider_images[:1]

    for split_name, paths in splits.items():
        img_dir = TRAIN_IMAGES if split_name == "train" else VALID_IMAGES
        lbl_dir = TRAIN_LABELS if split_name == "train" else VALID_LABELS
        for src in paths:
            fname   = os.path.basename(src)
            dst_img = os.path.join(img_dir, fname)
            dst_lbl = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
            shutil.copy2(src, dst_img)
            generate_label(dst_img, dst_lbl)
            print(f"[{split_name}] {fname} → labelled")

# ── Rejected frames → corrective helmet labels (train only) ───────────────────
for src in rejected_images:
    fname   = os.path.basename(src)
    dst_img = os.path.join(TRAIN_IMAGES, "rej_" + fname)   # rej_ prefix avoids filename collision
    dst_lbl = os.path.join(TRAIN_LABELS, "rej_" + os.path.splitext(fname)[0] + ".txt")
    shutil.copy2(src, dst_img)
    generate_corrective_label(dst_img, dst_lbl)
    print(f"[rejected→helmet] {fname} relabelled")

print("\nData preparation complete.")
print(f"  Challan train : {len(splits['train'])} image(s)")
print(f"  Challan valid : {len(splits['valid'])} image(s)")
print(f"  Corrective    : {len(rejected_images)} image(s)")
