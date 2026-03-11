"""
prepare_data.py
---------------
Two sources of training data:

1. violation_certain.csv  — rows with Status == "Challan Issued" and Trained != "yes"
   → confirmed nohelmet violations; auto-labelled with best.pt

2. violation_rejected.csv — rows with Trained != "yes"
   → false positives; nohelmet boxes are relabelled as helmet (corrective training)

After processing, each row is marked Trained=yes so it is never used again.
The updated CSVs are saved and committed back to the repo by CI step 10.

Run automatically by the CI pipeline, or manually:
    python prepare_data.py
"""

import os
import shutil
import random
import pandas as pd
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
    if not raw_path:
        return None
    raw_path = str(raw_path).strip()
    if os.path.exists(raw_path):
        return raw_path
    fname = os.path.basename(raw_path.replace("\\", "/"))
    fallback = os.path.join(VIOLATION_IMGS, fname)
    return fallback if os.path.exists(fallback) else None

# ── Load CSVs, ensure Trained column exists ───────────────────────────────────
certain_df = pd.read_csv(CERTAIN_CSV) if os.path.exists(CERTAIN_CSV) else pd.DataFrame()
if not certain_df.empty and "Trained" not in certain_df.columns:
    certain_df["Trained"] = ""

rejected_df = pd.read_csv(REJECTED_CSV) if os.path.exists(REJECTED_CSV) else pd.DataFrame()
if not rejected_df.empty and "Trained" not in rejected_df.columns:
    rejected_df["Trained"] = ""

# ── Filter to only untrained rows ─────────────────────────────────────────────
def untrained(df):
    if df.empty:
        return df
    return df[df["Trained"].astype(str).str.strip().str.lower() != "yes"]

new_certain  = untrained(certain_df[certain_df.get("Status", pd.Series(dtype=str)).astype(str).str.strip() == "Challan Issued"]) \
               if not certain_df.empty and "Status" in certain_df.columns else pd.DataFrame()
new_rejected = untrained(rejected_df) if not rejected_df.empty else pd.DataFrame()

rider_images   = []   # (index_in_certain_df, path)
rejected_pairs = []   # (index_in_rejected_df, path)

for idx, row in new_certain.iterrows():
    p = resolve_image_path(row.get("RiderImage", ""))
    if p:
        rider_images.append((idx, p))

for idx, row in new_rejected.iterrows():
    p = resolve_image_path(row.get("RiderImage", ""))
    if p:
        rejected_pairs.append((idx, p))

# ── Exit only if both sources are empty ───────────────────────────────────────
if not rider_images and not rejected_pairs:
    print("No new training data found. Nothing to prepare.")
    with open(".has_training_data", "w") as f:
        f.write("false")
    raise SystemExit(0)

print(f"New challan-issued image(s)  : {len(rider_images)}")
print(f"New rejected (corrective)    : {len(rejected_pairs)}")

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
    s = max(1, int(len(rider_images) * 0.8))
    splits["train"] = rider_images[:s]
    splits["valid"] = rider_images[s:] if len(rider_images) > 1 else rider_images[:1]

    for split_name, pairs in splits.items():
        img_dir = TRAIN_IMAGES if split_name == "train" else VALID_IMAGES
        lbl_dir = TRAIN_LABELS if split_name == "train" else VALID_LABELS
        for idx, src in pairs:
            fname   = os.path.basename(src)
            dst_img = os.path.join(img_dir, fname)
            dst_lbl = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
            shutil.copy2(src, dst_img)
            generate_label(dst_img, dst_lbl)
            certain_df.at[idx, "Trained"] = "yes"   # mark as trained
            print(f"[{split_name}] {fname} → labelled")

# ── Rejected frames → corrective helmet labels (train only) ───────────────────
for idx, src in rejected_pairs:
    fname   = os.path.basename(src)
    dst_img = os.path.join(TRAIN_IMAGES, "rej_" + fname)   # rej_ prefix avoids collision
    dst_lbl = os.path.join(TRAIN_LABELS, "rej_" + os.path.splitext(fname)[0] + ".txt")
    shutil.copy2(src, dst_img)
    generate_corrective_label(dst_img, dst_lbl)
    rejected_df.at[idx, "Trained"] = "yes"   # mark as trained
    print(f"[rejected→helmet] {fname} relabelled")

# ── Save updated CSVs (CI will commit them back) ──────────────────────────────
certain_df.to_csv(CERTAIN_CSV, index=False)
if not rejected_df.empty:
    rejected_df.to_csv(REJECTED_CSV, index=False)

print("\nData preparation complete.")
print(f"  Challan train : {len(splits['train'])} image(s)")
print(f"  Challan valid : {len(splits['valid'])} image(s)")
print(f"  Corrective    : {len(rejected_pairs)} image(s)")
