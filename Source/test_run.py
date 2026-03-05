"""
Quick test script to verify the full pipeline works:
    YOLO (best.pt) → rider association → PaddleOCR → plate text
"""

import sys
import os

# Make sure Source/ imports work
sys.path.insert(0, os.path.dirname(__file__))

import _Mainn

# ---- CONFIG ----
VIDEO_PATH = r"d:\NOTES\amrin notes\ML\FINAL\helmet_LP\Video_Demo\demo.mp4"
# ----------------

if __name__ == "__main__":
    print("=" * 50)
    print("Loading model...")
    model = _Mainn.Init_Model()
    print("Model loaded! Starting detection...\n")
    print("=" * 50)

    total_frames = _Mainn.Program(model, path=VIDEO_PATH)

    print("=" * 50)
    print(f"Done! Processed {total_frames} frames.")
    print("Press Q during video processing to stop early.")
