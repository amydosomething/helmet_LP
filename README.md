# Helmet Violation Detection — YOLOv8 + PaddleOCR

Detects motorbike riders not wearing helmets in traffic video, reads their license plate via OCR, looks up the registered owner, and logs violations for review through an admin dashboard.

## Models
| # | Model | Purpose |
|---|-------|---------|
| 1 | YOLOv8s | Detect riders, helmets, no-helmets, and license plates |
| 2 | PaddleOCR | Read license plate text |

## Pipeline
1. YOLOv8 tracks riders across frames with ByteTrack
2. Helmet/no-helmet status is determined per rider via majority vote
3. License plate is cropped, contrast-enhanced, and passed to PaddleOCR
4. Validated plate is looked up in `violations.csv` to identify the owner
5. Violation (certain or uncertain) is logged with evidence images
6. Admin reviews violations via the Streamlit dashboard (`dashboard.py`)

## Run

```bash
# Detection
python Source/_Mainn.py

# Admin dashboard
streamlit run dashboard.py
```

## Dataset
- Rider, Helmet & LP: https://universe.roboflow.com/mmoiz-17l7j/rider-8o9gz

