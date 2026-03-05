from ultralytics import YOLO
import cv2
import numpy as np
import os

# ================================================================================
# Helper: IoU and association logic to group detections by rider
# ================================================================================

def compute_iou(boxA, boxB):
    """Compute Intersection over Union between two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def box_center_in_rider(rider_box, obj_box):
    """Check if the center of obj_box falls within rider_box."""
    cx = (obj_box[0] + obj_box[2]) / 2
    cy = (obj_box[1] + obj_box[3]) / 2
    return rider_box[0] <= cx <= rider_box[2] and rider_box[1] <= cy <= rider_box[3]

def associate_to_rider(rider_boxes, obj_box):
    """
    Find which rider box best contains or overlaps with obj_box.
    Returns index of best matching rider, or -1 if none.
    """
    best_idx = -1
    best_score = -1

    # Prefer containment first
    for i, rider in enumerate(rider_boxes):
        if box_center_in_rider(rider, obj_box):
            iou = compute_iou(rider, obj_box)
            if iou > best_score:
                best_score = iou
                best_idx = i

    # Fallback to highest IoU if no containment found
    if best_idx == -1:
        for i, rider in enumerate(rider_boxes):
            iou = compute_iou(rider, obj_box)
            if iou > best_score:
                best_score = iou
                best_idx = i

    return best_idx

# ================================================================================
# Main detection function
# ================================================================================

def image_detect(model, path):
    """
    Detect helmet violations using a single YOLOv8s model with ByteTrack tracking.
    Model classes expected: rider, helmet, nohelmet, numberplate

    Args:
        model : Loaded YOLO model
        path  : Path to the image file

    Returns:
        (rider_results, annotated_frame)

        rider_results — list of dicts, one per rider with a valid track ID:
        [{
            'rider_id'   : int,       # Persistent tracker ID for this rider
            'rider_crop' : np.array,  # Cropped rider bounding box image
            'plate_crop' : np.array,  # Raw cropped plate image (empty if none found)
            'violation'  : bool       # True if nohelmet detected for this rider
        }, ...]

        NOTE: OCR is NOT run here. plate_crop is returned raw so _Mainn.py can
        call ReadLP() and accumulate readings across frames for majority voting.
    """
    im = cv2.imread(path)
    if im is None:
        print(f"[Error] Could not read image: {path}")
        return [], None

    # Use track() with persist=True so tracker maintains IDs across frames
    results = model.track(path, imgsz=640, conf=0.3, persist=True, verbose=False)

    # Color scheme: BGR format
    BOX_COLORS = {
        'rider':       (255, 165,   0),   # Orange
        'helmet':      (  0, 200,   0),   # Green
        'nohelmet':    (  0,   0, 255),   # Red
        'numberplate': (  0, 255, 255),   # Cyan
    }

    annotated = im.copy()
    rider_boxes = []   # list of {'bbox', 'crop', 'rider_id'}
    helmets     = []   # list of {'bbox', 'crop'}
    no_helmets  = []   # list of {'bbox', 'crop'}
    plates      = []   # list of {'bbox', 'crop'}

    for r in results:
        for box in r.boxes:
            # Skip boxes with no tracker ID (tracker not yet initialised for this box)
            if box.id is None:
                continue

            cls_name = model.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = im[y1:y2, x1:x2]
            bbox = [x1, y1, x2, y2]
            conf = float(box.conf)
            track_id = int(box.id)

            color = BOX_COLORS.get(cls_name, (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} #{track_id} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if cls_name == 'rider':
                rider_boxes.append({'bbox': bbox, 'crop': crop, 'rider_id': track_id})
            elif cls_name == 'helmet':
                helmets.append({'bbox': bbox, 'crop': crop})
            elif cls_name == 'nohelmet':
                no_helmets.append({'bbox': bbox, 'crop': crop})
            elif cls_name == 'numberplate':
                plates.append({'bbox': bbox, 'crop': crop})

    # Edge case: no riders detected — treat whole frame as one rider with ID -1
    if not rider_boxes:
        print(f"[Warning] No riders detected in {path}, using full frame.")
        rider_boxes.append({'bbox': [0, 0, im.shape[1], im.shape[0]], 'crop': im, 'rider_id': -1})

    # Draw violation banner if any no-helmet detected
    if no_helmets:
        cv2.putText(annotated, "!! VIOLATION DETECTED !!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    # Initialize per-rider result containers.
    # extended_crop = rider box extended downward by rider height to include bike + plate area.
    rider_results = []
    for rb in rider_boxes:
        x1, y1, x2, y2 = rb['bbox']
        h = y2 - y1
        y2_ext = min(y2 + h, im.shape[0])   # extend down by one rider-height, clamped
        extended_crop = im[y1:y2_ext, x1:x2]
        rider_results.append({
            'rider_id'     : rb['rider_id'],
            'rider_crop'   : rb['crop'],
            'extended_crop': extended_crop,
            'plate_crop'   : np.array([]),
            'violation'    : False,
        })

    # Associate helmets → riders (for display only, not stored in results)
    for obj in helmets:
        idx = associate_to_rider([r['bbox'] for r in rider_boxes], obj['bbox'])
        if idx >= 0:
            # Draw helmet label on annotated frame (already drawn above)
            pass

    # Associate nohelmet → riders (marks violation)
    for obj in no_helmets:
        idx = associate_to_rider([r['bbox'] for r in rider_boxes], obj['bbox'])
        if idx >= 0:
            rider_results[idx]['violation'] = True

    # Associate numberplate → riders (store raw crop, NO OCR here).
    # Use extended rider boxes (rider + bike area below) so plates that sit
    # below the person bounding box are still linked to the correct rider.
    extended_rider_boxes = []
    for r in rider_boxes:
        x1, y1, x2, y2 = r['bbox']
        h = y2 - y1
        extended_rider_boxes.append([x1, y1, x2, min(y2 + h, im.shape[0])])

    for obj in plates:
        idx = associate_to_rider(extended_rider_boxes, obj['bbox'])
        if idx >= 0:
            rider_results[idx]['plate_crop'] = obj['crop']

    return rider_results, annotated

