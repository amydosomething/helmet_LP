from ultralytics import YOLO
from collections import Counter
import os
import cv2
import numpy as np

import _LP_Helmet
import _ReadLP
import _myFunc

# ---------------------------------- CONFIG ----------------------------------
MODEL_PATH = r"d:\NOTES\amrin notes\ML\FINAL\helmet_LP\model\best.pt"

# Directory to temporarily save frames for processing
TRAFFIC_DIR = os.path.join(os.path.dirname(__file__), "img", "Traffic")

# Owner lookup CSV (plate → name/email)
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "violations.csv")

# Violation log CSVs — certain = plate matched Indian regex, uncertain = unreadable/no match
CERTAIN_LOG   = os.path.join(os.path.dirname(__file__), "..", "violation_certain.csv")
UNCERTAIN_LOG = os.path.join(os.path.dirname(__file__), "..", "violation_uncertain.csv")

# ---------------------------------- INIT MODEL ----------------------------------

def Init_Model():
    model = YOLO(MODEL_PATH)
    print(f"[Model] Loaded: {MODEL_PATH}")
    return model

# ---------------------------------- HELPERS ----------------------------------

def majority_vote(readings):
    """
    Pick the most frequent non-empty plate reading from a list of
    (text, preprocessed_img, rider_crop, plate_crop, extended_crop) tuples.
    Returns all five from the same frame.
    For uncertain violations, extended_crop shows rider + bike area.
    winning_text is "" if all readings were empty.
    """
    valid = [r for r in readings if r[0]]
    if not valid:
        # All unreadable — use middle frame: rider most centered there
        mid = readings[len(readings) // 2]
        return "", mid[1], mid[2], mid[3], mid[4]
    counts = Counter(r[0] for r in valid)
    best_text = counts.most_common(1)[0][0]
    best = next(r for r in valid if r[0] == best_text)
    return best[0], best[1], best[2], best[3], best[4]


def finalize_rider(rider_id, data, current_frame):
    """
    Called when a rider leaves the frame or video ends.
    Runs majority vote, looks up owner, saves images, logs to correct CSV.
    """
    if not data['violation']:
        return

    plate_text, preprocessed_img, rider_crop, raw_plate_img, extended_crop = majority_vote(data['plate_readings'])
    first_frame = data['first_violation_frame']

    if plate_text:
        # Plate matched Indian regex + valid state code — certain violation
        # Use tight rider crop + preprocessed plate image as evidence
        name, gmail = _myFunc.get_client_info(plate_text, CSV_PATH)
        print(f"[VIOLATION-CERTAIN] Rider #{rider_id} | Frame {first_frame} | "
              f"Plate: {plate_text} | Owner: {name or 'Unknown'}")
        _myFunc.log_violation(first_frame, plate_text, name, gmail, CERTAIN_LOG,
                              rider_img=rider_crop, plate_img=preprocessed_img)
    else:
        # Plate unreadable or invalid state code — uncertain, needs human review.
        # Use extended crop (rider + bike area) so reviewer sees the full bike.
        # For plate: raw colour plate crop if YOLO found one, else extended crop.
        evidence_rider = extended_crop if extended_crop is not None and extended_crop.size > 0 else rider_crop
        evidence_plate = raw_plate_img if raw_plate_img is not None else evidence_rider
        print(f"[VIOLATION-UNCERTAIN] Rider #{rider_id} | Frame {first_frame} | Plate: Unreadable")
        _myFunc.log_violation(first_frame, "", None, None, UNCERTAIN_LOG,
                              rider_img=evidence_rider, plate_img=evidence_plate)

# ---------------------------------- MAIN PROGRAM ----------------------------------

def Program(model, path, start_frame=0, stop_frame=0):
    """
    Process a video file frame by frame with rider tracking and majority-vote OCR.

    Each rider gets a persistent ID via ByteTrack. For violating riders (nohelmet
    detected), plate OCR results are collected across all frames they appear in.
    When a rider disappears, majority vote picks the best plate reading and logs once.

    Args:
        model       : Loaded YOLO model
        path        : Path to input video file
        start_frame : Frame number to start processing from (0 = beginning)
        stop_frame  : Frame number to stop at (0 = process till end)

    Returns:
        current_frame : Total frames processed
    """
    stop_flag = False

    # rider_tracker[rider_id] = {
    #   'violation'           : bool,
    #   'first_violation_frame': int,
    #   'rider_crop_first'    : np.array,   # rider crop from first violation frame
    #   'plate_readings'      : [(text, preprocessed_img), ...],
    #   'last_seen'           : int,        # video frame number last detected
    # }
    rider_tracker = {}

    cap = cv2.VideoCapture(path)
    seconds_interval = 0.1          # sample one frame every 0.2s → ~5 frames/sec
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_extract = max(1, int(fps * seconds_interval))
    # A rider is considered gone if not seen for 3 sample intervals (~0.6s)
    disappear_threshold = frames_to_extract * 3

    current_frame = 0
    frame_count   = 0

    _myFunc.FilePreProcess(TRAFFIC_DIR)
    _myFunc.init_owner_csv(CSV_PATH)

    while cap.isOpened() and not stop_flag:
        success, frame = cap.read()
        if not success:
            break

        if frame_count == 0:
            if stop_frame != 0 and current_frame < stop_frame:
                current_frame += 1
                frame_count = (frame_count + 1) % frames_to_extract
                continue

            # Save frame to disk for YOLO
            frame_path = os.path.join(TRAFFIC_DIR, f"frame{current_frame:04d}.jpg")
            cv2.imwrite(frame_path, frame)

            # Run detection + tracking
            rider_detections, annotated_frame = _LP_Helmet.image_detect(model, frame_path)

            active_ids = set()

            for rider in rider_detections:
                rid           = rider['rider_id']
                violation     = rider['violation']
                rider_crop    = rider['rider_crop']
                extended_crop = rider['extended_crop']
                plate_crop    = rider['plate_crop']

                active_ids.add(rid)

                # Init tracker entry on first appearance
                if rid not in rider_tracker:
                    rider_tracker[rid] = {
                        'violation'            : False,
                        'first_violation_frame': None,
                        'plate_readings'       : [],
                        'last_seen'            : current_frame,
                    }

                rider_tracker[rid]['last_seen'] = current_frame

                if violation:
                    if not rider_tracker[rid]['violation']:
                        # First frame we see this rider violating
                        rider_tracker[rid]['violation']            = True
                        rider_tracker[rid]['first_violation_frame'] = current_frame

                    # Run OCR on plate crop; store 5-tuple so majority-vote winner
                    # frame supplies all evidence images.
                    # extended_crop = rider box extended down to include bike+plate area.
                    if plate_crop is not None and plate_crop.size > 0:
                        plate_text, preprocessed = _ReadLP.ReadLP(plate_crop)
                        rider_tracker[rid]['plate_readings'].append(
                            (plate_text, preprocessed, rider_crop, plate_crop, extended_crop))
                    else:
                        rider_tracker[rid]['plate_readings'].append(
                            ("", None, rider_crop, None, extended_crop))

            # Check for riders that have disappeared (not seen this frame)
            gone_ids = [rid for rid, data in rider_tracker.items()
                        if rid not in active_ids
                        and current_frame - data['last_seen'] >= disappear_threshold]

            for rid in gone_ids:
                finalize_rider(rid, rider_tracker[rid], current_frame)
                del rider_tracker[rid]

            # Display annotated frame
            if annotated_frame is not None:
                cv2.imshow("Helmet Violation Detection", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    stop_flag = True

        current_frame += 1
        frame_count = (frame_count + 1) % frames_to_extract

    cap.release()
    cv2.destroyAllWindows()

    # Finalize any riders still in tracker at end of video
    print("[Info] Video ended — finalizing remaining tracked riders...")
    for rid, data in rider_tracker.items():
        finalize_rider(rid, data, current_frame)

    return current_frame

# -------------------------------------------------------- TEST --------------------------------------------------------
# Uncomment to test directly:
# if __name__ == "__main__":
#     model = Init_Model()
#     Program(model, path=r"path\to\your\video.mp4")
