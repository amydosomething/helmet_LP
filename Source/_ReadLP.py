import numpy as np
import cv2
import re
import Preprocess
from paddleocr import PaddleOCR

# Initialize PaddleOCR once
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Indian license plate formats:
# New format: MH12AB1234  (state code + district + series + number)
# Old format: MH12A1234
PLATE_PATTERNS = [
    r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}',   # MH12AB1234  (10 chars)
    r'[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}',    # MH12A1234   (9 chars)
    r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{3}',    # MH12AB123   (9 chars)
]

# Positional OCR correction maps
# In digit positions: letter lookalikes → correct digit
DIGIT_CORR  = {'O': '0', 'I': '1', 'Z': '2', 'A': '4', 'S': '5',
                'G': '6', 'B': '8', 'D': '0', 'Q': '0', 'T': '1'}
# In letter positions: digit lookalikes → correct letter
LETTER_CORR = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}

# All valid Indian RTO state/UT codes
VALID_STATE_CODES = {
    'AN','AP','AR','AS','BR','CG','CH','DD','DL','DN','GA','GJ','HP',
    'HR','JH','JK','KA','KL','LA','LD','MH','ML','MN','MP','MZ','NL',
    'OD','PB','PY','RJ','SK','TN','TR','TS','UK','UP','WB'
}
# Digit→letter substitutions only: OCR reads a digit where the state code needs a letter
# e.g. '0D' → 'OD' (Odisha), '1H' → 'IH' won't fix to a valid code → goes to uncertain
STATE_DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '5': 'S', '6': 'G', '8': 'B'}


def preprocess_plate(plate_img):
    """Preprocess plate image before passing to PaddleOCR."""
    plate_img    = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img_gray     = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    img_contrast = Preprocess.maximizeContrast(img_gray)   # tophat+blackhat enhancement
    img_blurred  = cv2.GaussianBlur(img_contrast, (3, 3), 0)  # light smoothing before OCR
    return cv2.cvtColor(img_blurred, cv2.COLOR_GRAY2BGR)

def clean_text(text):
    """Remove all characters except letters and digits, uppercase."""
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def correct_plate(text):
    """
    Apply positional OCR corrections based on expected char type at each position.
    Indian plate structure: [2 letters][2 digits][1-2 letters][3-4 digits]
    Only corrects if length matches a known format (9 or 10 chars).
    """
    n = len(text)
    if n == 10:  # 2L 2D 2L 4D  e.g. MH12AB1234
        return (
            ''.join(LETTER_CORR.get(c, c) for c in text[0:2]) +
            ''.join(DIGIT_CORR.get(c, c)  for c in text[2:4]) +
            ''.join(LETTER_CORR.get(c, c) for c in text[4:6]) +
            ''.join(DIGIT_CORR.get(c, c)  for c in text[6:10])
        )
    if n == 9:   # 2L 2D 1L 4D  e.g. MH12A1234
        return (
            ''.join(LETTER_CORR.get(c, c) for c in text[0:2]) +
            ''.join(DIGIT_CORR.get(c, c)  for c in text[2:4]) +
            ''.join(LETTER_CORR.get(c, c) for c in text[4:5]) +
            ''.join(DIGIT_CORR.get(c, c)  for c in text[5:9])
        )
    return text  # unknown length — regex will reject it

def match_indian_plate(text):
    """Return matched plate string, or empty string if no pattern fits."""
    for pattern in PLATE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group()
    return ""   # no match — don't return garbage

def validate_state_code(plate):
    """
    Check first two chars form a valid Indian state code.
    Only tries digit→letter fixes (e.g. '0D'→'OD') — no letter→letter guessing.
    Returns corrected plate if state code is valid, or "" if it cannot be fixed.
    "" causes the violation to be logged as uncertain (human review needed).
    """
    if not plate or len(plate) < 2:
        return ""
    state = plate[:2]
    if state in VALID_STATE_CODES:
        return plate
    # Try digit→letter substitution at position 0 and position 1
    for i in range(2):
        ch = state[i]
        if ch in STATE_DIGIT_TO_LETTER:
            candidate = state[:i] + STATE_DIGIT_TO_LETTER[ch] + state[i+1:]
            if candidate in VALID_STATE_CODES:
                return candidate + plate[2:]
    return ""  # state code unrecognisable — treat as uncertain

def extract_text_from_result(result):
    """Handle both old PaddleOCR (<3.0) and new PaddleOCR (>=3.0) result formats."""
    raw_text = ""
    try:
        if not result:
            return raw_text
        # New v3 format: list of dicts with 'rec_text' key
        if isinstance(result[0], dict):
            raw_text = ''.join(result[0].get('rec_text', []))
        # Old format: result[0] is list of [box, (text, score)]
        elif isinstance(result[0], list):
            for line in result[0]:
                if line and len(line) >= 2:
                    raw_text += line[1][0]
    except Exception:
        pass
    return raw_text

def ReadLP(plate_img):
    """
    Main function — takes a plate image.
    Returns (plate_text, preprocessed_img).
    preprocessed_img is the image fed into OCR (saved as evidence in violation log).
    """
    try:
        processed = preprocess_plate(plate_img)
        result    = ocr.ocr(processed, cls=True)
        raw_text  = extract_text_from_result(result)
        cleaned   = clean_text(raw_text)
        corrected = correct_plate(cleaned)
        plate     = match_indian_plate(corrected)
        plate     = validate_state_code(plate)
        print(f"[OCR] Raw: {raw_text} | Cleaned: {cleaned} | Corrected: {corrected} | Plate: {plate or 'no match'}")
        return plate, processed
    except Exception as e:
        print(f"[OCR Error] {e}")
        return "", None
