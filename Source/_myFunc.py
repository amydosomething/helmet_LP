import os
import cv2
import pandas as pd
import csv
from datetime import datetime


def get_latest_image_path(parent_dir):
    all_files = []
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(root, file)
                all_files.append((file_path, os.path.getmtime(file_path)))

    if not all_files:
        return None
    latest_image_path = max(all_files, key=lambda x: x[1])[0]
    
    return latest_image_path

# Pre-defined plate → owner database (acts as a mock RTO database)
# Add your known plates and their owners here
OWNER_DB = {
    "MH12AB1234": {"name": "Rahul Sharma",  "gmail": "rahul.sharma@gmail.com",  "address": "Mumbai, Maharashtra"},
    "DL15AE0190": {"name": "Priya Patel",   "gmail": "priya.patel@gmail.com",   "address": "Ahmedabad, Gujarat"},
    "KA03MF4321": {"name": "Amit Kumar",    "gmail": "amit.kumar@gmail.com",    "address": "Delhi"},
    "TN09BZ5678": {"name": "Sneha Reddy",   "gmail": "sneha.reddy@gmail.com",   "address": "Hyderabad, Telangana"},
    "RJ14CD2345": {"name": "Vikram Singh",  "gmail": "vikram.singh@gmail.com",  "address": "Jaipur, Rajasthan"},
    "KL07EF8901": {"name": "Ananya Nair",   "gmail": "ananya.nair@gmail.com",   "address": "Kochi, Kerala"},
    "UP32GH6789": {"name": "Rohit Verma",   "gmail": "rohit.verma@gmail.com",   "address": "Lucknow, Uttar Pradesh"},
    "MH14IJ3456": {"name": "Deepika Joshi", "gmail": "deepika.joshi@gmail.com", "address": "Pune, Maharashtra"},
    "MH12LC9488": {"name": "Rajesh Shah",    "gmail": "rajesh.shah@gmail.com", "address": "Pune, Maharashtra"},
    
}

def init_owner_csv(file_path):
    """Create the owner CSV from OWNER_DB if it doesn't exist."""
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["STT", "Name", "Gmail", "Address", "LP"])
            for i, (lp, info) in enumerate(OWNER_DB.items(), start=1):
                writer.writerow([i, info["name"], info["gmail"], info["address"], lp])


def get_client_info(lp_info, file_path):
    data = pd.read_csv(file_path)
    try:
        row = data[data['LP'] == lp_info].iloc[0]
        ten = row['Name']
        gmail = row['Gmail']
        return ten, gmail
    except IndexError:
        return None, None


def log_violation(frame_num, lp_text, name, gmail, log_path,
                  rider_img=None, plate_img=None):
    """
    Append one violation record to the given log CSV.
    Saves rider and plate images to violation_imgs/ next to the CSV.

    Args:
        frame_num  : Video frame number where violation was first confirmed
        lp_text    : Final plate text (from majority vote), or "" if unreadable
        name       : Owner name from RTO lookup, or None
        gmail      : Owner gmail from RTO lookup, or None
        log_path   : Path to the target CSV (certain or uncertain)
        rider_img  : numpy array — cropped rider image (from first violation frame)
        plate_img  : numpy array — preprocessed plate image (from majority-vote frame)
    """
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    imgs_dir = os.path.join(os.path.dirname(log_path), "violation_imgs")
    os.makedirs(imgs_dir, exist_ok=True)

    rider_img_path = ""
    plate_img_path = ""

    if rider_img is not None and rider_img.size > 0:
        rider_img_path = os.path.join(imgs_dir, f"rider_f{frame_num}_{ts_file}.jpg")
        cv2.imwrite(rider_img_path, rider_img)

    if plate_img is not None and plate_img.size > 0:
        plate_img_path = os.path.join(imgs_dir, f"plate_f{frame_num}_{ts_file}.jpg")
        cv2.imwrite(plate_img_path, plate_img)

    file_exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Frame", "Plate", "Owner", "Gmail",
                             "Status", "RiderImage", "PlateImage"])
        writer.writerow([
            ts,
            frame_num,
            lp_text if lp_text else "Unreadable",
            name  if name  else "Unknown",
            gmail if gmail else "N/A",
            "No Helmet",
            rider_img_path,
            plate_img_path,
        ])


def delete_files(path):
    [os.remove(os.path.join(path, tep_tin)) for tep_tin in os.listdir(path)]

def create_folder(path):
    os.makedirs(path)

def FilePreProcess(path):
    if not os.path.exists(path):
        create_folder(path)
    else:
        delete_files(path)