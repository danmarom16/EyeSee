"""
This module defines reusable constants and enumerations for tracking
and analysis operations in the project.
"""
import os
from enum import Enum
import cv2
import numpy as np
import logging
import csv

# ENUMS
class ExitType(Enum):
    CLEAN = "Clean Exit"  # Exited through the designated door.
    DIRTY = "Dirty Exit"  # Left the frame without a proper exit.


class EntranceType(Enum):
    INIT = "Initial Detection"
    CLEAN = "Entered Cleanly"
    DIRTY = "Entered from the Side"


class ResponseStatus(Enum):
    OK = 200
    ERROR = 503


class WeightsPath(Enum):
    PERSON_TRACKER = 'weights/yolo11n.pt'
    AGE_CLASSIFIER = 'weights/cctv-age-classifier.pt'
    GENDER_CLASSIFIER = 'weights/cctv-gender-classifier.pt'


class CountType(Enum):
    DIRTY_IN = "DIRTY_IN"
    DIRTY_OUT = "DIRTY_OUT"
    CLEAN_IN = "CLEAN_IN"
    CLEAN_OUT = "CLEAN_OUT"


class PastCustomer(Enum):
    TRACK_ID = "track_id"
    DWELL = "dwell"
    AGE = "age"
    GENDER = "gender"


class DwellTime(Enum):
    ENTRANCE = "entrance"
    EXIT = "exit"
    DWELL = "dwell"
    ENTRANCE_TYPE = "entrance_type"
    EXIT_TYPE = "exit_type"

class ClassifierType(Enum):
    AGE = "age"
    GENDER = "gender"


# CONSTANTS
REEVALUATION_INTERVAL = 5  # Frames interval for re-evaluating predictions.
LOW_CONF = 0.75  # Confidence threshold for predictions.
SAVING_INTERVAL = 5  # Number of frames between saving data.
NOT_DETECTED = "Not Detected"

RED = (255, 0, 0)
BLUE = (0, 0, 255)
PURPLE = (104, 9, 123)

SERVER_URL = 'http://127.0.0.1:4000'
HEATMAP_ENDPOINT = '/heatmap'
REPORT_ENDPOINT = '/report/create'
HEATMAP_ADD = '/heatmap/add'

ROW_TITLES = ["date", "timeSlice", "totalCustomers", "totalMaleCustomers", "totalFemaleCustomers",
                 "avgDwellTime", "customersByAge"]

CSV_PATH = "./logs/files/aggregated_data.csv"
TXT_PATH = "./logs/files/aggregated_data.txt"


def normalize_heatmap(im0, heatmap, colormap):
    return cv2.addWeighted(
        im0,
        0.5,
        cv2.applyColorMap(
            cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), colormap
        ),
        0.5,
        0,
    )


def box_to_image(image, bounding_box):
    """
    Extracts a bounding box region from an image and returns it as a YOLO-compatible input.

    Parameters:
        image (np.ndarray): The original image (in NumPy array format, e.g., read by OpenCV).
        bounding_box (tuple): The bounding box coordinates in the format (x1, y1, x2, y2).

    Returns:
        np.ndarray: The extracted region as a YOLO-compatible image with uint8 data type.
    """
    # Unpack bounding box coordinates
    x1, y1, x2, y2 = bounding_box

    # Ensure bounding box coordinates are integers and within the image dimensions
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    # Crop the region of interest (ROI) from the image
    cropped_image = image[y1:y2, x1:x2]

    # Resize to YOLO's input size (e.g., 640x640, adjust as needed)
    resized_image = cv2.resize(cropped_image, (640, 640))

    # Convert to uint8 (YOLO requires this)
    resized_image_uint8 = (resized_image * 255).astype(
        np.uint8) if resized_image.dtype == np.float32 else resized_image

    return resized_image_uint8


def annotate_object(track_id, cls, box, age, gender, annotator, color):
    label = f"{cls} id:{track_id} age:{age} gender: {gender}"  # Label includes class name and track ID
    annotator.box_label(box, label, color=color)  # Red bounding box for better visibility


def init_writer(path, video_manager):
    return cv2.VideoWriter(
        path,
        cv2.VideoWriter.fourcc(*"mp4v"),
        float(video_manager.get_fps()),
        (int(video_manager.get_width()), int(video_manager.get_height()))
    )


def get_logger():
    return logging.basicConfig(
        filename='./logs/id_log.txt',  # Log file name
        level=logging.INFO,  # Log level
        format='%(message)s'  # Log format
    )


# Create the CSV file and write the header if it doesn't exist
def open_csv_file(path=CSV_PATH):
    if not os.path.exists(path):
        with open(path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(ROW_TITLES)



def export_to_local_csv(reports, path=CSV_PATH):
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for report in reports:
            writer.writerow([
                report["date"],
                report["timeSlice"],
                report["totalCustomers"],
                report["totalMaleCustomers"],
                report["totalFemaleCustomers"],
                report["avgDwellTime"],
                report["customersByAge"]
            ])


def export_to_local_txt(reports, path=TXT_PATH):
    with open(path + ".txt", mode='a', encoding='utf-8') as file:
        for report in reports:
            line = "\t".join([
                str(report["date"]),
                str(report["timeSlice"]),
                str(report["totalCustomers"]),
                str(report["totalMaleCustomers"]),
                str(report["totalFemaleCustomers"]),
                str(report["avgDwellTime"]),
                str(report["customersByAge"])
            ])
            file.write(line + "\n")