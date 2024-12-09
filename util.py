"""
This module defines reusable constants and enumerations for tracking
and analysis operations in the project.
"""

from enum import Enum
import cv2
import numpy as np


# ENUMS
class ExitType(Enum):
    CLEAN = "Clean Exit"  # Exited through the designated door.
    DIRTY = "Dirty Exit"  # Left the frame without a proper exit.


class EntranceType(Enum):
    INIT = "Initial Detection"
    CLEAN = "Entered Cleanly"
    DIRTY = "Entered from the Side"


# CONSTANTS
REEVALUATION_INTERVAL = 5  # Frames interval for re-evaluating predictions.
LOW_CONF = 0.75  # Confidence threshold for predictions.
SAVING_INTERVAL = 5  # Number of frames between saving data.


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


def init_writer(path, video_collector):
    return cv2.VideoWriter(
        path,
        cv2.VideoWriter.fourcc(*"mp4v"),
        float(video_collector.get_fps()),
        (int(video_collector.get_width()), int(video_collector.get_height()))
    )
