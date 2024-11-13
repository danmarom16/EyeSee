import csv
import os
import cv2
from PIL import Image, PngImagePlugin
import cv2
import numpy as np
from datetime import datetime


class DataProvider:
    def __init__(self, file_path="./logs/aggregated_data.csv"):
        self.file_path = file_path
        # Create the CSV file and write the header if it doesn't exist
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["start_time", "end_time", "count"])

    def save_data(self, data, frame):

        # Convert OpenCV frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (PIL)
        pil_image = Image.fromarray(frame_rgb)

        # Add metadata to the image (PNG format for text metadata)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Start Time", str(data['start_time']))
        metadata.add_text("End Time", str(data['end_time']))

        valid_saving_start_time = data['start_time'].strftime("%Y-%m-%d_%H-%M-%S")

        # Save the frame with metadata
        image_path = "./logs/" + valid_saving_start_time + "_heatmap.png"  # Customize the path as needed
        pil_image.save(image_path, "PNG", pnginfo=metadata)

        # Write data to CSV
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                data['start_time'],
                data['end_time'],
                data['counter'],
            ])
