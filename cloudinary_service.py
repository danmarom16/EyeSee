import os
import requests
import cloudinary
from cloudinary import CloudinaryImage
import cloudinary.uploader
import cloudinary.api
import numpy as np
from ultralytics.utils import LOGGER
import cv2

config = cloudinary.config(secure=True)


class CloudinaryService:
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        self.config = cloudinary.config(
            cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
            api_key=os.getenv('CLOUDINARY_API_KEY'),
            api_secret=os.getenv('CLOUDINARY_API_SECRET'),
            secure=True
        )

        LOGGER.info(
            f"****1. Set up and configure the SDK:****\nCredentials: {self.config.cloud_name}, {self.config.api_key}\n")

    def upload_heatmap(self, heatmap_path, public_image_id):
        # Upload the image.
        # Set the asset's public ID and allow overwriting the asset with new versions
        cloudinary.uploader.upload(heatmap_path, public_id=public_image_id,asset_folder="heatmaps",
                                   unique_filename=False, overwrite=True)

        # Build the URL for the image and save it in the variable 'srcURL'
        srcURL = CloudinaryImage(public_image_id).build_url()

        return srcURL

    def download_video(self, url, save_path="downloaded_video.mp4"):
        # Fetch video from URL
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            print("Successfully saved data")

            # Save video to a temporary file
            with open(save_path, "wb") as video_file:
                video_file.write(response.content)

            # Load the video using OpenCV
            video_capture = cv2.VideoCapture(save_path)

            # Check if the video was loaded successfully
            if not video_capture.isOpened():
                raise Exception("Failed to load video. Check the file format or path.")

            return video_capture
        elif response.status_code == 404:
            raise Exception("Failed to save data. URL not found.")
        else:
            raise Exception(f"Failed to fetch data. HTTP Status Code: {response.status_code}")

