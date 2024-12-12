import requests
from PIL import Image, PngImagePlugin
import cv2
from ultralytics.utils import LOGGER

from util1 import (SERVER_URL, HEATMAP_ENDPOINT, REPORT_ENDPOINT,
                   open_csv_file, export_to_local_csv, export_to_local_txt)


class DataProvider:
    def __init__(self, cloudinary_service, video_manager):
        self.start_time = None

        # Local saved reports to be sent to server.
        self.reports = []

        # Custom used objects.
        self.video_manager = video_manager
        self.cloudinary_service = cloudinary_service

        # Initialize CSV file for program documentation.
        open_csv_file()

    def local_save_metrics(self, count, ages, dwell_times, genders, start_time, past_customers, end_time):
        # Initialize counters
        age_groups = {"young": 0, "children": 0, "adult": 0, "elder": 0}
        gender_counts = {"male": 0, "female": 0}
        total_dwell = 0

        # Process past customers
        for customer in past_customers:
            age, gender = customer["age"][0], customer["gender"][0]
            total_dwell += customer["dwell"]["dwell"].total_seconds()

            # Increment gender and age group counts
            gender_counts[gender] += 1
            age_groups[age] += 1

        # Process current customers from `ages` and `genders`
        for track_id in ages:
            age_groups[ages[track_id][0]] += 1

        for track_id in genders:
            gender_counts[genders[track_id][0]] += 1

        # Calculate dwell times for ongoing customers
        total_dwell += sum(
            (end_time - dwell_times[track_id]["entrance"]).total_seconds()
            for track_id in dwell_times
        )

        # Calculate average dwell time
        avg_dwell_time = total_dwell / count if count > 0 else 0

        # Append the report
        self.reports.append({
            "date": start_time.strftime("%Y-%m-%d"),
            "timeSlice": f"{start_time.strftime('%H-%M-%S')}-{end_time.strftime('%H-%M-%S')}",
            "totalCustomers": count,
            "totalMaleCustomers": gender_counts["male"],
            "totalFemaleCustomers": gender_counts["female"],
            "avgDwellTime": avg_dwell_time,
            "customersByAge": age_groups,
        })

    def local_save_heatmap(self, frame):
        # Convert OpenCV frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (PIL)
        pil_image = Image.fromarray(frame_rgb)

        # Add metadata to the image (PNG format for text metadata)
        metadata = PngImagePlugin.PngInfo()
        valid_saving_start_time = self.video_manager.get_start_time().strftime("%Y-%m-%d_%H-%M-%S")
        valid_saving_end_time = self.video_manager.get_end_time().strftime("%Y-%m-%d_%H-%M-%S")

        metadata.add_text("Start Time", valid_saving_start_time)
        metadata.add_text("End Time", valid_saving_end_time)

        # Save the frame with metadata
        image_path = "./logs/heatmaps_snapshots/" + valid_saving_start_time + "_" + valid_saving_end_time + "_heatmap.png"  # Customize the path as needed
        pil_image.save(image_path, "PNG", pnginfo=metadata)
        return image_path, valid_saving_start_time + "_" + valid_saving_end_time

    def local_save(self, count, ages, dwell_times, genders, start_time, past_customers):
        end_time = self.video_manager.get_current_time()
        self.local_save_metrics(count, ages, dwell_times, genders, start_time, past_customers, end_time)

    def provide_heatmap(self, frame):
        image_path, public_image_id = self.local_save_heatmap(frame)
        url = self.cloudinary_service.upload_heatmap(image_path, public_image_id)
        print("****2. Upload an image****\nDelivery URL: ", url, "\n")

    def provide_metrics(self, url):

        # Export outputs locally for internal debugging of the application.
        export_to_local_csv(self.reports)
        export_to_local_txt(self.reports)

        # Data in the format that the server expect
        data = {"reports": self.reports, "jobId": self.video_manager.get_job_id()}
        response = requests.post(url, json=data)

        # Handle response
        if response.status_code == 200:
            LOGGER.info("Successfully saved data")
        else:
            LOGGER.info("Failed to save data")

    def provide(self, last_frame):
        self.provide_metrics(SERVER_URL + REPORT_ENDPOINT)
        self.provide_heatmap(last_frame)

    def set_start_time(self, start_time):
        self.start_time = start_time

    def download_video(self, url):
        try:
            return self.cloudinary_service.download_video(url)
        except Exception as e:
            print(e)
            raise
