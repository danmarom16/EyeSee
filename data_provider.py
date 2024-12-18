import requests
from PIL import Image, PngImagePlugin
import cv2
from ultralytics.utils import LOGGER

from util import (SERVER_URL, REPORT_ENDPOINT, HeatmapType, DwellTime, make_dirs,
                  open_csv_file, export_to_local_csv, export_to_local_txt)


class DataProvider:
    def __init__(self, cloudinary_service, video_manager):

        # Local saved reports to be sent to server.
        self.reports = []

        # Video manager who will be responsible to provide the data provider with metadata regarding the video.
        self.video_manager = video_manager

        # Cloudinary service that is in use to upload and download videos and images to a remote cloud.
        self.cloudinary_service = cloudinary_service

        # Initialize base directory for further created outputs of the program.
        self.base_dir = make_dirs()

        # Open CSV file.
        open_csv_file(self.base_dir)

    def local_save_metrics(self, count, ages, dwell_times, genders, past_customers):

        date = self.video_manager.get_date()
        start_time = self.video_manager.get_current_timeslice_start()
        end_time = self.video_manager.get_current_time()

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
            (end_time - dwell_times[track_id][DwellTime.ENTRANCE.value]).total_seconds()
            for track_id in dwell_times
        )

        # Calculate average dwell time
        avg_dwell_time = total_dwell / count if count > 0 else 0

        # Append the report
        self.reports.append({
            "date": date,
            "timeSlice": f"{start_time.strftime('%H-%M-%S')}-{end_time.strftime('%H-%M-%S')}",
            "totalCustomers": count,
            "totalMaleCustomers": gender_counts["male"],
            "totalFemaleCustomers": gender_counts["female"],
            "avgDwellTime": avg_dwell_time,
            "customersByAge": age_groups,
        })

    def local_save_heatmap(self, frame, heatmap_type):

        # Convert OpenCV frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (PIL)
        pil_image = Image.fromarray(frame_rgb)

        # Add metadata to the image (PNG format for text metadata)
        metadata = PngImagePlugin.PngInfo()
        valid_saving_start_time = self.video_manager.get_analysis_start_time().strftime("%Y-%m-%d_%H-%M-%S")
        valid_saving_end_time = self.video_manager.get_current_time().strftime("%Y-%m-%d_%H-%M-%S")

        metadata.add_text("Start Time", valid_saving_start_time)
        metadata.add_text("End Time", valid_saving_end_time)

        # Save the frame with metadata
        image_path = (f"{self.base_dir}/heatmaps_snapshots/" + valid_saving_start_time + "_" + valid_saving_end_time +
                      heatmap_type + "_heatmap.png")  # Customize the path as needed
        pil_image.save(image_path, "PNG", pnginfo=metadata)
        return image_path, heatmap_type + valid_saving_start_time + "_" + valid_saving_end_time


    def provide_heatmap(self, annotated_heatmap, clean_heatmap):

        # Save annotated frame
        self.local_save_heatmap(annotated_heatmap, HeatmapType.ANNOTATED.value)
        image_path, public_image_id = self.local_save_heatmap(clean_heatmap, HeatmapType.CLEAN.value)
        url = self.cloudinary_service.upload_heatmap(image_path, public_image_id)
        print("****2. Upload an image****\nDelivery URL: ", url, "\n")

    def provide_metrics(self, url):

        # Export outputs locally for internal debugging of the application.
        export_to_local_csv(self.reports, self.base_dir)
        export_to_local_txt(self.reports, self.base_dir)

        # Data in the format that the server expect
        data = {"reports": self.reports, "jobId": self.video_manager.get_job_id()}
        response = requests.post(url, json=data)

        # Handle response
        if response.status_code == 200:
            LOGGER.info("Successfully saved data")
        else:
            LOGGER.info("Failed to save data")

    def local_save(self, count, ages, dwell_times, genders, past_customers):
        self.local_save_metrics(count, ages, dwell_times, genders, past_customers)

    def provide(self, annotated_heatmap, clean_heatmap):
        self.provide_metrics(SERVER_URL + REPORT_ENDPOINT)
        self.provide_heatmap(annotated_heatmap, clean_heatmap)

    def download_video(self, url):
        try:

            # Calling cloudinary service to download the video.
            return self.cloudinary_service.download_video(url, self.base_dir)

        # Raise exception if there was some issue with video download by the service.
        except Exception as e:
            print(e)
            raise
