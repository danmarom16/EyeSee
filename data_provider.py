import csv
import os
import base64
import requests
from PIL import Image, PngImagePlugin
import cv2
import numpy as np
from datetime import datetime

SERVER_URL = 'http://127.0.0.1:4000'
HEATMAP_ENDPOINT = '/heatmap'
REPORT_ENDPOINT = '/report/create'
HEATMAP_ADD = '/heatmap/add'

class DataProvider:
    def __init__(self, cloudinary_service, video_collector,file_path="./logs/files/aggregated_data.csv"):
        self.video_collector = video_collector
        self.file_path = file_path
        self.cloudinary_service = cloudinary_service
        # Create the CSV file and write the header if it doesn't exist
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["date", "timeSlice", "totalCustomers", "totalMaleCustomers", "totalFemaleCustomers",
                     "avgDwellTime", "customersByAge"])
        self.reports = []
        self.heatmap_snapshots = []
        self.start_time = None


    def local_save_metrics(self, count, ages, dwell_times, genders, start_time, past_customers, end_time):

        young = 0
        children = 0
        adult = 0
        elder = 0
        male = 0
        female = 0
        total_dwell = 0

        for customer in past_customers:
            age = customer["age"][0]
            gender = customer["gender"][0]
            total_dwell += customer["dwell"]["dwell"].total_seconds()

            if gender == "male":
                male += 1
                if age == "adult":
                    adult += 1
                elif age == "elder":
                    elder += 1
                elif age == "young":
                    young += 1
                else:
                    children += 1

            if gender == "female":
                female += 1
                if age == "adult":
                    adult += 1
                elif age == "elder":
                    elder += 1
                elif age == "young":
                    young += 1
                else:
                    children += 1

        for track_id in ages:
            if ages[track_id][0] == "young":
                young += 1
            elif ages[track_id][0] == "children":
                children += 1
            elif ages[track_id][0] == "adult":
                adult += 1
            else:
                elder += 1

        for track_id in genders:
            if genders[track_id][0] == "male":
                male += 1
            else:
                female += 1

        # All customers that are present in dwell times are currently in store.
        # Therefor there dwell time is ongoing and will be calculated with the current time - entrance time for each.
        for track_id in dwell_times:
            total_dwell += (end_time - dwell_times[track_id]["entrance"]).total_seconds()

        if count == 0 or total_dwell == 0:
            avg_dwell_time = 0
        else:
            avg_dwell_time = total_dwell / count

        self.reports.append({
            "date": start_time.strftime("%Y-%m-%d"),
            "timeSlice": start_time.strftime("%H-%M-%S") + "-" + end_time.strftime("%H-%M-%S"),
            "totalCustomers": count,
            "totalMaleCustomers": male,
            "totalFemaleCustomers": female,
            "avgDwellTime": avg_dwell_time,
            "customersByAge": {
                "young": young,
                "children": children,
                "adult": adult,
                "elder": elder,
            }
        })

    def local_save(self, count, ages, dwell_times, genders, start_time, past_customers):
        end_time = self.video_collector.get_current_time()
        self.local_save_metrics(count, ages, dwell_times, genders, start_time, past_customers, end_time)

    #TODO:Test this call.
    def provide_metrics(self, url):
        #For debugging
        self.export_to_local_csv()
        self.export_to_local_txt()
        data = {"reports":self.reports, "jobId":self.video_collector.get_job_id()}
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Successfully saved data")
        else:
            print("Failed to save data")

    def local_save_heatmap(self, frame):
        # Convert OpenCV frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (PIL)
        pil_image = Image.fromarray(frame_rgb)

        # Add metadata to the image (PNG format for text metadata)
        metadata = PngImagePlugin.PngInfo()
        valid_saving_start_time = self.video_collector.get_start_time().strftime("%Y-%m-%d_%H-%M-%S")
        valid_saving_end_time = self.video_collector.get_end_time().strftime("%Y-%m-%d_%H-%M-%S")

        metadata.add_text("Start Time", valid_saving_start_time)
        metadata.add_text("End Time", valid_saving_end_time)

        # Save the frame with metadata
        image_path = "./logs/heatmaps_snapshots/" + valid_saving_start_time + "_" + valid_saving_end_time + "_heatmap.png"  # Customize the path as needed
        pil_image.save(image_path, "PNG", pnginfo=metadata)
        return image_path, valid_saving_start_time + "_" + valid_saving_end_time

    def provide_heatmap(self, url, frame, end_time):
        image_path, public_image_id = self.local_save_heatmap(frame)
        url = self.cloudinary_service.upload_heatmap(image_path, public_image_id)
        print("****2. Upload an image****\nDelivery URL: ", url, "\n")

    def export_to_local_csv(self):
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for report in self.reports:
                writer.writerow([
                    report["date"],
                    report["timeSlice"],
                    report["totalCustomers"],
                    report["totalMaleCustomers"],
                    report["totalFemaleCustomers"],
                    report["avgDwellTime"],
                    report["customersByAge"]
                ])

    def export_to_local_txt(self):
        with open(self.file_path + ".txt", mode='a', encoding='utf-8') as file:
            for report in self.reports:
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

    def provide(self, last_frame):
        end_time = self.video_collector.get_current_time()
        self.provide_metrics(SERVER_URL + REPORT_ENDPOINT)
        self.provide_heatmap(SERVER_URL + HEATMAP_ENDPOINT, last_frame, end_time)

    def set_start_time(self, start_time):
        self.start_time = start_time

    def download_video(self, url):
        try:
            return self.cloudinary_service.download_video(url)
        except Exception as e:
            print(e)
            raise

    def populate_video_data(self, video, data):
        self.video_collector.populate_video_data(video, data)
