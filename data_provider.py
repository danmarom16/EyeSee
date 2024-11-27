import csv
import os
import cv2
from PIL import Image, PngImagePlugin
import cv2
import numpy as np
from datetime import datetime


class DataProvider:
    def __init__(self, video_collector, file_path="./logs/files/aggregated_data.csv"):
        self.file_path = file_path
        self.video_collector = video_collector
        # Create the CSV file and write the header if it doesn't exist
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["start_time", "end_time", "count", "young", "children", "adult", "elder", "Male", "Female", "avg_dwell_time" ])


    def provide_heatmap_image(self, frame, start_time):

        # Convert OpenCV frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (PIL)
        pil_image = Image.fromarray(frame_rgb)

        # Add metadata to the image (PNG format for text metadata)
        metadata = PngImagePlugin.PngInfo()
        valid_saving_start_time = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        valid_saving_end_time = self.video_collector.get_current_time().strftime("%Y-%m-%d_%H-%M-%S")

        metadata.add_text("Start Time", valid_saving_start_time)
        metadata.add_text("End Time", valid_saving_end_time)


        # Save the frame with metadata
        image_path = "./logs/heatmaps_snapshots/" + valid_saving_start_time + "_" + valid_saving_end_time+ "_heatmap.png"  # Customize the path as needed
        pil_image.save(image_path, "PNG", pnginfo=metadata)

    def provide_metrics(self, count, ages, dwell_times, genders, start_time, past_customers):
        # Write data to CSV
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            end_time = self.video_collector.get_current_time()

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

            for track_id in dwell_times:
                if dwell_times[track_id]["is_exit"] is True:
                    total_dwell += dwell_times[track_id]["dwell"].total_seconds()

            if count == 0 or total_dwell == 0:
                avg_dwell_time = 0
            else:
                avg_dwell_time = total_dwell / count

            writer.writerow([
                start_time,
                end_time,
                count,
                young,
                children,
                adult,
                elder,
                male,
                female,
                avg_dwell_time
            ])
    def provide(self, count, ages, dwell_times, genders, frame, start_time, past_customers):
        self.provide_heatmap_image(frame, start_time)
        self.provide_metrics(count, ages, dwell_times, genders, start_time, past_customers)

