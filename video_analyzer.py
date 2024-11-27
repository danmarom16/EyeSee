import time

from ultralytics import YOLO
import cv2
from ultralytics.solutions import Heatmap
import json
import numpy as np
from ultralytics.utils.plotting import Annotator
import datetime
from ultralytics import YOLO
from enum import Enum

class ExitType(Enum):
    CLEAN = 1 # Exit store through the entrance door.
    DIRTY = 2 # Exit the frame before exiting the store

REEVALUATION_INTERVAL = 5
LOW_CONF = 0.75
SAVING_INTERVAL = 10

class VideoAnalyzer(Heatmap):

    def __init__(self, video_collector, age_classifier, gender_classifier, data_provider, **kwargs):
        super().__init__(**kwargs)

        self.video_collector = video_collector
        # Initialize dwell time tracking
        # {track_id: {"entrance": timestamp, "exit": timestamp, "dwell": duration, "exit_type":type, is_exit:true/false}}
        self.dwell_times = {}
        self.colormap = cv2.COLORMAP_PARULA if self.CFG["colormap"] is None else self.CFG["colormap"]
        self.prev_track_ids = {}
        # init video writer
        self.video_writer = cv2.VideoWriter(
            "./logs/outputs/output2.mp4",
            cv2.VideoWriter.fourcc(*"mp4v"),
            float(self.video_collector.get_fps()),
            (int(self.video_collector.get_width()), int(self.video_collector.get_height()))
        )
        self.ages = {}
        self.genders = {}

        self.age_classifier = YOLO(age_classifier)
        self.gender_classifier = YOLO(gender_classifier)
        self.i = 0
        self.data_provider = data_provider
        self.CFG["classes"] = [0] #Detect only persons
        self.start_time = None

        self.past_customers = []

    # Override
    def count_objects(self, track_line, box, track_id, prev_position, cls):
        if prev_position is None or track_id in self.counted_ids:
            return

        centroid = self.r_s.centroid
        dx = (box[0] - prev_position[0]) * (centroid.x - prev_position[0])
        dy = (box[1] - prev_position[1]) * (centroid.y - prev_position[1])

        if len(self.region) >= 3 and self.r_s.contains(self.Point(track_line[-1])):
            self.counted_ids.append(track_id)
            # For polygon region
            if dx > 0:
                self.in_count += 1
                self.classwise_counts[self.names[cls]]["IN"] += 1
            else:
                self.out_count += 1
                self.classwise_counts[self.names[cls]]["OUT"] += 1
                if track_id in self.dwell_times:
                    self.dwell_times[track_id]["exit"] = self.video_collector.get_current_time()
                    self.dwell_times[track_id]["dwell"] = self.dwell_times[track_id]["exit"] - \
                                                          self.dwell_times[track_id]["entrance"]
                    self.dwell_times[track_id]["exit_type"] = ExitType.CLEAN
                    self.dwell_times[track_id]["is_exit"] = True
                #TODO:add to past customers.

        elif len(self.region) < 3 and self.LineString([prev_position, box[:2]]).intersects(self.r_s):
            self.counted_ids.append(track_id)
            # For linear region
            if dx > 0 and dy > 0:
                self.in_count += 1
                self.classwise_counts[self.names[cls]]["IN"] += 1
            else:
                self.out_count += 1
                self.classwise_counts[self.names[cls]]["OUT"] += 1
                if track_id in self.dwell_times:
                    self.dwell_times[track_id]["exit"] = self.video_collector.get_current_time()
                    self.dwell_times[track_id]["dwell"] = self.dwell_times[track_id]["exit"] - \
                                                          self.dwell_times[track_id]["entrance"]
                    self.dwell_times[track_id]["exit_type"] = ExitType.CLEAN
                    self.dwell_times[track_id]["is_exit"] = True


    def remove_lost_ids(self):
        # Identify IDs in dwell_times that are no longer in track_ids
        lost_ids = [track_id for track_id in self.prev_track_ids if track_id not in self.track_ids]

        #TODO:take fare of the case when the client does a clean exit - make sure that is_exit will be true.
        for track_id in lost_ids:
            # The client exited the current camera frame.
            if track_id in self.dwell_times and self.dwell_times[track_id]["is_exit"] is False:
                self.dwell_times[track_id]["exit"] = self.video_collector.get_current_time()
                self.dwell_times[track_id]["dwell"] = self.dwell_times[track_id]["exit"] - self.dwell_times[track_id][
                    "entrance"]
                self.dwell_times[track_id]["exit_type"] = ExitType.DIRTY
                self.dwell_times[track_id]["is_exit"] = True

            # Add customers to past customer as it left the frame.
            self.past_customers.append({
                "track_id": track_id,
                "dwell": self.dwell_times[track_id],
                "age": self.ages[track_id],
                "gender": self.genders[track_id],
            })

            #TODO:make sure to pop clean exit id from ages with the counting object method.
            # Safely remove the track_id from all related data structures
            self.dwell_times.pop(track_id, None)
            self.ages.pop(track_id)
            self.genders.pop(track_id)

        # Optional: Log the removed IDs for debugging
        if lost_ids:
            print(f"Removed lost track_ids: {lost_ids}")

    def box_to_image(self, image, bounding_box):
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

    def add_gender(self, original_frame, box, track_id):
        gender_result = self.gender_classifier(self.box_to_image(original_frame, box))
        gender_group = gender_result[0].names[gender_result[0].probs.top1].lower()
        gender_group_conf = gender_result[0].probs.top1conf
        gender_result[0].save(f"./logs/classifications/gender/{gender_group}/{track_id}_iteration_{self.i}.jpg")
        self.genders[track_id] = (gender_group, gender_group_conf)

    def add_age(self, original_frame, box, track_id):
        age_result = self.age_classifier(self.box_to_image(original_frame, box))
        age_group = age_result[0].names[age_result[0].probs.top1]
        age_group_conf = age_result[0].probs.top1conf
        age_result[0].save(f"./logs/classifications/age/{age_group}/{track_id}_iteration_{self.i}.jpg")
        self.ages[track_id] = (age_group, age_group_conf)


    def is_first_iteration_of_interval(self):
        return self.i == 0

    def mark_end_of_interval(self):
        self.i = 0


    def save_and_reset(self, im0):
        self.data_provider.provide(self.in_count - self.out_count,
                                   self.ages, self.dwell_times, self.genders, im0, self.start_time, self.past_customers)
        self.mark_end_of_interval()
        self.past_customers = [] # erase all past customers as they already have been saved.


    def analyze_video_frame(self, im0):

        # Catch starting time of current interval;
        if self.is_first_iteration_of_interval():
          self.start_time = self.video_collector.get_current_time()

        original_frame = im0.copy() #For age and gender classification.

        if not self.initialized:
            self.heatmap = np.zeros_like(im0, dtype=np.float32)
        self.initialized = True

        self.annotator = Annotator(im0, line_width=self.line_width)

        # Run tracking
        self.extract_tracks(im0)

        # Remove IDs that are no longer in the frame
        self.remove_lost_ids()

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.heatmap_effect(box)
            self.i += 1

            if self.region is not None:
                self.annotator.draw_region(reg_pts=self.region, color=(104, 9, 123), thickness=self.line_width * 2)
                self.store_tracking_history(track_id, box)
                self.store_classwise_counts(cls)

                # Draw bounding boxes and labels for each detected object
                age = self.ages.get(track_id, "Not Detected")
                gender = self.genders.get(track_id, "Not Detected")

                label = f"{self.names[cls]} id:{track_id} age:{age} gender: {gender}"  # Label includes class name and track ID
                self.annotator.box_label(box, label, color=(255, 0, 0))  # Red bounding box for better visibility

                prev_position = None
                if len(self.track_history[track_id]) > 1:
                    prev_position = self.track_history[track_id][-2]
                self.count_objects(self.track_line, box, track_id, prev_position, cls)

            # New ID detected
            if track_id not in self.prev_track_ids:
                self.dwell_times[track_id] = {"entrance": None, "exit": None, "dwell": None, "is_exit": False}
                self.dwell_times[track_id]["entrance"] = self.video_collector.get_current_time()
                self.dwell_times[track_id]["is_exit"] = False
                self.add_gender(original_frame, box, track_id)
                self.add_age(original_frame, box, track_id)

            # Reevaluate previous prediction and fix it if needed.
            if self.i % REEVALUATION_INTERVAL == 0:
                if self.ages[track_id][1] < LOW_CONF:
                    self.add_age(original_frame, box, track_id)

                if self.genders[track_id][1] < LOW_CONF:
                    self.add_gender(original_frame, box, track_id)

        if self.region is not None:
            self.display_counts(im0)

        if self.track_data.id is not None:
            im0 = cv2.addWeighted(
                im0,
                0.5,
                cv2.applyColorMap(
                    cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), self.colormap
                ),
                0.5,
                0,
            )

        self.prev_track_ids = self.track_ids
        self.display_output(im0)
        self.video_collector.increment_current_time()   # Update local time of the video

        # if self.i % SAVING_INTERVAL == 0:
        #     self.save_and_reset(im0)

        return im0

    def close_dwell_times(self):
        for track_id in self.dwell_times:
            if not self.dwell_times[track_id]["is_exit"]:
                self.dwell_times[track_id]["is_exit"] = True
                self.dwell_times[track_id]["exit"] = self.video_collector.get_current_time()
                self.dwell_times[track_id]["dwell"] = self.dwell_times[track_id]["exit"] - self.dwell_times[track_id]["entrance"]

    def analyze_final_frame(self, frame):
        heatmap_image = self.analyze_video_frame(frame)
        self.close_dwell_times()
        self.data_provider.provide(self.in_count - self.out_count,
                           self.ages, self.dwell_times, self.genders, heatmap_image)