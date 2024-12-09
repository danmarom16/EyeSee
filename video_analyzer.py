from ultralytics.solutions import Heatmap
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
from util import *
import logging

#TODO:Debug changes.

logging.basicConfig(
    filename='id_log.txt',  # Log file name
    level=logging.INFO,  # Log level
    format='%(message)s'  # Log format
)

class VideoAnalyzer(Heatmap):
    def __init__(self, video_collector, age_classifier, gender_classifier, data_provider, **kwargs):
        super().__init__(**kwargs)

        # Objects
        self.video_collector = video_collector
        self.data_provider = data_provider

        # Overridden Fields
        self.colormap = cv2.COLORMAP_PARULA if self.CFG["colormap"] is None else self.CFG["colormap"]
        self.video_writer = init_writer("./logs/outputs/output2.mp4", video_collector)
        self.CFG["classes"] = [0]

        # Custom data structures for customize business logic.
        self.prev_track_ids = dict()
        self.dwell_times = dict()
        self.ages = dict()
        self.genders = dict()
        self.is_client_in_store = dict()
        self.past_customers_in_timeslice = []

        # Classifiers
        self.age_classifier = YOLO(age_classifier)
        self.gender_classifier = YOLO(gender_classifier)

        # Custom fields
        self.i = 0
        self.start_time = None
        self.dirty_in_count = 0
        self.dirty_out_count = 0
        self.dirty_in_ids = []
        self.past_customers = []

    def classify(self, track_id, original_frame, box):
        self.dwell_times[track_id]["entrance"] = self.video_collector.get_current_time()
        self.add_gender(original_frame, box, track_id)
        self.add_age(original_frame, box, track_id)

    #Override
    def store_classwise_counts(self, cls):
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"CLEAN_IN": 0, "DIRTY_IN": 0, "CLEAN_OUT": 0, "DIRTY_OUT": 0}

    def count_in_and_analyze(self, track_id, cls, box, original_frame):
        if track_id not in self.counted_ids:
            logging.info(f"ID: {track_id} Clean Enter")

            self.counted_ids.append(track_id)
            self.is_client_in_store[track_id] = True

            # Counting part
            self.in_count += 1
            if self.names[cls] not in self.classwise_counts:
                self.classwise_counts[self.names[cls]] = {"CLEAN_IN": 0, "DIRTY_IN": 0, "CLEAN_OUT": 0, "DIRTY_OUT": 0}
            self.classwise_counts[self.names[cls]]["CLEAN_IN"] += 1

            # No longer dirty.
            if track_id in self.dirty_in_ids:
                self.dirty_in_ids.remove(track_id)
                self.classwise_counts[self.names[cls]]["DIRTY_IN"] -= 1
                logging.info(f"ID: {track_id} Removed from dirty list")

            self.dwell_times[track_id] = {"entrance": None, "exit": None, "dwell": None}
            self.dwell_times[track_id]["entrance_type"] = EntranceType.CLEAN

            self.store_classwise_counts(cls)

            self.classify(track_id, original_frame, box)


    def count_out(self, track_id, cls):

        if track_id in self.counted_ids:
            if track_id not in self.dirty_in_ids:
                # Count out.
                logging.info(f"ID: {track_id} Clean Exit")
                self.out_count += 1
                self.is_client_in_store[track_id] = False
                self.classwise_counts[self.names[cls]]["CLEAN_OUT"] += 1

                # Close dwell time.
                if track_id in self.dwell_times:
                    self.dwell_times[track_id]["exit"] = self.video_collector.get_current_time()
                    self.dwell_times[track_id]["dwell"] = (self.dwell_times[track_id]["exit"] -
                                                           self.dwell_times[track_id]["entrance"])
                    self.dwell_times[track_id]["exit_type"] = ExitType.CLEAN

                # Add customers to past customer as it left the frame.
                self.past_customers_in_timeslice.append({
                    "track_id": track_id,
                    "dwell": self.dwell_times[track_id],
                    "age": self.ages[track_id],
                    "gender": self.genders[track_id],
                })

                self.past_customers.append({
                    "track_id": track_id,
                    "dwell": self.dwell_times[track_id],
                    "age": self.ages[track_id],
                    "gender": self.genders[track_id],
                })

                # Pop customers from all relevant data structures.
                self.dwell_times.pop(track_id, None)
                self.ages.pop(track_id)
                self.genders.pop(track_id)
                self.counted_ids.remove(track_id)


    #Override
    def display_counts(self, im0):
        labels_dict = {
            str.capitalize(key): f"{'CLEAN_IN ' + str(value['CLEAN_IN']) if self.show_in else ''} "
            f"{' DIRTY_IN ' + str(value['DIRTY_IN']) if self.show_out else ''}"
            f"{' CLEAN_OUT ' + str(value['CLEAN_OUT']) if self.show_out else ''}"
            f"{' DIRTY_OUT ' + str(value['DIRTY_OUT']) if self.show_out else ''}"
            .strip()
            for key, value in self.classwise_counts.items()
            if value['CLEAN_IN'] != 0 or value['DIRTY_IN'] != 0 or value['CLEAN_OUT'] != 0 or value['DIRTY_OUT'] != 0
        }
        if labels_dict:
            self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)


    def analyze_object(self, box, track_id, prev_position, cls, original_frame):
        if prev_position is None:
            return

        dy = ((box[1] + box[3]) / 2) - prev_position[1]

        if len(self.region) < 3 and self.LineString([prev_position, box[:2]]).intersects(self.r_s):
            # If dy < 0, it means that prev_y > current_y, and therefore client moving to the top of the screen.
            if dy < 0:
                self.count_in_and_analyze(track_id, cls, box, original_frame)
            else:
                self.count_out(track_id, cls)
        else:
            if track_id not in self.counted_ids and track_id not in self.dirty_in_ids:
                logging.info(f"ID: {track_id} Dirty Entrance")
                self.classwise_counts[self.names[cls]]["DIRTY_IN"] += 1
                self.dirty_in_ids.append(track_id)

    def remove_lost_ids(self, cls=0):
        lost_ids = [track_id for track_id in self.prev_track_ids if track_id not in self.track_ids]

        for track_id in lost_ids:
            if track_id not in self.dirty_in_ids:
                # The client exited the current camera frame.
                if self.is_client_in_store[track_id] == True:
                    self.is_client_in_store[track_id] = False
                    self.dwell_times[track_id]["exit"] = self.video_collector.get_current_time()
                    self.dwell_times[track_id]["dwell"] = self.dwell_times[track_id]["exit"] - self.dwell_times[track_id][
                        "entrance"]
                    self.dwell_times[track_id]["exit_type"] = ExitType.DIRTY
                    self.dirty_out_count += 1
                    self.in_count -= 1
                    self.classwise_counts[self.names[cls]]["DIRTY_OUT"] += 1


                # Add customers to past customer as it left the frame.
                    self.past_customers_in_timeslice.append({
                        "track_id": track_id,
                        "dwell": self.dwell_times[track_id],
                        "age": self.ages[track_id],
                        "gender": self.genders[track_id],
                    })

                    self.past_customers.append({
                        "track_id": track_id,
                        "dwell": self.dwell_times[track_id],
                        "age": self.ages[track_id],
                        "gender": self.genders[track_id],
                    })

                    self.dwell_times.pop(track_id, None)
                    self.ages.pop(track_id)
                    self.genders.pop(track_id)
                    self.counted_ids.remove(track_id)
                    logging.info(f"ID: {track_id} was counted and performed Dirty Exit")

                # Optional: Log the removed IDs for debugging
                if lost_ids:
                    print(f"Removed lost track_ids: {lost_ids}")

        # Dirty ID that is lost
            else:
                self.dirty_in_ids.remove(track_id)
                self.classwise_counts[self.names[cls]]["DIRTY_IN"] -= 1
                logging.info(f"ID: {track_id} was NOT counted and performed Dirty Exit")

    def add_gender(self, original_frame, box, track_id):
        gender_result = self.gender_classifier(box_to_image(original_frame, box))
        gender_group = gender_result[0].names[gender_result[0].probs.top1].lower()
        gender_group_conf = gender_result[0].probs.top1conf
        gender_result[0].save(f"./logs/classifications/gender/{gender_group}/{track_id}_iteration_{self.i}.jpg")
        self.genders[track_id] = (gender_group, gender_group_conf)

    def add_age(self, original_frame, box, track_id):
        age_result = self.age_classifier(box_to_image(original_frame, box))
        age_group = age_result[0].names[age_result[0].probs.top1]
        age_group_conf = age_result[0].probs.top1conf
        age_result[0].save(f"./logs/classifications/age/{age_group}/{track_id}_iteration_{self.i}.jpg")
        self.ages[track_id] = (age_group, age_group_conf)

    def save_and_reset(self):
        self.data_provider.local_save(self.in_count - self.out_count,
                                      self.ages, self.dwell_times, self.genders, self.start_time, self.past_customers_in_timeslice)
        self.i = 0
        self.past_customers_in_timeslice = []  # erase all past customers as they already have been saved.

    def analyze_video_frame(self, im0):
        # Create a copy of the original frame for classification tasks
        original_frame = im0.copy()
        if self.i == 0:
            self.start_time = self.video_collector.get_current_time()

        self.i += 1
        # Initialize heatmap if not already done
        if not self.initialized:
            self.heatmap = np.zeros_like(im0, dtype=np.float32)
            self.initialized = True

        # Initialize the annotator for the frame
        self.annotator = Annotator(im0, line_width=self.line_width)

        # Run object tracking
        self.extract_tracks(im0)

        # Remove IDs that are no longer in the frame
        self.remove_lost_ids()

        # Process each detected object
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # Apply heatmap effect to the bounding box
            self.heatmap_effect(box)
            self.store_tracking_history(track_id, box)

            # Draw the defined region if specified
            if self.region is not None:
                self.annotator.draw_region(reg_pts=self.region, color=(104, 9, 123), thickness=self.line_width * 2)

            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
                if not any(customer['track_id'] == track_id for customer in self.past_customers):
                    self.analyze_object(box, track_id, prev_position, cls, original_frame)
            annotate_object(track_id, cls, box, self.ages.get(track_id, "Not Detected"),
                            self.genders.get(track_id, "Not Detected"), self.annotator, (255, 0, 0))

            if track_id in self.counted_ids:
                annotate_object(track_id, cls, box, self.ages.get(track_id, "Not Detected"),
                                self.genders.get(track_id, "Not Detected"), self.annotator, (0, 0, 255))

                # Reevaluation for low-confidence classifications at specified intervals
                if self.i % REEVALUATION_INTERVAL == 0:
                    if self.ages[track_id][1] < LOW_CONF:
                        self.add_age(original_frame, box, track_id)

                    if self.genders[track_id][1] < LOW_CONF:
                        self.add_gender(original_frame, box, track_id)

        # Display counts on the frame if a region is defined
        if self.region is not None:
            self.display_counts(im0)

        # Overlay the heatmap on the frame if track data exists
        if self.track_data.id is not None:
            im0 = normalize_heatmap(im0, self.heatmap, self.colormap)

        # Update previous track IDs and display the output
        self.prev_track_ids = self.counted_ids.copy()
        self.display_output(im0)

        # Increment the current time of the video collector
        self.video_collector.increment_current_time()

        # Save results and reset counters at the specified saving interval
        if self.i % SAVING_INTERVAL == 0:
            self.save_and_reset()

        # Return the processed frame
        return im0


    def initialize(self, im0):
        # Catch starting time of current interval;
        self.start_time = self.video_collector.get_current_time()
        self.i += 1
        self.data_provider.set_start_time(self.video_collector.get_current_time()) #For heatmap image

        original_frame = im0.copy()  # For age and gender classification.

        if not self.initialized:
            self.heatmap = np.zeros_like(im0, dtype=np.float32)
        self.initialized = True

        self.annotator = Annotator(im0, line_width=self.line_width)

        # Run tracking
        self.extract_tracks(im0)

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.heatmap_effect(box)

            if self.region is not None:
                self.annotator.draw_region(reg_pts=self.region, color=(104, 9, 123), thickness=self.line_width * 2)
            self.count_in_and_analyze(track_id, cls, box, original_frame)


        if self.region is not None:
            self.display_counts(im0)

        if self.track_data.id is not None:
            im0 = normalize_heatmap(im0, self.heatmap, self.colormap)

        self.prev_track_ids = self.counted_ids.copy()
        self.display_output(im0)
        self.video_collector.increment_current_time()  # Update local time of the video

        return im0


    def analyze_final_frame(self, frame):
        self.data_provider.provide(frame)
