from collections import defaultdict
from util1 import ExitType, get_logger, EntranceType, LOW_CONF
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_SOL_DICT, LOGGER
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.solutions import ObjectCounter
from classifiers.age_classifier import AgeClassifier
from classifiers.gender_classifier import GenderClassifier
=

class ObjectTracker:

    def __init__(self, object_counter, video_manager, model):

        self.model = model

        # Custom Object for handling logic.
        self.video_manager = video_manager
        self.object_counter = object_counter

        # Classifiers
        self.age_classifier = AgeClassifier()
        self.gender_classifier = GenderClassifier()

        # Track related data structures
        self.track_history = defaultdict(list)
        self.tracks = None
        self.track_data = None
        self.track_ids = None
        self.clss = None
        self.boxes = None
        self.prev_track_ids = None
        self.dirty_ids = []
        self.counted_ids = []
        self.dwell_times = None
        self.present_clients = None
        self.past_customers = None
        self.past_customers_in_timeslice = None


    def extract_tracks(self, im0):

        self.tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"])

        # Extract tracks for OBB or object detection
        self.track_data = self.tracks[0].obb or self.tracks[0].boxes

        if self.track_data and self.track_data.id is not None:
            self.boxes = self.track_data.xyxy.cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
        else:
            LOGGER.warning("WARNING ⚠️ no tracks found!")
            self.boxes, self.clss, self.track_ids = [], [], []

    def remove_lost_ids(self, cls=0):
        lost_ids = [track_id for track_id in self.prev_track_ids if track_id not in self.track_ids]

        for track_id in lost_ids:

            # If this is a counter client, and he exited the current camera frame.
            if track_id not in self.dirty_ids:
                if self.present_clients[track_id]:

                    # Mark client exit as true and count him as "dirty out".
                    self.present_clients[track_id] = False
                    self.object_counter.count_client_dirty_exit(self.dwell_times, track_id)

                    # Append him to past customers and pop from all data structures.
                    self.add_to_past_customers(track_id)
                    self.pop_from_data_structures(track_id)
                    LOGGER.info(f"ID: {track_id} was counted and performed Dirty Exit")

                # Optional: Log the removed IDs for debugging
                if lost_ids:
                    LOGGER.info(f"Removed lost track_ids: {lost_ids}")

            # Else this is a client that was entered the video "dirty" and exiting "dirty".
            else:
                self.object_counter.count_not_client_dirty_exit(track_id, self.dirty_ids)

    def pop_from_data_structures(self, track_id):
        self.dwell_times.pop(track_id, None)
        self.age_classifier.data(track_id)
        self.gender_classifier.data(track_id)
        self.object_counter.counted_ids.remove(track_id)

    def store_tracking_history(self, track_id, box):
        self.track_line = self.track_history[track_id]
        self.track_line.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        if len(self.track_line) > 30:
            self.track_line.pop(0)

    def close_dwell_time(self, track_id):
        self.dwell_times[track_id]["exit"] = self.video_manager.get_current_time()
        self.dwell_times[track_id]["dwell"] = (self.dwell_times[track_id]["exit"] -
                                               self.dwell_times[track_id]["entrance"])
        self.dwell_times[track_id]["exit_type"] = ExitType.CLEAN

    def add_to_past_customers(self, track_id):
        # Add customers to past customer as it left the frame.
        self.past_customers_in_timeslice.append({
            "track_id": track_id,
            "dwell": self.dwell_times[track_id],
            "age": self.age_classifier.data[track_id],
            "gender": self.gender_classifier.data[track_id],
        })

        self.past_customers.append({
            "track_id": track_id,
            "dwell": self.dwell_times[track_id],
            "age": self.age_classifier.data[track_id],
            "gender": self.gender_classifier.data[track_id],
        })

    def is_object_has_history(self, track_id):
        return len(self.track_history[track_id]) > 1

    def get_prev_position(self, track_id):
        return self.track_history[track_id][-2]

    def is_customer_a_past_customer(self, track_id):
        return any(customer['track_id' == track_id] for customer in self.past_customers)

    def count_in(self, track_id, cls):

        # If this is a new ID, count it in
        if track_id not in self.counted_ids:
            LOGGER.info(f"ID: {track_id} Clean Enter")
            self.present_clients[track_id] = True
            self.object_counter.count_in(track_id, cls, self.dwell_times,
                                         self.present_clients, self.dirty_ids, self.counted_ids)

        # If the ID performed a "dirty enter", its no longer dirty.
        if track_id is self.dirty_ids:
            self.dirty_ids.remove(track_id)
            self.object_counter.decrement_count("DIRTY_IN")
            LOGGER.info(f"ID: {track_id} Removed from dirty list")

        # Dwell time initialization for a new client
        self.dwell_times[track_id] = {"entrance": None, "exit": None, "dwell": None, "entrance_type": None}
        self.dwell_times[track_id]["entrance_type"] = EntranceType.CLEAN

    def count_out(self, track_id, cls):

        # Id track_id is a present client and not dirty:
        if track_id in self.counted_ids:
            if track_id not in self.dirty_ids:

                LOGGER.into(f"ID: {track_id} clean exit")
                self.present_clients[track_id] = False
                self.object_counter.count_out(track_id, cls)

                # Close dwell time.
                if track_id in self.dwell_times:
                    self.close_dwell_time(track_id)
                    self.add_to_past_customers(track_id)
                    self.pop_from_data_structures(track_id)


    def classify(self, im0, track_id, box):
        self.age_classifier.classify(im0, track_id, box)
        self.gender_classifier.classify(im0, track_id, box)

    def reevaluate_classification(self, im0, track_id, box):
        if self.age_classifier.data[track_id][1] < LOW_CONF:
            self.age_classifier.classify(im0, track_id, box)
        if self.gender_classifier.data[track_id][1] < LOW_CONF:
            self.gender_classifier.classify(im0, track_id, box)

    def display_counts(self, im0, annotator):
        labels_dict = self.object_counter.display_counts(im0)
        if labels_dict:
            annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def set_prev_track_ids(self, current_ids):
        self.prev_track_ids = current_ids.copy()

