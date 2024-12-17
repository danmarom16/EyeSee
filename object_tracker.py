from collections import defaultdict
from util1 import ExitType, EntranceType, PastCustomer, DwellTime, CountType, ClassifierType, LOW_CONF
from ultralytics.utils import LOGGER
from classifiers.age_classifier import AgeClassifier
from classifiers.gender_classifier import GenderClassifier


class ObjectTracker:

    def __init__(self, object_counter, video_manager, model, CFG):

        # YOLO Model that will be used in order to track customers.
        self.model = model

        # Configuration structure.
        self.CFG = CFG

        # Video Manager - will be responsible to handle all metadata regarding the video.
        self.video_manager = video_manager

        # Object Counter - its responsibility is to handle the counting logic in the program.
        self.object_counter = object_counter

        # Classifiers for demographic analysis.
        self.age_classifier = AgeClassifier()
        self.gender_classifier = GenderClassifier()

        # Track the history of each detected ID (up to 30 frames).
        self.track_history = defaultdict(list)

        # Holds the raw tracking info fo each detection.
        self.tracks = None

        # Holds the boxes of each track object.
        self.track_data = None

        # Holds the unique ids detected in the current frame.
        self.track_ids = None

        # Holds the classes of each detection.
        self.clss = None

        # Holds the boxes for each detection.
        self.boxes = None

        # ID Structures that will help with recognizing new customers.
        # Clients from previous frame.
        self.prev_track_ids = None

        # Dirty IDs - Clients who entered the frame not through the entrance line.
        self.dirty_ids = []

        # Counted IDs - representing the real clients in the store who entered cleanly through the entrance.
        self.counted_ids = []

        # Data structure that assign each client all the information needed to count his dwell time in the store.
        self.dwell_times = defaultdict()

        # Boolean array that is used to distinguish between dirty clients and real clients.
        self.present_clients = defaultdict()

        # Past customers - global array that won't be reset until the termination of the program.
        self.past_customers = []

        # Past customers in current timeslice - will be reseated with each new timeslice.
        self.past_customers_in_timeslice = []


    def extract_tracks(self, im0):

        self.tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"])
        LOGGER.info("Extracting tracks")

        # Extract tracks for OBB or object detection
        self.track_data = self.tracks[0].obb or self.tracks[0].boxes

        if self.track_data and self.track_data.id is not None:
            self.boxes = self.track_data.xyxy.cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()

            # Still don't know if this person is dirty or not, but he is present in the frame. This will be determined
            # later on.
            for track_id in self.track_ids:
                if track_id not in self.present_clients:
                    self.present_clients[track_id] = False

        else:
            LOGGER.warning("WARNING ⚠️ no tracks found!")
            self.boxes, self.clss, self.track_ids = [], [], []

    def remove_lost_ids(self):

        # Extract lost Ids using comparison to previous ids.
        lost_ids = [track_id for track_id in self.prev_track_ids if track_id not in self.track_ids]

        for track_id in lost_ids:

            # If this is a counted client, and he exited the current camera frame.
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
                self.dirty_ids.remove(track_id)
                self.object_counter.count_dirty_and_dirty_exit()

    def pop_from_data_structures(self, track_id):
        self.dwell_times.pop(track_id, None)
        self.age_classifier.remove_id(track_id)
        self.gender_classifier.remove_id(track_id)
        self.counted_ids.remove(track_id)
        self.present_clients.pop(track_id, None)

    def store_tracking_history(self, track_id, box):
        self.track_line = self.track_history[track_id]
        self.track_line.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        if len(self.track_line) > 30:
            self.track_line.pop(0)

    def close_dwell_time(self, track_id):

        # Set exit to the current time.
        self.dwell_times[track_id][DwellTime.EXIT.value] = self.video_manager.get_current_time()

        # Calculate dwell time
        self.dwell_times[track_id][DwellTime.DWELL.value] = (self.dwell_times[track_id][DwellTime.EXIT.value] -
                                               self.dwell_times[track_id][DwellTime.ENTRANCE.value])

        # Mark as a clean exit, as only clean exit clients will be called with this function.
        self.dwell_times[track_id][DwellTime.EXIT_TYPE.value] = ExitType.CLEAN.value

    def add_to_past_customers(self, track_id):
        # Add customers to past customer as it left the frame.
        self.past_customers_in_timeslice.append({
            PastCustomer.TRACK_ID.value: track_id,
            PastCustomer.DWELL.value: self.dwell_times[track_id],
            PastCustomer.AGE.value: self.age_classifier.data[track_id],
            PastCustomer.GENDER.value: self.gender_classifier.data[track_id],
        })

        self.past_customers.append({
            PastCustomer.TRACK_ID.value: track_id,
            PastCustomer.DWELL.value: self.dwell_times[track_id],
            PastCustomer.AGE.value: self.age_classifier.data[track_id],
            PastCustomer.GENDER.value: self.gender_classifier.data[track_id],
        })

    def get_prev_position(self, track_id):
        return self.track_history[track_id][-2]

    def is_customer_a_past_customer(self, track_id):
        return any(customer['track_id'] == track_id for customer in self.past_customers)

    def is_object_has_history(self, track_id):
        return len(self.track_history[track_id]) > 1

    def add_new_client(self, frame, track_id, cls, box):

        # Add to counted IDs
        self.counted_ids.append(track_id)
        LOGGER.info(f"ID: {track_id} Clean Enter")

        # Mark it as client in the boolean array.
        self.present_clients[track_id] = True

        # Call object counter to count it in.
        self.object_counter.count_in(cls)

        # Run classification on a new client.
        self.classify(frame, track_id, box)

        # If the ID performed a "dirty enter", its no longer dirty.
        if track_id is self.dirty_ids:
            self.dirty_ids.remove(track_id)
            self.object_counter.decrement_count(CountType.DIRTY_IN)
            LOGGER.info(f"ID: {track_id} Removed from dirty list")

        # Dwell time initialization for a new client
        self.dwell_times[track_id] = {DwellTime.ENTRANCE.value: None, DwellTime.EXIT.value: None,
                                      DwellTime.DWELL.value: None, DwellTime.EXIT_TYPE.value: None}
        self.dwell_times[track_id][DwellTime.ENTRANCE_TYPE.value] = EntranceType.CLEAN.value
        self.dwell_times[track_id][DwellTime.ENTRANCE.value] = self.video_manager.get_current_time()

    def is_current_customer(self, track_id):
        return self.present_clients[track_id]

    def remove_client(self, track_id, cls):

        # Track_id is a present client and not dirty:
        if track_id in self.counted_ids:
            if track_id not in self.dirty_ids:
                LOGGER.info(f"ID: {track_id} clean exit")
                self.present_clients[track_id] = False
                self.object_counter.count_out(cls)

                # Close dwell time.
                if track_id in self.dwell_times:
                    self.close_dwell_time(track_id)
                    self.add_to_past_customers(track_id)
                    self.pop_from_data_structures(track_id)

    def count_dirty_id(self, track_id, cls):
        self.object_counter.count_dirty_entrance(cls)
        self.dirty_ids.append(track_id)

    def classify(self, im0, track_id, box):
        LOGGER.info("Classifying object")
        self.age_classifier.classify(im0, track_id, box, ClassifierType.AGE.value)
        self.gender_classifier.classify(im0, track_id, box, ClassifierType.GENDER.value)

    def reevaluate_classification(self, im0, track_id, box):
        age_conf = self.age_classifier.get_track_id_conf(track_id)
        gender_conf = self.gender_classifier.get_track_id_conf(track_id)

        if age_conf != "Not Detected" and age_conf < LOW_CONF:
            LOGGER.info(f"ID: {track_id} has low confidence age prediction of: {age_conf}."
                        f"Triggering classification again.")
            self.age_classifier.classify(im0, track_id, box, ClassifierType.AGE.value)
        if gender_conf != "Not Detected" and gender_conf < LOW_CONF:
            LOGGER.info(f"ID: {track_id} has low confidence gender prediction of: {gender_conf}."
                        f"Triggering classification again.")
            self.gender_classifier.classify(im0, track_id, box, ClassifierType.GENDER.value)

    def display_counts(self, im0, annotator):
        labels_dict = self.object_counter.display_counts()
        annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def set_prev_track_ids(self, current_ids):
        self.prev_track_ids = current_ids.copy()

    def get_tracks(self):
        return self.tracks

    def get_boxes(self):
        return self.boxes

    def get_track_ids(self):
        return self.track_ids

    def get_classes(self):
        return self.clss

    def get_track_id_classifier_data(self, track_id, classifier_type):
        if classifier_type == ClassifierType.AGE:
            return self.age_classifier.get_track_id_data(track_id)
        else:
            return self.gender_classifier.get_track_id_data(track_id)

    def save_prev_ids(self):
        self.prev_track_ids = self.counted_ids.copy()

    def calculate_current_count(self):
        return self.object_counter.calculate_current_count()