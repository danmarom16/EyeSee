from ultralytics.solutions import Heatmap
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
from util import *

#TODO:Debug changes.
class VideoAnalyzer(Heatmap):
    """
    Extends the Heatmap class for advanced video frame analysis.

    The `VideoAnalyzer` class tracks objects across video frames, counts them entering or exiting the store,
    and classifies their demographic attributes such as age and gender. It also provides tools for generating
    heatmaps and saving annotated results.

    Attributes:
        video_collector (object): Manages video frames and timestamps.
        data_provider (object): Handles the provision of data to external systems.
        colormap (OpenCV colormap): Colormap used for heatmap visualization.
        video_writer (cv2.VideoWriter): Writer object to save annotated video outputs.
        prev_track_ids (dict): Tracks previously detected object IDs.
        dwell_times (dict): Maintains dwell times for tracked objects.
        ages (dict): Stores age classifications for detected objects.
        genders (dict): Stores gender classifications for detected objects.
        past_customers (list): Stores data of past customers who exited the store.
        age_classifier (YOLO): YOLO model used for age classification.
        gender_classifier (YOLO): YOLO model used for gender classification.
        i (int): Frame index counter.
        start_time (datetime): Timestamp for the start of the current analysis interval.

    Methods:
        classify: Performs detailed analysis for an object.
        count_in_and_analyze: Handles object entry and detailed analysis.
        count_out: Handles object exit and data cleanup.
        analyze_object: Determines if an object should be counted or ignored based on its position.
        remove_lost_ids: Cleans up IDs that are no longer present in the frame.
        add_gender: Classifies the gender of a detected object.
        add_age: Classifies the age group of a detected object.
        analyze_video_frame: Processes a single video frame for tracking, analysis, and visualization.
        initialize: analyze the first frame of the video, initializing all the present people and classify them.
    """
    def __init__(self, video_collector, age_classifier, gender_classifier, data_provider, **kwargs):
        """
        Initializes the VideoAnalyzer class with essential components and configurations.

        This method sets up the necessary objects, data structures, and classifiers for the
        `VideoAnalyzer` to perform video analysis, including tracking, heatmap generation,
        and demographic classification.

        Args:
            video_collector (object): An object responsible for managing video frames and timestamps.
            age_classifier (str): Path to the YOLO model for age classification.
            gender_classifier (str): Path to the YOLO model for gender classification.
            data_provider (object): An object that handles providing processed data to external systems.
            **kwargs: Additional arguments for the parent `Heatmap` class.

        Attributes:
            video_collector (object): Stores the video collector for frame and timestamp management.
            data_provider (object): Stores the data provider for external data output.
            colormap (int): OpenCV colormap used for heatmap visualization.
            video_writer (cv2.VideoWriter): Writer object for saving annotated output video.
            CFG["classes"] (list): Configuration to detect only persons (class index 0).
            prev_track_ids (dict): Tracks IDs of previously detected objects.
            dwell_times (dict): Tracks dwell time and movement details for detected objects.
            ages (dict): Stores age classifications for tracked objects.
            genders (dict): Stores gender classifications for tracked objects.
            past_customers (list): Maintains data of customers who have exited the frame.
            age_classifier (YOLO): YOLO model for age classification.
            gender_classifier (YOLO): YOLO model for gender classification.
            i (int): Frame index counter for interval-based analysis and saving.
            start_time (datetime): Timestamp marking the start of the current analysis interval.

        Returns:
            None
        """
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
        self.past_customers = []

        # Classifiers
        self.age_classifier = YOLO(age_classifier)
        self.gender_classifier = YOLO(gender_classifier)

        # Custom fields
        self.i = 0
        self.start_time = None
        self.dirty_in_count = 0
        self.dirty_out_count = 0

    def classify(self, track_id, original_frame, box):
        """
        Initializes and performs classification for a tracked object.

        This method sets up the initial dwell time data structure for a new tracked object,
        recording its entrance timestamp and marking it as active (not exited). It then
        classifies the object's gender and age using the provided YOLO models.

        Args:
            track_id (int): Unique ID of the tracked object.
            original_frame (np.ndarray): The original video frame containing the object.
            box (tuple): Bounding box coordinates of the object in the format (x1, y1, x2, y2).

        Side Effects:
            - Updates `self.dwell_times` with the object's entrance time and status.
            - Updates `self.genders` and `self.ages` with classification results for the object.

        Returns:
            None
        """
        self.dwell_times[track_id]["entrance"] = self.video_collector.get_current_time()
        self.add_gender(original_frame, box, track_id)
        self.add_age(original_frame, box, track_id)

    #Override
    def store_classwise_counts(self, cls):
        """
        Initialize class-wise counts for a specific object class if not already present.

        Args:
            cls (int): Class index for classwise count updates.

        This method ensures that the 'classwise_counts' dictionary contains an entry for the specified class,
        initializing 'CLEAN_IN', 'DIRTY_IN', 'CLEAN_OUT' and 'DIRTY_OUT' counts to zero if the class is not already
        present.
        """
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"CLEAN_IN": 0, "DIRTY_IN": 0, "CLEAN_OUT": 0, "DIRTY_OUT": 0}

    def count_in_and_analyze(self, track_id, cls, box, original_frame):
        """
        Handles the logic for counting an object entering the store and performing demographic analysis.

        This method updates the count of objects entering the store, stores tracking history, and performs demographic
        analysis (age and gender classification). The method also annotates the object on the frame for visualization.

        Args:
            track_id (int): Unique ID of the tracked object.
            cls (int): Class index of the detected object.
            box (tuple): Bounding box coordinates of the object in the format (x1, y1, x2, y2).
            original_frame (np.ndarray): The original video frame containing the object.

        Side Effects:
            - Appends the `track_id` to `self.counted_ids` to mark the object as processed.
            - Updates `self.in_count` and `self.classwise_counts` with the new entry.
            - Updates `self.dwell_times` with the entrance type of the object.
            - Updates `self.ages` and `self.genders` with classification results.
            - Annotates the object on the frame with classification and tracking information.

        Returns:
            None
        """
        if track_id not in self.counted_ids:
            self.counted_ids.append(track_id)
            self.is_client_in_store[track_id] = True

            # Counting part
            self.in_count += 1
            if self.names[cls] not in self.classwise_counts:
                self.classwise_counts[self.names[cls]] = {"CLEAN_IN": 0, "DIRTY_IN": 0, "CLEAN_OUT": 0, "DIRTY_OUT": 0}
            self.classwise_counts[self.names[cls]]["CLEAN_IN"] += 1
            self.dwell_times[track_id] = {"entrance": None, "exit": None, "dwell": None}
            self.dwell_times[track_id]["entrance_type"] = EntranceType.CLEAN

            self.store_classwise_counts(cls)

            self.classify(track_id, original_frame, box)


    def count_out(self, track_id, cls):
        """
        Handles the logic for counting an object exiting the store and cleaning up associated data.

        This method updates the count of objects exiting the store, calculates the dwell time
        for the object, and stores its details in the list of past customers. It also removes the
        object's data from internal tracking structures.

        Args:
            track_id (int): Unique ID of the tracked object.
            cls (int): Class index of the detected object.

        Side Effects:
            - Increments the `self.out_count` to track the total number of objects that exited.
            - Updates `self.classwise_counts` to increment the exit count for the specific class.
            - Updates `self.dwell_times` with the object's exit time, dwell duration, and exit type.
            - Appends the object's details (dwell time, age, and gender) to `self.past_customers`.
            - Removes the object's data from `self.dwell_times`, `self.ages`, `self.genders`, and `self.counted_ids`.

        Returns:
            None
        """
        if track_id in self.counted_ids:
            # Count out.
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
        """
        Displays object counts on the input image or frame.

        Args:
            im0 (numpy.ndarray): The input image or frame to display counts on.
        """
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
        """
        Analyzes the movement and position of a tracked object to determine whether it
        should be counted as entering or exiting the region of interest.
        """
        if prev_position is None:
            return

        if len(self.region) < 3 and self.LineString([prev_position, box[:2]]).intersects(self.r_s):
            print()

        dx = ((box[0] + box[2]) / 2) - prev_position[0]
        dy = ((box[1] + box[3]) / 2) - prev_position[1]
        direction = "UP" if dy > 0 else "DOWN"

        print(f"Track ID: {track_id}, Prev_y: {prev_position[1]}, Current_y: {box[1]}, Moving {direction}")
        # For polygon region
        if len(self.region) >= 3 and self.r_s.contains(self.Point(self.track_line[-1])):
            if dy < 0:
                self.count_in_and_analyze(track_id, cls, box, original_frame)
            else:
                self.count_out(track_id, cls)
        else:
            self.dirty_in_count += 1

        # For linear region
        if len(self.region) < 3 and self.LineString([prev_position, box[:2]]).intersects(self.r_s):
            # If dy < 0, it means that prev_y > current_y, and therefore client moving to the top of the screen.
            if dy < 0:
                self.count_in_and_analyze(track_id, cls, box, original_frame)
            else:
                self.count_out(track_id, cls)
        else:
            self.dirty_in_count += 1

    def remove_lost_ids(self, cls=0):
        """
        Identifies and processes IDs of objects that have exited the store by exiting the frame.

        This method checks for tracked IDs that are no longer visible in the current frame,
        updates their dwell times, and moves their data to the `past_customers` list. It
        also marks objects that exited without completing the defined exit process as
        having a "dirty" exit and removes their data from active tracking structures.

        Side Effects:
            - Updates `self.dwell_times` with the exit time, dwell duration, and exit type
              for lost IDs.
            - Appends data of lost IDs (track ID, dwell time, age, gender) to `self.past_customers`.
            - Removes lost IDs from `self.dwell_times`, `self.ages`, and `self.genders`.
            - Optionally logs the IDs of removed objects for debugging.

        Returns:
            None

        Behavior:
            - A lost ID is defined as an object ID present in `self.prev_track_ids` but
              not in `self.track_ids`.
            - If the lost ID has not been marked as exited, it is
              updated with the current exit time and marked as having a "dirty" exit
              (exited the frame without using the defined exit point).
        """
        lost_ids = [track_id for track_id in self.prev_track_ids if track_id not in self.track_ids]

        for track_id in lost_ids:

            # The client exited the current camera frame.
            if self.is_client_in_store[track_id] == True:
                self.is_client_in_store[track_id] = False
                self.dwell_times[track_id]["exit"] = self.video_collector.get_current_time()
                self.dwell_times[track_id]["dwell"] = self.dwell_times[track_id]["exit"] - self.dwell_times[track_id][
                    "entrance"]
                self.dwell_times[track_id]["exit_type"] = ExitType.DIRTY
                self.dirty_out_count += 1
                self.classwise_counts[self.names[cls]]["DIRTY_OUT"] += 1


            # Add customers to past customer as it left the frame.
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

        # Optional: Log the removed IDs for debugging
        if lost_ids:
            print(f"Removed lost track_ids: {lost_ids}")

    def add_gender(self, original_frame, box, track_id):
        """
        Performs gender classification for a tracked object and stores the result.

        This method uses the YOLO gender classifier to analyze the bounding box region
        of the object in the given frame, determines the most probable gender, and
        saves the classification result along with its confidence score. The result
        is also stored in the `self.genders` dictionary.

        Args:
            original_frame (np.ndarray): The original video frame containing the object.
            box (tuple): Bounding box coordinates of the object in the format (x1, y1, x2, y2).
            track_id (int): Unique ID of the tracked object.

        Side Effects:
            - Updates `self.genders` with the classified gender and confidence score for the object.
            - Saves the classification result as an image in the logs directory.

        Returns:
            None
        """
        gender_result = self.gender_classifier(box_to_image(original_frame, box))
        gender_group = gender_result[0].names[gender_result[0].probs.top1].lower()
        gender_group_conf = gender_result[0].probs.top1conf
        gender_result[0].save(f"./logs/classifications/gender/{gender_group}/{track_id}_iteration_{self.i}.jpg")
        self.genders[track_id] = (gender_group, gender_group_conf)

    def add_age(self, original_frame, box, track_id):
        """
        Performs age classification for a tracked object and stores the result.

        This method uses the YOLO age classifier to analyze the bounding box region
        of the object in the given frame, determines the most probable age group, and
        saves the classification result along with its confidence score. The result
        is also stored in the `self.ages` dictionary.

        Args:
            original_frame (np.ndarray): The original video frame containing the object.
            box (tuple): Bounding box coordinates of the object in the format (x1, y1, x2, y2).
            track_id (int): Unique ID of the tracked object.

        Side Effects:
            - Updates `self.ages` with the classified age group and confidence score for the object.
            - Saves the classification result as an image in the logs directory.

        Returns:
            None
        """
        age_result = self.age_classifier(box_to_image(original_frame, box))
        age_group = age_result[0].names[age_result[0].probs.top1]
        age_group_conf = age_result[0].probs.top1conf
        age_result[0].save(f"./logs/classifications/age/{age_group}/{track_id}_iteration_{self.i}.jpg")
        self.ages[track_id] = (age_group, age_group_conf)

    def save_and_reset(self):
        """
        Saves the current analysis data and resets relevant counters and data structures.

        This method provides the current analysis data (counts, demographics, dwell times,
        and past customers) to the data provider for storage or further processing. It then
        resets the frame index counter and clears the list of past customers.

        Args:
            im0 (np.ndarray): The current video frame being processed.

        Side Effects:
            - Calls `self.data_provider.provide` to save the current state of analysis data.
            - Resets the frame index counter (`self.i`) to 0.
            - Clears the list of past customers (`self.past_customers`).

        Returns:
            None
        """
        self.data_provider.local_save(self.in_count - self.out_count,
                                   self.ages, self.dwell_times, self.genders, self.start_time, self.past_customers)
        self.i = 0
        self.past_customers = []  # erase all past customers as they already have been saved.
        self.out_count = 0        # refresh out to be 0. The in_count remains the same as they are still present.

    def analyze_video_frame(self, im0):
        """
        Processes a single video frame for object detection, tracking, and analysis.

        This method performs the following tasks:
        1. Tracks objects across frames using YOLO-based tracking.
        2. Updates and cleans up IDs of objects no longer visible in the frame.
        3. Applies heatmap effects to the frame based on object activity.
        4. Analyzes newly detected objects for demographic information (age and gender).
        5. Reevaluates previously analyzed objects for accuracy at specified intervals.
        6. Displays counts, annotations, and the processed heatmap on the frame.
        7. Saves intermediate results and resets counters at regular intervals.

        Args:
            im0 (np.ndarray): The current video frame to be processed.

        Returns:
            np.ndarray: The processed video frame with annotations and heatmap overlay.
        """
        # Create a copy of the original frame for classification tasks
        original_frame = im0.copy()
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
        """
        Initializes tracking, heatmap, and analysis for the first video frame.

        This method prepares the system to process a new video frame by:
        1. Setting up the starting timestamp for the current interval.
        2. Initializing the heatmap and annotator if not already done.
        3. Running object tracking and processing detected objects.
        4. Performing initial demographic classification (age and gender).
        5. Displaying region boundaries, counts, and annotations on the frame.
        6. Normalizing and applying the heatmap to the frame.

        Args:
            im0 (np.ndarray): The input video frame to initialize and process.

        Returns:
            np.ndarray: The initialized and annotated video frame with applied heatmap.

        Side Effects:
            - Updates `self.start_time` with the current time from the video collector.
            - Sets up the heatmap and annotator if they are not already initialized.
            - Updates `self.in_count`, `self.prev_track_ids`, and object-specific dictionaries (e.g., `self.ages`, `self.genders`).
            - Updates and increments the current time in the video collector.

        Behavior:
            - Each detected object is analyzed for demographic attributes (age and gender).
            - Region boundaries and counts are drawn if a region is defined.
            - A baseline count is established based on the number of initial detections.
        """
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
