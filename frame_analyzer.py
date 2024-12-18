from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator
from shapely.geometry import LineString
from util import annotate_object, RED, GREEN, PURPLE, REEVALUATION_INTERVAL, ClassifierType

class FrameAnalyzer:

    def __init__(self, object_tracker, CFG, heatmap_manager):
        self.object_tracker = object_tracker
        self.CFG = CFG

        self.region = self.CFG["region"]  # Store region data for other classes usage
        self.line_width = (
            self.CFG["line_width"] if self.CFG["line_width"] is not None else 2
        )  # Store line_width for usage

        self.LineString = LineString
        self.r_s = self.LineString(self.region)
        self.heatmap_manager = heatmap_manager

    def initialize(self, im0):
        LOGGER.info("Initializing frame analyzer")

        # Copy of the original image for later classification.
        original_frame = im0.copy()

        # Copy for a clean heatmap with no box annotations that will be uploaded eventually to the server.
        heatmap_copy = im0.copy()

        # Extract tracks with the use of object_tracker.
        self.object_tracker.extract_tracks(im0)

        self.annotator = Annotator(im0, line_width=self.line_width)

        # Go through each detection.
        for box, track_id, cls in zip(self.object_tracker.get_boxes(), self.object_tracker.get_track_ids(),
                                      self.object_tracker.get_classes()):

            # Apply heatmap effect on the current box.
            self.heatmap_manager.apply_heatmap_effect(box)

            # Store tracking history for each box.
            self.object_tracker.store_tracking_history(track_id, box)

            # Draw the defined region if specified
            if self.region is not None:
                self.annotator.draw_region(reg_pts=self.region, color=PURPLE, thickness=self.line_width * 2)

            # Skipping analysis as all objects detected in the first frame are counted as clients, therefore we need
            # to count and classify them as they are considered clients already, regardless if they crossed the entrance
            # line or not.
            self.object_tracker.add_new_client(original_frame, track_id, cls, box)

            color = RED if track_id not in self.object_tracker.counted_ids else GREEN

            annotate_object(track_id, cls, box,
                            self.object_tracker.get_track_id_classifier_data(track_id, ClassifierType.AGE),
                            self.object_tracker.get_track_id_classifier_data(track_id, ClassifierType.GENDER),
                            self.annotator, color)

        # Display counts on the frame if a region is defined
        if self.region is not None:
            self.object_tracker.display_counts(im0, self.annotator)

        if self.object_tracker.track_data.id is not None:
            heatmap_copy = self.heatmap_manager.normalize_heatmap(heatmap_copy)
            im0 = self.heatmap_manager.normalize_heatmap(im0)

        self.object_tracker.save_prev_ids()
        return im0, heatmap_copy

    def analyze(self, im0, current_timeslice_frame_count):

        # Copy of the original image for later classification.
        original_frame = im0.copy()

        # Copy for a clean heatmap with no box annotations that will be uploaded eventually to the server.
        heatmap_copy = im0.copy()

        # Extract tracks
        self.object_tracker.extract_tracks(im0)
        self.object_tracker.remove_lost_ids()

        self.annotator = Annotator(im0, line_width=self.line_width)

        for box, track_id, cls in zip(self.object_tracker.boxes, self.object_tracker.track_ids, self.object_tracker.clss):
            self.heatmap_manager.apply_heatmap_effect(box)
            self.object_tracker.store_tracking_history(track_id, box)

            # Draw the defined region if specified
            if self.region is not None:
                self.annotator.draw_region(reg_pts=self.region, color=PURPLE, thickness=self.line_width * 2)

            # Track history and analyse the object.
            prev_position = None
            if self.object_tracker.is_object_has_history(track_id):
                prev_position = self.object_tracker.get_prev_position(track_id)
                if not self.object_tracker.is_customer_a_past_customer(track_id):
                    self.perform_analysis(box, track_id, prev_position, cls, original_frame)

            # From this line and on, the ID may be popped out of all data structures. Therefor we don't use the
            # present client ID because this ID may not have entry there, which will make the program collapse.
            # Instead, we verify it with counted IDs array.
            color = RED if track_id not in  self.object_tracker.counted_ids else GREEN

            annotate_object(track_id, cls, box,
                            self.object_tracker.get_track_id_classifier_data(track_id, ClassifierType.AGE),
                            self.object_tracker.get_track_id_classifier_data(track_id, ClassifierType.GENDER),
                            self.annotator, color)

            # If its re-evaluating time.
            if track_id is self.object_tracker.counted_ids:
                if current_timeslice_frame_count % REEVALUATION_INTERVAL == 0:
                    self.object_tracker.reevaluate_classification(original_frame, track_id, box)

        # Display counts on the frame if a region is defined, and only for the annotated heatmap.
        if self.region is not None:
            self.object_tracker.display_counts(im0, self.annotator)

        # If the ID exist (safety check) annotate its heatmap values.
        if self.object_tracker.track_data.id is not None:
            heatmap_copy = self.heatmap_manager.normalize_heatmap(heatmap_copy)
            im0 = self.heatmap_manager.normalize_heatmap(im0)

        self.object_tracker.save_prev_ids()

        return im0, heatmap_copy

    def perform_analysis(self, box, track_id, prev_position, cls, original_frame):
        if prev_position is None:
            return

        # Calculate the change in the object's y_axis to determine the direction.
        dy = ((box[1] + box[3]) / 2) - prev_position[1]

        # If the object path intersect with the defined line region.
        if len(self.region) < 3 and self.LineString([prev_position, box[:2]]).intersects(self.r_s):

            # If dy < 0, it means that prev_y > current_y, and therefore client moving to the top of the screen.
            if dy < 0:
                if not self.object_tracker.is_current_customer(track_id):
                    self.object_tracker.add_new_client(original_frame, track_id, cls, box)
            else:
                self.object_tracker.remove_client(track_id, cls)

        else:

            # If object is not already customer, and not already in dirty IDs, and do not have recent history recoreded,
            # by the tracker, it is a new dirty entrance ID. The recent history part was added because of flapps the
            # detector has when it misses object for 1-2 frames and then re-track it.
            if (not self.object_tracker.is_current_customer(track_id) and track_id not in self.object_tracker.dirty_ids
                    and track_id not in self.object_tracker.track_data):
                LOGGER.info(f"ID: {track_id}, Dirty Entrance")
                self.object_tracker.count_dirty_id(track_id, cls)
