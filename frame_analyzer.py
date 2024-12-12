from shapely.geometry import LineString
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_SOL_DICT, LOGGER
from ultralytics.utils.plotting import Annotator
from shapely.geometry import LineString
from util1 import annotate_object, RED, BLUE, PURPLE, REEVALUATION_INTERVAL

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
        original_frame = im0.copy()

        # Extract tracks
        self.object_tracker.extract_tracks(im0)

        self.annotator = Annotator(im0, line_width=self.line_width)

        for box, track_id, cls in zip(self.object_tracker.boxes, self.object_tracker.track_ids,
                                      self.object_tracker.clss):
            self.heatmap_manager.apply_heatmap_effect(box)
            self.object_tracker.store_tracking_history(track_id, box)

            # Draw the defined region if specified
            if self.region is not None:
                self.annotator.draw_region(reg_pts=self.region, color=PURPLE, thickness=self.line_width * 2)

            # Count the object in as all objects detected in the  first frame are counted as clients.
            self.object_tracker.count_in(track_id, cls, box)

            annotate_object(track_id, cls, box,
                            self.object_tracker.age_classifier.data.get(track_id, "Not Detected"),
                            self.object_tracker.age_classifier.data.get(track_id, "Not Detected"), self.annotator,
                            RED)

            # Display counts on the frame if a region is defined
            if self.region is not None:
                self.object_tracker.display_countes(self, im0, self.annotator)

            if self.object_tracker.data.id is not None:
                im0 = self.heatmap_manager.normalize_heatmap(im0)

        return im0


    def analyze(self, im0, current_timeslice_frame_count):
        original_frame = im0.copy()

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

            # Annotate all objects in blue
            annotate_object(track_id, cls, box, self.object_tracker.age_classifier.data.get(track_id, "Not Detected"),
                            self.object_tracker.age_classifier.data.get(track_id, "Not Detected"), self.annotator,
                            BLUE)

            # Add annotation exclusively to the clients of the store - red.
            if track_id in self.object_tracker.counted_ids:
                annotate_object(track_id, cls, box,
                                self.object_tracker.age_classifier.data.get(track_id, "Not Detected"),
                                self.object_tracker.age_classifier.data.get(track_id, "Not Detected"), self.annotator,
                                RED)
                if current_timeslice_frame_count % REEVALUATION_INTERVAL == 0:
                    self.object_tracker.reevaluate_classification(original_frame, track_id, box)

            # Display counts on the frame if a region is defined
            if self.region is not None:
                self.object_tracker.display_countes(self, im0, self.annotator)

            if self.object_tracker.data.id is not None:
                im0 = self.heatmap_manager.normalize_heatmap(im0)

        return im0

    def perform_analysis(self, box, track_id, prev_position, cls, original_frame):
        if prev_position is None:
            return

        dy = ((box[1] + box[3]) / 2) - prev_position[1]

        # If the object path intersect with the defined line region.
        if len(self.region) < 3 and self.LineString([prev_position, box[:2]]).intersects(self.r_s):
            # If dy < 0, it means that prev_y > current_y, and therefore client moving to the top of the screen.
            if dy < 0:
                self.object_tracker.count_in(track_id, cls, box)
                self.object_tracker.classify(track_id, original_frame, box)
            else:
                self.object_tracker.count_out(track_id, cls)