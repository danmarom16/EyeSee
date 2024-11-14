import time

from ultralytics import YOLO
import cv2
from ultralytics.solutions import Heatmap
import json
import numpy as np
from ultralytics.utils.plotting import Annotator


class VideoAnalyzer(Heatmap):

    def __init__(self, video_collector, **kwargs):
        super().__init__(**kwargs)

        self.video_collector = video_collector
        # Initialize dwell time tracking
        self.dwell_times = {}  # {track_id: {"entrance": timestamp, "exit": timestamp, "dwell": duration}}
        self.colormap = cv2.COLORMAP_PARULA if self.CFG["colormap"] is None else self.CFG["colormap"]

        # init video writer
        self.video_writer = cv2.VideoWriter(
            "output.avi",
            cv2.VideoWriter.fourcc(*"mp4v"),
            float(self.video_collector.get_fps()),
            (int(self.video_collector.get_width()), int(self.video_collector.get_height()))
        )

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
                if track_id not in self.dwell_times:
                    self.dwell_times[track_id] = {"entrance": self.video_collector.get_starting_time()}
            else:
                self.out_count += 1
                self.classwise_counts[self.names[cls]]["OUT"] += 1
                if track_id in self.dwell_times:
                    self.dwell_times[track_id]["exit"] = self.video_collector.get_current_time()
                    self.dwell_times[track_id]["dwell"] = self.dwell_times[track_id]["exit"] - self.dwell_times[track_id]["entrance"]
                    self.dwell_times[track_id]["exit_type"] = "Exited Store"

        elif len(self.region) < 3 and self.LineString([prev_position, box[:2]]).intersects(self.r_s):
            self.counted_ids.append(track_id)
            # For linear region
            if dx > 0 and dy > 0:
                self.in_count += 1
                self.classwise_counts[self.names[cls]]["IN"] += 1
                if track_id not in self.dwell_times:
                    self.dwell_times[track_id] = {"entrance": self.video_collector.get_current_time()}
            else:
                self.out_count += 1
                self.classwise_counts[self.names[cls]]["OUT"] += 1
                if track_id in self.dwell_times:
                    self.dwell_times[track_id]["exit"] = self.video_collector.get_current_time()
                    self.dwell_times[track_id]["dwell"] = self.dwell_times[track_id]["exit"] - self.dwell_times[track_id]["entrance"]
                    self.dwell_times[track_id]["exit_type"] = "Exited Store"

    def remove_lost_ids(self):
        # Identify IDs in dwell_times that are no longer in track_ids
        lost_ids = [track_id for track_id in self.dwell_times if track_id not in self.track_ids]

        for track_id in lost_ids:
            # Mark exit time and calculate dwell time if not already recorded
            if "exit" not in self.dwell_times[track_id]:
                self.dwell_times[track_id]["exit"] = self.video_collector.get_current_time()
                self.dwell_times[track_id]["dwell"] = self.dwell_times[track_id]["exit"] - self.dwell_times[track_id][
                    "entrance"]
                self.dwell_times[track_id]["exit_type"] = "Exited Frame"

            # Remove the track_id from dwell_times as it is no longer detected
            self.dwell_times.pop(track_id, None)

    def analyze_video_frame(self, im0):
        # Run detection with tracking enabled

        if not self.initialized:
            self.heatmap = np.zeros_like(im0, dtype=np.float32)
        self.initialized = True

        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        # Remove IDs that are no longer in the frame
        self.remove_lost_ids()

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.heatmap_effect(box)

            if self.region is not None:
                self.annotator.draw_region(reg_pts=self.region, color=(104, 9, 123), thickness=self.line_width * 2)
                self.store_tracking_history(track_id, box)
                self.store_classwise_counts(cls)

                prev_position = None
                if len(self.track_history[track_id]) > 1:
                    prev_position = self.track_history[track_id][-2]
                self.count_objects(self.track_line, box, track_id, prev_position, cls)

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

        self.display_output(im0)
        return {"counter": self.in_count - self.out_count}, im0, self.dwell_times
