from ultralytics import YOLO
import cv2
from ultralytics import solutions
import json

class VideoAnalyzer:
    def __init__(self, video_collector):
        # Initialize YOLOv8 model
        self.model = YOLO("yolo11n.pt")

        # cap
        self.video_collector = video_collector

        # counting line points
        self.line_points = [(0, 400), (int(self.video_collector.get_width() - self.video_collector.get_width()/3),
                                       self.video_collector.get_height())]

        self.heatmap = solutions.Heatmap(show=True, model="yolo11n.pt", colormap=cv2.COLORMAP_PARULA,
                                         region=self.line_points, show_in=True, show_out=True)

        # init video writer
        self.video_writer = cv2.VideoWriter(
            "output.avi",
            cv2.VideoWriter.fourcc(*"mp4v"),
            float(self.video_collector.get_fps()),
            (int(self.video_collector.get_width()), int(self.video_collector.get_height()))
        )
    def analyze_video_frame(self, frame):
        # Run detection with tracking enabled

        heatmap_image = self.heatmap.generate_heatmap(frame)

        # Extract data.
        counter = self.heatmap.in_count - self.heatmap.out_count
        self.video_writer.write(heatmap_image)

        return {'counter': counter}, heatmap_image

