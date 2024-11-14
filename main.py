import time

from flask import Flask, request, jsonify
from data_provider import DataProvider
from data_aggregator import DataAggregator
from video_analyzer import VideoAnalyzer
from video_collector import VideoCollector
import cv2


def main():
    video_path = "resources/video.mp4"

    # Initialize collectors
    video_collector = VideoCollector(video_path)

    line_points = [(0, 400), (int(video_collector.get_width() - video_collector.get_width() / 3),
                              video_collector.get_height())]

    video_analyzer = VideoAnalyzer(video_collector, show=True, model="yolo11n.pt", colormap=cv2.COLORMAP_PARULA,
                                   region=line_points, show_in=True, show_out=True)

    data_provider = DataProvider()  # Initialize DataProvider for database operations
    data_aggregator = DataAggregator(data_provider, video_collector)  # Save every 100 frames

    cap = video_collector.get_cap()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Analyze frame and aggregate data
        raw_processed_frame_data, heatmap_image, dwell_times = video_analyzer.analyze_video_frame(frame)
        data_aggregator.add_frame_data(raw_processed_frame_data, heatmap_image, dwell_times)
    cap.release()  # Release video after processing


if __name__ == '__main__':
    main()
