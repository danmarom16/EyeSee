import time

from flask import Flask, request, jsonify
from data_provider import DataProvider
from data_aggregator import DataAggregator
from video_analyzer import VideoAnalyzer
from video_collector import VideoCollector
import cv2


# TODO: Solve the hours bug. It increases end time by 3, and start_time by 3, instead of doing their difference by 3. (can see in logs)
def main():

    video_path = "resources/video.mp4"

    # Initialize collectors
    video_collector = VideoCollector(video_path)
    video_analyzer = VideoAnalyzer(video_collector)
    data_provider = DataProvider()  # Initialize DataProvider for database operations
    data_aggregator = DataAggregator(data_provider, video_collector)  # Save every 100 frames

    cap = video_collector.get_cap()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Analyze frame and aggregate data
        raw_processed_frame_data, heatmap_image = video_analyzer.analyze_video_frame(frame)
        data_aggregator.add_frame_data(raw_processed_frame_data, heatmap_image)
    cap.release()  # Release video after processing


if __name__ == '__main__':
    main()