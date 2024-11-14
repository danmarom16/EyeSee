from collections import defaultdict
import numpy as np
import datetime


class DataAggregator:
    def __init__(self, data_provider, video_collector, save_interval_seconds=3):
        self.heatmap = defaultdict(int)
        self.video_collector = video_collector
        self.unique_individuals = set()
        self.data_provider = data_provider  # Instance of DataProvider
        self.save_interval_seconds = int(
            video_collector.get_fps() * save_interval_seconds)  # Number of frames before triggering save
        self.frame_count = 0
        self.reset_data()

    def reset_data(self):
        # Reset aggregator data after each save
        self.unique_individuals = set()
        self.heatmap = defaultdict(int)

    def add_frame_data(self, frame_data, frame, dwell_times):
        self.frame_count += 1  # Increment frame counter
        self.video_collector.set_current_time(1)  # adds 3 seconds

        # Save to database if frame count reaches save interval
        if self.frame_count >= self.save_interval_seconds:

            frame_data['start_time'] = self.video_collector.get_starting_time()
            frame_data['end_time'] = self.video_collector.get_current_time()
            frame_data['dwell'] = dwell_times

            self.trigger_save(frame_data, frame)
            self.reset_data()  # Reset aggregator data
            self.frame_count = 0  # Reset frame count
            self.video_collector.set_start_to_current_time()

    def trigger_save(self, frame_data, frame):
        # Prepare data for saving and delegate to DataProvider

        # Delegate saving to DataProvider
        self.data_provider.save_data(frame_data, frame)
