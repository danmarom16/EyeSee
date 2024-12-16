from object_tracker import ObjectTracker
from util1 import init_writer, SAVING_INTERVAL
from frame_analyzer import FrameAnalyzer
from heatmap_manager import HeatmapManager
from object_counter import ObjectCounter
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_SOL_DICT, LOGGER
from ultralytics import YOLO


class VideoAnalyzer:
    def __init__(self, data_provider, video_manager, line_points, **kwargs):

        # Load Ultralytics config and update with args
        DEFAULT_SOL_DICT.update(kwargs)
        DEFAULT_CFG_DICT.update(kwargs)
        self.CFG = {**DEFAULT_SOL_DICT, **DEFAULT_CFG_DICT}
        LOGGER.info(f"Ultralytics Solutions: ✅ {DEFAULT_SOL_DICT}")
        self.model = YOLO(self.CFG["model"] if self.CFG["model"] else "yolov8n.pt")

        # Custom classes for flow handling
        self.data_provider = data_provider
        self.video_manager = video_manager
        self.heatmap_manager = HeatmapManager(self.CFG)

        # Custom classes for analysis logic.
        self.object_tracker = ObjectTracker(ObjectCounter(line_points, self.model, self.video_manager),
                                            self.video_manager, self.model, self.CFG)
        self.frame_analyzer = FrameAnalyzer(self.object_tracker, self.CFG, self.heatmap_manager)

        self.video_writer = init_writer("./logs/outputs/output.mp4", video_manager)
        self.annotator = None

    def initialize(self, first_frame):

        # Initialize heatmap
        self.heatmap_manager.initialize_heatmap(first_frame.copy())
        self.frame_analyzer.initialize(first_frame)

    def analyze(self):
        cap = self.video_manager.get_cap()
        ret, frame = cap.read()
        heatmap_image = None
        prev_heatmap_image = None

        if ret:
            self.initialize(frame)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                self.data_provider.provide(prev_heatmap_image)
                break

            # If this is the 1st frame analyzed in this timeslice, set its start time to the current time.
            if self.video_manager.get_current_timeslice_frame_count() == 0:
                self.video_manager.set_current_timeslice_start(self.video_manager.get_current_time())


            # Run frame analysis
            heatmap_image = self.frame_analyzer.analyze(frame, self.video_manager.get_current_timeslice_frame_count())

            # Save prev heatmap image for saving.
            prev_heatmap_image = heatmap_image

            # Write frame
            self.video_writer.write(heatmap_image)

            if self.video_manager.get_current_timeslice_frame_count() % SAVING_INTERVAL == 0:
                self.save_and_reset()

        # Release resources
        cap.release()
        self.video_writer.release()


    def save_and_reset(self):
        self.data_provider.local_save(
            self.object_tracker.object_counter.in_count - self.object_tracker.object_counter.out_count,
            self.object_tracker.age_classifier.data, self.object_tracker.dwell_times,
            self.object_tracker.gender_classifier.data, self.object_tracker.past_customers_in_timeslice
        )
        self.video_manager.set_new_time_slice()
        self.object_tracker.past_customers_in_timeslice = []
