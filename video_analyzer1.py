from object_tracker import ObjectTracker
from util1 import init_writer, OUTPUT_VID_PATH
from frame_analyzer import FrameAnalyzer
from heatmap_manager import HeatmapManager
from object_counter import ObjectCounter
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_SOL_DICT, LOGGER
from ultralytics import YOLO

class VideoAnalyzer:
    def __init__(self, data_provider, video_manager, **kwargs):

        # Load Ultralytics config and update with args
        DEFAULT_SOL_DICT.update(kwargs)
        DEFAULT_CFG_DICT.update(kwargs)
        self.CFG = {**DEFAULT_SOL_DICT, **DEFAULT_CFG_DICT}
        LOGGER.info(f"Ultralytics Solutions: ✅ {DEFAULT_SOL_DICT}")

        # YOLO Model that will be used in order to track customers - Only Video Analyzer will create a new instance.
        # All other classes will receive references of this one - Singleton.
        self.model = YOLO(self.CFG["model"].value if self.CFG["model"] else "yolov8n.pt")

        # Data Provider - will handle data flow in the program, from local saving to triggering API calls.
        self.data_provider = data_provider

        # Video Manager - will be responsible to handle all metadata regarding the video.
        self.video_manager = video_manager

        # Heatmap Manager - will handle all operations related heatmap.
        self.heatmap_manager = HeatmapManager(self.CFG)

        # Object Tracker - will handle all the tracking logic. All relevant data structures that are used for the logic
        # of tracking will live in its scope. It is using also Object Counter who's single responsibility is to handle
        # the counting logic in the program. Notice that ObjectCounter is dynamically injected into Object Tracker,
        # which will allows us to change counter with no need to change code while they will implement the same
        # interface.
        self.object_tracker = ObjectTracker(ObjectCounter(self.model, self.video_manager, self.CFG),
                                            self.video_manager, self.model, self.CFG)

        # Frame Analyzer -
        self.frame_analyzer = FrameAnalyzer(self.object_tracker, self.CFG, self.heatmap_manager)

        self.video_writer = init_writer(self.data_provider.base_dir + "/" + OUTPUT_VID_PATH, video_manager)
        self.annotator = None

        # Set saving interval to number of frames per second.
        self.saving_interval = self.video_manager.get_fps()

    def initialize(self, first_frame):

        # Initialize heatmap
        self.heatmap_manager.initialize_heatmap(first_frame.copy())
        return self.frame_analyzer.initialize(first_frame)

    def analyze(self):

        ret, frame = self.video_manager.read_frame()
        heatmap_image = None
        prev_annotated_heatmap_image = None
        prev_clean_heatmap_image = None

        if ret:
            im0, heatmap_copy = self.initialize(frame)
            self.video_manager.increment_current_time()

        while self.video_manager.has_frames_left():
            ret, frame = self.video_manager.read_frame()
            if not ret:
                self.data_provider.provide(prev_annotated_heatmap_image, prev_clean_heatmap_image)
                break

            # If this is the 1st frame analyzed in this timeslice, set its local start time to the current time.
            if self.video_manager.get_current_timeslice_frame_count() == 0:
                self.video_manager.set_current_timeslice_start(self.video_manager.get_current_time())

            # Run frame analysis
            annotated_heatmap_image, clean_heatmap_image = (
                self.frame_analyzer.analyze(frame, self.video_manager.get_current_timeslice_frame_count()))

            # Save prev heatmap image for saving.
            prev_annotated_heatmap_image = annotated_heatmap_image
            prev_clean_heatmap_image = clean_heatmap_image

            # Write frame
            self.video_writer.write(annotated_heatmap_image)

            # Increment number of counted frames.
            self.video_manager.increment_frame_count()

            # Increment time slice time
            self.video_manager.increment_current_time()

            if self.video_manager.get_current_timeslice_frame_count() % self.saving_interval == 0:
                self.save_and_reset()

        # Release resources
        self.video_manager.cap_release()
        self.video_writer.release()


    def save_and_reset(self):
        self.data_provider.local_save(
            self.object_tracker.calculate_current_count(),
            self.object_tracker.age_classifier.data, self.object_tracker.dwell_times,
            self.object_tracker.gender_classifier.data, self.object_tracker.past_customers_in_timeslice
        )
        self.video_manager.set_new_time_slice()
        self.object_tracker.past_customers_in_timeslice = []
