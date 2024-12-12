from data_provider1 import DataProvider
from cloudinary_service import CloudinaryService
from video_analyzer1 import VideoAnalyzer
from util1 import WeightsPath
from video_manager import VideoManager
import cv2


class ComputerVisionService:
    def __init__(self, data):
        self.video_manager = VideoManager()
        self.data_provider = DataProvider(CloudinaryService(), self.video_manager)
        self.data = data
        self.video_analyzer = None

    def start(self):

        # Download video and populate all data fields relating to it
        video_cap = self.data_provider.download_video(self.data["url"])
        self.video_manager.populate_video_data(video_cap, self.data)

        # Initialize the counting line.
        height = (int(self.video_manager.get_height()) -
                  int(self.video_manager.get_height() / 3))
        line_points = [(int(self.video_manager.get_width()), height),
                       (int(self.video_manager.get_start_x()), height)]

        # Set video analyzer
        self.video_analyzer = VideoAnalyzer(self.data_provider, self.video_manager, line_points)


        self.video_analyzer.analyze()