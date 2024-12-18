from data_provider import DataProvider
from cloudinary_service import CloudinaryService
from video_analyzer import VideoAnalyzer
from util import WeightsPath
from video_manager import VideoManager


class ComputerVisionService:
    def __init__(self, data):

        # Video Manager - will be responsible to handle all metadata regarding the video.
        self.video_manager = VideoManager()

        # Data Provider - will handle data flow in the program, from local saving to triggering API calls. Dynamically
        # injected with video uploading and downloading service - Cloudinary. If we we're to try different logic, We
        # would have just created VideoUploadDownload interface that CloudinaryService would've implemented, and the
        # different services that we would try, would just implement this interface and this code remains the same.
        # Open for extension, but close for modification.
        self.data_provider = DataProvider(CloudinaryService(), self.video_manager)

        # Data given by the server through the API call. Consist of jobId and a download link for the video.
        self.data = data

        # Will handle the analysis flow with the use of multiple objects to enforce Single Responsibility.
        self.video_analyzer = None

    def start(self):

        # Download video and populate all data fields relating to it.
        video_cap = self.data_provider.download_video(self.data["url"])
        self.video_manager.populate_video_data(video_cap, self.data)

        # Initialize the counting line.
        height = (int(self.video_manager.get_height()) -
                  int(self.video_manager.get_height() / 3))
        line_points = [(int(self.video_manager.get_width()), height),
                       (int(self.video_manager.get_start_x()), height)]

        # Set video analyzer
        self.video_analyzer = VideoAnalyzer(self.data_provider, self.video_manager, region=line_points,
                                            model=WeightsPath.PERSON_TRACKER, classes=[0])

        self.video_analyzer.analyze()
