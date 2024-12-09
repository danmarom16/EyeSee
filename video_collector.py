import datetime
import cv2


class VideoCollector:
    def __init__(self, video_path, video_start_time=datetime.datetime.now()):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_start_time = video_start_time
        # width, height, fsp
        self.w, self.h, self.fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                                                   cv2.CAP_PROP_FRAME_HEIGHT,
                                                                   cv2.CAP_PROP_FPS))
        self.video_current_time = video_start_time

    def get_cap(self):
        return self.cap  # Return video capture object for frame-by-frame processing

    def get_width(self):
        return self.w

    def get_height(self):
        return self.h

    def get_fps(self):
        return self.fps

    def get_starting_time(self):
        return self.video_start_time

    def increment_current_time(self):
        time_increment = 1 / self.fps  # Duration of a single frame in seconds
        self.video_current_time += datetime.timedelta(seconds=time_increment)

    def set_current_time(self, interval_time=3):
        self.video_current_time = self.video_current_time + datetime.timedelta(seconds=interval_time)

    def get_current_time(self):
        return self.video_current_time

    def get_start_x(self):
        return 0

    def get_start_y(self):
        return 0
