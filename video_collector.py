import datetime
import cv2


class VideoCollector:
    def __init__(self):
        self.current_time = None
        self.fps = None
        self.h = None
        self.w = None
        self.date = None
        self.start_time = None
        self.end_time = None
        self.length = None
        self.cap = None
        self.jobId = None

    def populate_video_data(self, video_cap, data):
        self.cap = video_cap
        self.date = data["date"]
        self.start_time = datetime.datetime.strptime(f"{data['date']} {data['start']}", "%Y-%m-%d %H:%M:%S")
        self.end_time = datetime.datetime.strptime(f"{data['date']} {data['end']}", "%Y-%m-%d %H:%M:%S")
        self.length = data["length"]
        self.current_time = self.start_time  # Use datetime object instead of string
        self.jobId = data["jobId"]
        self.w, self.h, self.fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                                                   cv2.CAP_PROP_FRAME_HEIGHT,
                                                                   cv2.CAP_PROP_FPS))
    def get_end_time(self):
        return self.end_time

    def get_cap(self):
        return self.cap  # Return video capture object for frame-by-frame processing

    def get_width(self):
        return self.w

    def get_height(self):
        return self.h

    def get_fps(self):
        return self.fps

    def get_start_time(self):
        return self.start_time

    def increment_current_time(self):
        time_increment = 1 / self.fps  # Duration of a single frame in seconds
        self.current_time += datetime.timedelta(seconds=time_increment)

    def set_current_time(self, interval_time=3):
        self.current_time = self.current_time + datetime.timedelta(seconds=interval_time)

    def get_current_time(self):
        return self.current_time

    def get_start_x(self):
        return 0

    def get_start_y(self):
        return 0

    def get_job_id(self):
        return self.jobId
