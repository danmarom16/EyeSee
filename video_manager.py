import datetime
import cv2


class VideoManager:
    def __init__(self):

        # Global time related variables for calculating correct time of the video.
        self.current_time = None
        self.date = None
        self.analysis_start_time = None
        self.analysis_end_time = None

        # Timeslice (local) related variables for calculating correct time of the video.
        self.current_timeslice_start = None
        self.current_timeslice_frame_count = 0

        # Video meta-data
        self.fps = None
        self.h = None
        self.w = None
        self.length = None

        # Will store CV2 VideoCapture and encapsulate it.
        self.cap = None

        # JobID that is used by the server to recognize this analysis process.
        self.jobId = None

    def populate_video_data(self, video_cap, data):

        # Global time related population
        self.date = data["date"]
        self.analysis_start_time = datetime.datetime.strptime(f"{data['date']} {data['start']}", "%Y-%m-%d %H:%M:%S")
        self.analysis_end_time = datetime.datetime.strptime(f"{data['date']} {data['end']}", "%Y-%m-%d %H:%M:%S")
        self.current_time = self.analysis_start_time

        # Video Capture object that is already initialized by the data provider.
        self.cap = video_cap

        # Meta data related population.
        self.length = data["length"]
        self.w, self.h, self.fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                                                   cv2.CAP_PROP_FRAME_HEIGHT,
                                                                   cv2.CAP_PROP_FPS))
        # JobID
        self.jobId = data["jobId"]

    def increment_current_time(self):
        time_increment = 1 / self.fps  # Duration of a single frame in seconds
        self.current_time += datetime.timedelta(seconds=time_increment)

    def increment_frame_count(self):
        self.current_timeslice_frame_count += 1

    def read_frame(self):
        return self.cap.read()

    def cap_release(self):
        self.cap.release()

# ----------------------Getters--------------------------
    def get_fps(self):
        return self.fps

    def get_width(self):
        return self.w

    def get_height(self):
        return self.h

    def get_date(self):
        return self.date

    def get_current_timeslice_start(self):
        return self.current_timeslice_start

    def get_current_time(self):
        return self.current_time

    def get_analysis_start_time(self):
        return self.analysis_start_time

    def get_length(self):
        return self.length

    def get_job_id(self):
        return self.jobId

    def get_current_timeslice_frame_count(self):
        return self.current_timeslice_frame_count

    def get_start_y(self):
        return 0
    def get_start_x(self):
        return 0

# ----------------------Setters--------------------------
    def set_current_time(self, interval_time=3):
        self.current_time = self.current_time + datetime.timedelta(seconds=interval_time)

    def set_new_time_slice(self):
        self.current_timeslice_frame_count = 0

    def set_current_timeslice_start(self, time):
        self.current_timeslice_start = time

# ----------------------Booleans--------------------------
    def has_frames_left(self):
        return self.cap.isOpened()