import video_manager
from util1 import ExitType, EntranceType
from ultralytics.utils import LOGGER

class ObjectCounter:


    def __init__(self, line_points, model, video_manager):

        self.model = model
        self.video_manager = video_manager

        # Counters
        self.in_count = 0
        self.out_count = 0
        self.dirty_in_count = 0
        self.dirty_out_count = 0

        # Data Structures
        self.classwise_counts = {}
        self.region_initialized = False
        self.line_points = line_points

        # Variables
        self.show_in = True
        self.show_out = True



    #Override
    def store_classwise_counts(self, cls):
        if self.model.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.model.names[cls]] = {"CLEAN_IN": 0, "DIRTY_IN": 0, "CLEAN_OUT": 0, "DIRTY_OUT": 0}


    def count_client_dirty_exit(self, dwell_times, track_id, cls=0):

        dwell_times[track_id]["exit"] = self.video_manager.get_current_time()
        dwell_times[track_id]["dwell"] = dwell_times[track_id]["exit"] - dwell_times[track_id][
            "entrance"]
        dwell_times[track_id]["exit_type"] = ExitType.DIRTY
        self.dirty_out_count += 1
        self.in_count -= 1
        self.classwise_counts[self.model.names[cls]]["DIRTY_OUT"] += 1

    def count_not_client_dirty_exit(self, track_id, dirty_ids, cls=0):
        dirty_ids.remove(track_id)
        self.classwise_counts[self.model.names[cls]]["DIRTY_IN"] -= 1
        LOGGER.info(f"ID: {track_id} was NOT counted and performed Dirty Exit")

    def count_in(self, track_id, cls, counted_ids):
        counted_ids.append(track_id)
        self.in_count += 1
        if self.model.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.model.names[cls]] = {"CLEAN_IN": 0, "DIRTY_IN": 0, "CLEAN_OUT": 0, "DIRTY_OUT": 0}
        self.classwise_counts[self.model.names[cls]]["CLEAN_IN"] += 1
        self.store_classwise_counts(cls)

    def count_out(self, track_id, cls):
        self.out_count += 1
        self.classwise_counts[self.model.names[cls]]["CLEAN_OUT"] += 1

    def decrement_count(self, count_to_decrement, cls=0):
        self.classwise_counts[self.model.names[cls]][count_to_decrement] -= 1
        self.dirty_in_count -= 1

    def display_counts(self, im0):
        labels_dict = {
            str.capitalize(key): f"{'CLEAN_IN ' + str(value['CLEAN_IN']) if self.show_in else ''} "
            f"{' DIRTY_IN ' + str(value['DIRTY_IN']) if self.show_out else ''}"
            f"{' CLEAN_OUT ' + str(value['CLEAN_OUT']) if self.show_out else ''}"
            f"{' DIRTY_OUT ' + str(value['DIRTY_OUT']) if self.show_out else ''}"
            .strip()
            for key, value in self.classwise_counts.items()
            if value['CLEAN_IN'] != 0 or value['DIRTY_IN'] != 0 or value['CLEAN_OUT'] != 0 or value['DIRTY_OUT'] != 0
        }

