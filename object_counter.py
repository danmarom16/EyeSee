from util import ExitType, CountType, DwellTime


class ObjectCounter:

    def __init__(self, model, video_manager):

        self.model = model

        # Video Manager - will be responsible to handle all metadata regarding the video.
        self.video_manager = video_manager

        # Counters
        self.in_count = 0
        self.out_count = 0
        self.dirty_in_count = 0
        self.dirty_out_count = 0

        # Dict stores the different types of counts and their value.
        self.classwise_counts = {}

        # Variables to handle showing of counter on the screen.
        self.show_in = True
        self.show_out = True

    def store_classwise_counts(self, cls):
        if self.model.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.model.names[cls]] = {CountType.CLEAN_IN.value: 0, CountType.DIRTY_IN.value: 0,
                                                            CountType.CLEAN_OUT.value: 0, CountType.DIRTY_OUT.value: 0}

    def count_client_dirty_exit(self, dwell_times, track_id, cls=0):

        dwell_times[track_id][DwellTime.EXIT.value] = self.video_manager.get_current_time()
        dwell_times[track_id][DwellTime.DWELL.value] = (dwell_times[track_id][DwellTime.EXIT.value] -
                                                        dwell_times[track_id][DwellTime.ENTRANCE.value])
        dwell_times[track_id][DwellTime.EXIT_TYPE.value] = ExitType.DIRTY.value

        self.dirty_out_count += 1
        self.classwise_counts[self.model.names[cls]][CountType.DIRTY_OUT.value] += 1
        self.in_count -= 1

    def count_in(self, cls):
        self.in_count += 1
        if self.model.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.model.names[cls]] = {CountType.CLEAN_IN.value: 0, CountType.DIRTY_IN.value: 0,
                                                            CountType.CLEAN_OUT.value: 0, CountType.DIRTY_OUT.value: 0}
        self.classwise_counts[self.model.names[cls]][CountType.CLEAN_IN.value] += 1
        self.store_classwise_counts(cls)

    def count_out(self, cls):
        self.out_count += 1
        self.classwise_counts[self.model.names[cls]][CountType.CLEAN_OUT.value] += 1

    def count_dirty_entrance(self, cls=0):
        self.classwise_counts[self.model.names[cls]][CountType.DIRTY_IN.value] += 1

    def count_dirty_and_dirty_exit(self, cls=0):
        self.dirty_out_count += 1
        self.classwise_counts[self.model.names[cls]][CountType.DIRTY_IN.value] -= 1

    def display_counts(self):
        return self.classwise_counts

    def calculate_current_count(self):
        return self.in_count - self.out_count

    def decrement_count(self, count_to_decrement, cls=0):
        self.classwise_counts[self.model.names[cls]][count_to_decrement] -= 1
        self.dirty_in_count -= 1
