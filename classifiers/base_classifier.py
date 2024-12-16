import random
from collections import defaultdict

from ultralytics import YOLO
from util1 import box_to_image


class BaseClassifier:

    def __init__(self, weight_path):
        self.model = YOLO(weight_path)
        self.data = defaultdict()

    def classify(self, im0, track_id, box):
        res = self.model(box_to_image(im0, box))
        res_val = res[0].names[res[0].probs.top1].lower()
        res_conf = res[0].probs.top1conf
        res[0].save(f"./logs/classifications/gender/{res_val}/{track_id}_iteration_{random.random()}.jpg")
        self.data[track_id] = (res_val, res_conf)

    def get_track_id_data(self, track_id):
        data = self.data.get(track_id, "Not Detected")
        if data != "Not Detected":
            return data[0]
        else:
            return "Not Detected"

    def get_track_id_conf(self, track_id):
        data = self.data.get(track_id, "Not Detected")
        if data != "Not Detected":
            return data[1]
        else:
            return "Not Detected"

    def remove_id(self, track_id):
        if (self.data.get(track_id, None) is not None):
            self.data.pop(track_id)
