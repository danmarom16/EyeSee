import random
from collections import defaultdict

from ultralytics import YOLO
from util1 import box_to_image


class BaseClassifier:

    def __init__(self, weight_path):

        # YOLO Model that will be used in order to track customers.
        self.model = YOLO(weight_path.value)

        # Will obtain the classification results.
        self.data = defaultdict()

    def classify(self, im0, track_id, box, classification_type):

        # Run YOLO Classifier on the bounding box.
        res = self.model(box_to_image(im0, box))

        # Extract the prediction.
        res_val = res[0].names[res[0].probs.top1].lower()

        # Extract the confidence of the prediction,
        res_conf = res[0].probs.top1conf

        # Local save for internal debugging.
        res[0].save(f"./logs/classifications/{classification_type}/{res_val}/{track_id}_{random.random()}.jpg")

        # Save this ID classification.
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
