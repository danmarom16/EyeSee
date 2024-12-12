from base_classifier import BaseClassifier
from util1 import WeightsPath


class AgeClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(WeightsPath.AGE_CLASSIFIER)
