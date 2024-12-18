from classifiers.base_classifier import BaseClassifier
from util import WeightsPath


class GenderClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(WeightsPath.GENDER_CLASSIFIER)
