from .Abstract_Metric import Abstract_Metric
from EasyChemML.Metrik.MetricEnum import MetricType, MetricClass, MetricOutputType, MetricDirection
import deepdiff
from typing import Dict
import numpy as np


class ExactMatch(Abstract_Metric):

    _settings: Dict

    def __init__(self, settings: Dict = {}):
        self._settings = settings
        super().__init__()


    def calc(self, y_true, y_pred, y_predict_proba):
        if y_true == y_pred:
            return 1
        else:
            return 0

    def __eq__(self, other):
        if isinstance(other, ExactMatch):
            result = len(deepdiff.DeepDiff(self._settings, other.get_settings())) == 0
            return result
        else:
            return False

    @staticmethod
    def getMetric_Outputtype():
        return MetricOutputType.singleValue

    @staticmethod
    def getMetricType():
        return MetricType.absolute

    @staticmethod
    def getMetricClass():
        return MetricClass.classification

    def get_settings(self) -> dict:
        return self._settings

    @staticmethod
    def getDirection():
        return MetricDirection.oneIsBest
