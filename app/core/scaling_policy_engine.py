from collections import deque
from typing import Deque

from app.analysis.ARIMA_analyzer import ARIMAAnalyzer
from app.domain.ueSessionInfo import UeSessionInfo


class ScalingPolicyEngine:
    def __init__(
            self,
            analyzer: ARIMAAnalyzer,
            max_history: int | None = None,
            retrain_interval: int | None = None
    ):
        period = analyzer.period
        if max_history is None:
            max_history = period * 4  # bufor 4× longer than min
        if retrain_interval is None:
            retrain_interval = period

        self.analyzer = analyzer
        self.history: Deque[UeSessionInfo] = deque(maxlen=max_history)
        self.retrain_interval = retrain_interval
        self._since_full_train = 0

    def add_sample(self, sample: UeSessionInfo):
        self.history.append(sample)

        self._since_full_train += 1
        if self.retrain_interval and self._since_full_train >= self.retrain_interval:
            self.analyzer.train(list(self.history))
            self._since_full_train = 0

    def evaluate(self, part_of_period: float = 0):
        return self.analyzer.evaluate(list(self.history), part_of_period)
