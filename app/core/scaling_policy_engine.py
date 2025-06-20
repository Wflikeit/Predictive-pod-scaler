from collections import deque
from typing import Deque

from analysis.ARIMA_analyzer import ARIMAAnalyzer
from app.domain.ueSessionInfo import UeSessionInfo


class ScalingPolicyEngine:
    def __init__(
        self,
        analyzer: ARIMAAnalyzer,
        max_history: int | None = None,      # ← może być None
        retrain_interval: int | None = None
    ):
        period = analyzer.period
        if max_history is None:
            max_history = period * 4         # bufor 4× dłuższy od minimum
        if retrain_interval is None:
            retrain_interval = period        # retrain co 1 okres

        self.analyzer = analyzer
        self.history: Deque[UeSessionInfo] = deque(maxlen=max_history)
        self.retrain_interval = retrain_interval
        self._since_full_train = 0


    def add_sample(self, sample: UeSessionInfo):
        self.history.append(sample)

        # 2) ewentualny pełny retrain co retrain_interval próbek
        if self.retrain_interval and self._since_full_train >= self.retrain_interval:
            self.analyzer.train(list(self.history))
            self._since_full_train = 0
        else:
            self._since_full_train += 1

    def evaluate(self, part_of_period: float = 0):
        return self.analyzer.evaluate(list(self.history), part_of_period)
