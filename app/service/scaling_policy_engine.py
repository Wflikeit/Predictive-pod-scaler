from dataclasses import dataclass
from typing import List

from app.domain.ueSessionInfo import UeSessionInfo

MIN_HISTORY_FOR_SLOPE = 5


@dataclass
class TrendResult:
    delta: float
    slope: float
    current_sessions: int


class ScalingPolicyEngine:
    def __init__(self):
        self.history: List[UeSessionInfo] = []

    def add_sample(self, sample: UeSessionInfo):
        """
        Adds a new sample to the internal history buffer (max length 50).
        """
        self.history.append(sample)
        self.history = self.history[-50:]  # trim to max length

    def analyze_short_term(self) -> float:
        """
        Calculates short-term delta (change between last two samples).
        """
        if len(self.history) < 2:
            return 0.0
        return self.history[-1].session_count - self.history[-2].session_count

    # TODO: Test its accuracy and add improvements if needed
    def analyze_trend(self) -> float:
        if len(self.history) < MIN_HISTORY_FOR_SLOPE:
            return 0.0

        n = len(self.history)
        weights = [i + 1 for i in range(n)]
        x = list(range(n))
        y = [s.session_count for s in self.history]

        avg_x = sum(w * x_ for w, x_ in zip(weights, x)) / sum(weights)
        avg_y = sum(w * y_ for w, y_ in zip(weights, y)) / sum(weights)

        numerator = sum(w * (xi - avg_x) * (yi - avg_y)
                        for w, xi, yi in zip(weights, x, y))
        denominator = sum(w * (xi - avg_x) ** 2 for w, xi in zip(weights, x))

        return numerator / denominator if denominator else 0.0

    def get_latest_session_count(self) -> int:
        """
        Returns the latest number of sessions (or 0 if not available).
        """
        return self.history[-1].session_count if self.history else 0

    def evaluate(self) -> TrendResult:
        """
        Returns all useful analysis results in one object.
        """
        return TrendResult(
            delta=self.analyze_short_term(),
            slope=self.analyze_trend(),
            current_sessions=self.get_latest_session_count()
        )
