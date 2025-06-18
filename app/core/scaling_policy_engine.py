from collections import deque
from typing import List, Deque

from app.domain.trendResult import TrendResult
from app.domain.trend_analyzer import TrendAnalyzer
from app.domain.ueSessionInfo import UeSessionInfo


class ScalingPolicyEngine:
    def __init__(self, analyzer: TrendAnalyzer, max_history: int = 50):
        self.analyzer = analyzer
        self.history: Deque[UeSessionInfo] = deque(maxlen=max_history)
        self.max_history = max_history

    def add_sample(self, sample: UeSessionInfo):
        self.history.append(sample)

    def evaluate(self) -> TrendResult:
        return self.analyzer.evaluate(self.history)

    def get_latest_session_count(self) -> int:
        return self.history[-1].session_count if self.history else 0
