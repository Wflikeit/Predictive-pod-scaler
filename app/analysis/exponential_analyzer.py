from typing import Deque, List, Tuple

from app.domain.trendResult import TrendResult
from app.domain.trend_analyzer import TrendAnalyzer
from app.domain.ueSessionInfo import UeSessionInfo


class ExponentialRegressionAnalyzer(TrendAnalyzer):
    def __init__(self, alpha: float = 0.1, min_history_length: int = 5):
        self.alpha = alpha
        self.min_history_length = min_history_length
        self._weights_cache: dict[int, Tuple[List[float], float]] = {}

    def evaluate(self, history: Deque[UeSessionInfo]) -> TrendResult:
        return TrendResult(
            delta=self._analyze_short_term(history),
            slope=self._analyze_trend(history),
            current_sessions=history[-1].session_count if history else 0
        )

    def _analyze_short_term(self, history: Deque[UeSessionInfo]) -> float:
        if len(history) < 2:
            return 0.0
        return history[-1].session_count - history[-2].session_count

    def _analyze_trend(self, history: Deque[UeSessionInfo]) -> float:
        if len(history) < self.min_history_length:
            return 0.0

        n = len(history)
        sessions = [s.session_count for s in history]
        indices = list(range(n))
        weights, weight_sum = self._get_weights_and_sum(n)

        avg_x = sum(w * i for w, i in zip(weights, indices)) / weight_sum
        avg_y = sum(w * s for w, s in zip(weights, sessions)) / weight_sum

        weighted_cov = sum(
            w * (i - avg_x) * (s - avg_y)
            for w, i, s in zip(weights, indices, sessions)
        )
        weighted_var = sum(
            w * (i - avg_x) ** 2
            for w, i in zip(weights, indices)
        )

        return weighted_cov / weighted_var if weighted_var else 0.0

    def _get_weights_and_sum(self, n: int) -> Tuple[List[float], float]:
        if n not in self._weights_cache:
            self._weights_cache[n] = self._compute_exp_weights(n)
        return self._weights_cache[n]

    def _compute_exp_weights(self, n: int) -> Tuple[List[float], float]:
        weights = [self.alpha * (1 - self.alpha) ** (n - i - 1) for i in range(n)]
        return weights, sum(weights)
