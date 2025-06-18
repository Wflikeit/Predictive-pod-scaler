import time
from typing import Optional

from app.analysis.exponential_analyzer import ExponentialRegressionAnalyzer
from app.domain.Intent import Intent
from app.domain.ueSessionInfo import UeSessionInfo
from app.core.scaling_policy_engine import ScalingPolicyEngine, TrendResult


class ScalingDecisionService:
    def __init__(self, intent: Intent, underscale_delay_sec: int = 120):
        self.intent = intent
        self.engine = ScalingPolicyEngine(ExponentialRegressionAnalyzer())
        self._current_cpu: Optional[str] = None
        self._last_scale_ts: float = 0.0
        self._last_underscale_ts: Optional[float] = None
        self._underscale_delay_sec = underscale_delay_sec

    def _cooldown_active(self) -> bool:
        return (time.time() - self._last_scale_ts) < self.intent.cooldown_sec

    def _get_matching_rule(self, session_count: int):
        return next(
            (r for r in self.intent.thresholds
             if r.min_sessions <= session_count <= r.max_sessions),
            None
        )

    def _should_force_scale(self, rule, current_sessions: int) -> bool:
        if rule is None:
            return True
        return (
                current_sessions > (rule.max_sessions + self.intent.margin_up) or
                current_sessions < (rule.min_sessions - self.intent.margin_down)
        )

    def _should_overscale(self, trend: TrendResult, in_cooldown: bool) -> bool:
        if in_cooldown:
            return False
        return (
                trend.delta >= self.intent.dy_threshold or
                trend.slope >= self.intent.slope_threshold
        )

    def _should_underscale(self, trend: TrendResult) -> bool:
        return trend.slope <= -self.intent.slope_threshold

    def decide(self, sample: UeSessionInfo) -> Optional[str]:
        self.engine.add_sample(sample)
        trend = self.engine.evaluate()

        rule = self._get_matching_rule(trend.current_sessions)
        target_cpu = rule.resources.cpu if rule else None

        if self._should_force_scale(rule, trend.current_sessions):
            self._record_scale(target_cpu)
            return self._current_cpu

        if self._should_overscale(trend, self._cooldown_active()):
            self._record_scale(target_cpu)
            return self._current_cpu

        if self._should_underscale(trend):
            if self._last_underscale_ts is None:
                self._last_underscale_ts = time.time()
            elif time.time() - self._last_underscale_ts >= self._underscale_delay_sec:
                self._record_scale(target_cpu)
                return self._current_cpu
        else:
            self._last_underscale_ts = None

        return None

    def _record_scale(self, cpu: Optional[str]):
        self._current_cpu = cpu
        self._last_scale_ts = time.time()
        self._last_underscale_ts = None
