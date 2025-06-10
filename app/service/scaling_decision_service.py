import time
from typing import Optional

from app.domain.Intent import Intent
from app.domain.ueSessionInfo import UeSessionInfo
from app.service.scaling_policy_engine import ScalingPolicyEngine, TrendResult


class ScalingDecisionService:
    def __init__(self, intent: Intent):
        self.intent = intent
        self.engine = ScalingPolicyEngine()
        self._current_cpu: Optional[str] = None
        self._last_scale_ts: float = 0.0

    # -------------------------------------------------------------
    def _cooldown_active(self) -> bool:
        return (time.time() - self._last_scale_ts) < self.intent.cooldown_sec

    # -------------------------------------------------------------
    def decide(self, sample: UeSessionInfo) -> Optional[str]:
        self.engine.add_sample(sample)
        r: TrendResult = self.engine.evaluate()

        rule = next(
            (rul for rul in self.intent.thresholds
             if rul.min_sessions <= r.current_sessions <= rul.max_sessions),
            None
        )

        target_cpu = rule.resources.cpu if rule else None

        over = (
            rule and
            r.current_sessions > (rule.max_sessions + self.intent.margin_up)
        )
        under = (
            rule and
            r.current_sessions < (rule.min_sessions - self.intent.margin_down)
        )

        in_cooldown = self._cooldown_active()

        if over or under or (rule is None):
            self._current_cpu = target_cpu
            self._last_scale_ts = time.time()
            return self._current_cpu

        if not in_cooldown and (
            r.delta >= self.intent.dy_threshold or
            r.slope >= self.intent.slope_threshold or
            r.delta <= -self.intent.dy_threshold or
            r.slope <= -self.intent.slope_threshold
        ):
            self._current_cpu = target_cpu
            self._last_scale_ts = time.time()
            return self._current_cpu

        return None
