# app/core/scaling_decision_service.py

import time
from typing import Optional, Union

from analysis.ARIMA_analyzer import ARIMAAnalyzer
from app.domain.Intent import Intent
from app.domain.trendResult import TrendResult
from app.domain.ueSessionInfo import UeSessionInfo
from app.core.scaling_policy_engine import ScalingPolicyEngine


class ScalingDecisionService:
    def __init__(
        self,
        intent: Intent,
        underscale_delay_sec: int = 120
    ):
        period = 12
        arima = ARIMAAnalyzer(period=period, minimal_reaction_time=5)

        # bufor 4× dłuższy od potrzebnego i retrain co pełny okres
        self.engine = ScalingPolicyEngine(
            analyzer=arima,
            max_history=period * 4,  # 240 > 120  → historia się zmieści
            retrain_interval=period  # co 60 nowych próbek pełny retrain
        )

        self.intent = intent

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
        # 1) add new sample
        self.engine.add_sample(sample)

        # 2) warm-up: if we don't yet have enough real samples, fallback reactive
        period = self.engine.analyzer.period
        needed = period * 2
        hist_len = len(self.engine.history)

        if hist_len < needed:
            if hist_len == needed - 1:
                print("[DEBUG] Przy kolejnej próbce będzie train i ARIMA się odpali.")
            rule = self._get_matching_rule(sample.session_count)
            return rule.resources.cpu if rule else None

        if hist_len == needed:
            print("[DEBUG] Dokładnie potrzebna liczba próbek – trening ARIMA na realnych danych")
            self.engine.analyzer.train(list(self.engine.history))

        # tu już ARIMA działa
        print("[DEBUG] Uruchamiam evaluate() z ARIMA")
        raw = self.engine.evaluate(part_of_period=1)


        trend_min = trend_max = raw
        current_sessions = raw.current_sessions

        rule = self._get_matching_rule(current_sessions)
        target_cpu = rule.resources.cpu if rule else None

        if self._should_force_scale(rule, current_sessions):
            self._record_scale(target_cpu)
            return self._current_cpu

        if self._should_overscale(trend_max, self._cooldown_active()):
            self._record_scale(target_cpu)
            return self._current_cpu

        if self._should_underscale(trend_min):
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
