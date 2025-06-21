# app/core/scaling_decision_service.py
import time
from typing import Optional

from app.analysis.ARIMA_analyzer import ARIMAAnalyzer
from app.domain.Intent import Intent
from app.domain.trendResult import TrendResult
from app.domain.ueSessionInfo import UeSessionInfo
from app.core.scaling_policy_engine import ScalingPolicyEngine


class ScalingDecisionService:
    """
    Algorytm:
    1.  Dopóki nie zbierzemy 2×period realnych próbek — tryb reaktywny.
    2.  Co 'period' próbek retrenujemy ARIMA na *ostatnich* 120 próbkach
        (rolling-window, bez syntetyków).
    3.  Jeżeli górny 80-percentyl prognozy przekracza (max_sessions-1)
        lub slope > slope_threshold ⇒ skala w górę.
    4.  Jeżeli dolny 50-percentyl jest poniżej (min_sessions+1) ⇒ skala w dół
        po `underscale_delay_sec`.
    5.  Cool-down = Intent.cooldown_sec (w testach 5 s).
    """
    def __init__(self, intent: Intent, underscale_delay_sec: int = 10) -> None:
        self.intent = intent

        self._period = 20
        self._arima = ARIMAAnalyzer(period=self._period, minimal_reaction_time=5)

        # zapamiętujemy tylko 120 ostatnich próbek
        self.engine = ScalingPolicyEngine(
            analyzer=self._arima,
            max_history=120,
            retrain_interval=self._period,     # pełny retrain co 20 próbek
        )

        self._current_cpu: Optional[str] = None
        self._last_scale_ts: float = 0.0
        self._last_underscale_ts: Optional[float] = None
        self._underscale_delay_sec = underscale_delay_sec

    # ─────────────────────────── helpers ────────────────────────────
    def _cooldown(self) -> bool:
        return (time.time() - self._last_scale_ts) < self.intent.cooldown_sec

    def _rule_for(self, sessions: int):
        return next(r for r in self.intent.thresholds
                    if r.min_sessions <= sessions <= r.max_sessions)

    def _record_scale(self, cpu: str) -> None:
        if cpu == self._current_cpu:
            return                    # brak fizycznej zmiany
        self._current_cpu = cpu
        self._last_scale_ts = time.time()
        self._last_underscale_ts = None

    # ───────────────────────────── API ──────────────────────────────
    def decide(self, sample: UeSessionInfo) -> Optional[str]:
        self.engine.add_sample(sample)

        # ---------- 1. warm-up -----------
        period_needed = self._period * 2
        if len(self.engine.history) < period_needed:
            rule = self._rule_for(sample.session_count)
            self._record_scale(rule.resources.cpu)
            return self._current_cpu

        # ---------- 2. prognoza -----------
        horizon = 1.0                        # pełny period naprzód
        trend: TrendResult = self.engine.analyzer.evaluate(
            list(self.engine.history),
            part_of_period=horizon,
        )

        rule_now = self._rule_for(trend.current_sessions)
        cpu_target = rule_now.resources.cpu

        # ---------- 3. wymuszone skalowanie, gdy “na krawędzi” ----------
        if sample.session_count >= rule_now.max_sessions - 1:
            self._record_scale(cpu_target)
            return self._current_cpu

        # ---------- 4. over-scale ----------
        # --- overscale (w górę) ---
        need_up = (
                trend.current_sessions >= rule_now.max_sessions - 1 or
                trend.delta >= self.intent.dy_threshold or
                trend.slope >= self.intent.slope_threshold
        )
        if need_up:
            self._record_scale(cpu_target)  # ignorujemy cooldown
            return self._current_cpu

        # ---------- 5. under-scale ----------
        if trend.delta <= -self.intent.dy_threshold:
            if self._last_underscale_ts is None:
                self._last_underscale_ts = time.time()
            elif (time.time() - self._last_underscale_ts) >= self._underscale_delay_sec:
                self._record_scale(cpu_target)
                return self._current_cpu
        else:
            self._last_underscale_ts = None

        return None
