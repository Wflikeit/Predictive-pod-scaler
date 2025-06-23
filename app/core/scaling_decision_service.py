# app/core/scaling_decision_service.py
import logging
import time
from typing import Optional

from app.analysis.ARIMA_analyzer import ARIMAAnalyzer
from app.core.scaling_policy_engine import ScalingPolicyEngine
from app.domain.Intent import Intent
from app.domain.trendResult import TrendResult
from app.domain.ueSessionInfo import UeSessionInfo
from app.infra.kubernetes_client import KubernetesClient
from app.infra.prometheus_client import PrometheusClient

logger = logging.getLogger(__name__)


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

    def __init__(self, intent: Intent, metrics_client: PrometheusClient, scaler: KubernetesClient,
                 underscale_delay_sec: int = 10) -> None:
        self.metrics_client: PrometheusClient = metrics_client
        self.scaler: KubernetesClient = scaler
        self.intent: Intent = intent

        self._period: int = 20
        self.analyzer: ARIMAAnalyzer = ARIMAAnalyzer(period=self._period, minimal_reaction_time=5)

        self.engine = ScalingPolicyEngine(
            analyzer=self.analyzer,
            max_history=120,
            retrain_interval=self._period,  # pełny retrain co 20 próbek
        )

        self._current_cpu: Optional[str] = None
        self._last_scale_ts: float = 0.0
        self._last_underscale_ts: Optional[float] = None
        self._underscale_delay_sec: int = underscale_delay_sec

    # ─────────────────────────── helpers ────────────────────────────
    def _cooldown(self) -> bool:
        return (time.time() - self._last_scale_ts) < self.intent.cooldown_sec

    def _rule_for(self, sessions: int):
        return next(r for r in self.intent.thresholds
                    if r.min_sessions <= sessions <= r.max_sessions)

    def _record_scale(self, cpu: str) -> None:
        if cpu == self._current_cpu:
            return
        self._current_cpu = cpu
        self._last_scale_ts = time.time()
        self._last_underscale_ts = None

    def _determine_target_cpu(self, sample: UeSessionInfo) -> Optional[str]:
        self.engine.add_sample(sample)

        if len(self.engine.history) < 2 * self._period:
            # warm-up: tryb reaktywny
            rule = self._rule_for(sample.session_count)
            return rule.resources.cpu

        trend: TrendResult = self.analyzer.evaluate(
            list(self.engine.history),
            part_of_period=1.0
        )

        forecasted_sessions = int(trend.current_sessions + trend.delta)
        target_rule = self._rule_for(forecasted_sessions)
        cpu_target = target_rule.resources.cpu

        # Wymuszone skalowanie przy granicy
        if sample.session_count >= target_rule.max_sessions - 1:
            return cpu_target

        # Skalowanie w górę
        if (trend.current_sessions >= target_rule.max_sessions - 1 or
                trend.delta >= self.intent.dy_threshold or
                trend.slope >= self.intent.slope_threshold):
            return cpu_target

        # Skalowanie w dół (z opóźnieniem)
        if trend.delta <= -self.intent.dy_threshold:
            if self._last_underscale_ts is None:
                self._last_underscale_ts = time.time()
            elif time.time() - self._last_underscale_ts >= self._underscale_delay_sec:
                return cpu_target
        else:
            self._last_underscale_ts = None

        return None

    def apply_scaling_if_needed(self):
        sample = self.metrics_client.fetch_sessions()
        cpu_target = self._determine_target_cpu(sample)

        if cpu_target and not self._cooldown() and cpu_target != self._current_cpu:
            logger.info(
                f"[{sample.timestamp:.3f}] Scaling triggered: "
                f"session_count={sample.session_count}, "
                f"target_cpu={cpu_target}, previous_cpu={self._current_cpu}"
            )
            self._record_scale(cpu_target)
            self.scaler.scale_cpu_if_needed(
                pod_label="open5gs-amf-57c6c6c65b-gzcs8",
                container="open5gs-upf",
                cpu_target_millicores=int(cpu_target.replace("m", ""))
            )
