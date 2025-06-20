import random
import numpy as np
import pytest

from analysis.ARIMA_analyzer import ARIMAAnalyzer
from analysis.exponential_analyzer import ExponentialRegressionAnalyzer
from app.domain.Intent import Intent, ThresholdPolicy
from app.domain.resourceSpec import ResourceSpec
from app.domain.ueSessionInfo import UeSessionInfo
from app.core.scaling_decision_service import ScalingDecisionService
from core.scaling_policy_engine import ScalingPolicyEngine

# --------------------------------------------------------------------------------
# Config dla produkcyjnego scenariusza
# --------------------------------------------------------------------------------
PROBE_INTERVAL_SEC = 5
APPLY_DELAY_PROBES = 6
TEST_DURATION_MIN = 30

CONFIG = {
    "slope_threshold": 0.5,
    "dy_threshold": 1.0,
    "thresholds": [
        {"min_sessions": 0, "max_sessions": 50, "resources": {"cpu": "100m", "memory": "128Mi"}},
        {"min_sessions": 51, "max_sessions": 150, "resources": {"cpu": "200m", "memory": "256Mi"}},
        {"min_sessions": 151, "max_sessions": 300, "resources": {"cpu": "300m", "memory": "384Mi"}},
        {"min_sessions": 301, "max_sessions": 500, "resources": {"cpu": "400m", "memory": "512Mi"}},
        {"min_sessions": 501, "max_sessions": 750, "resources": {"cpu": "600m", "memory": "768Mi"}},
        {"min_sessions": 751, "max_sessions": 1000, "resources": {"cpu": "800m", "memory": "1024Mi"}},
        {"min_sessions": 1001, "max_sessions": 99999, "resources": {"cpu": "1000m", "memory": "1280Mi"}},
    ]
}


class MockCluster:
    def __init__(self, apply_delay: int = APPLY_DELAY_PROBES):
        self.pending_cpu = None
        self.current_cpu = None
        self._counter = 0
        self.delay = apply_delay

    def tick(self):
        if self.pending_cpu is not None:
            self._counter += 1
            if self._counter >= self.delay:
                self.current_cpu = self.pending_cpu
                self.pending_cpu = None
                self._counter = 0

    def apply_cpu(self, cpu: str):
        if self.pending_cpu or self.current_cpu == cpu:
            return
        self.pending_cpu = cpu


def build_intent(cfg: dict) -> Intent:
    return Intent(
        dy_threshold=cfg["dy_threshold"],
        slope_threshold=cfg["slope_threshold"],
        thresholds=[
            ThresholdPolicy(
                min_sessions=t["min_sessions"],
                max_sessions=t["max_sessions"],
                resources=ResourceSpec(**t["resources"]),
            ) for t in cfg["thresholds"]
        ],
        cooldown_sec=10
    )


def rule_for(sess_count: int, intent: Intent):
    return next(
        r for r in intent.thresholds
        if r.min_sessions <= sess_count <= r.max_sessions
    )


def generate_realistic_traffic(duration_min: int) -> list[int]:
    total = duration_min * 60 // PROBE_INTERVAL_SEC
    base = 200 + 150 * np.sin(np.linspace(0, 6 * np.pi, total))
    noise = np.random.normal(0, 25, total)
    burst = np.zeros_like(base)
    burst[::100] = np.random.randint(100, 200, size=len(burst[::100]))
    return np.clip(base + noise + burst, 0, 1200).astype(int).tolist()


def backtest(service: ScalingDecisionService, traffic: list[int], intent: Intent):
    cluster = MockCluster()
    fit = under = over = 0

    for count in traffic:
        cpu = service.decide(UeSessionInfo(0, count))
        if cpu:
            cluster.apply_cpu(cpu)
        cluster.tick()

        expected = rule_for(count, intent).resources.cpu
        effective = cluster.current_cpu

        if effective == expected:
            fit += 1
        elif effective is None or expected > effective:
            under += 1
        else:
            over += 1

    total = len(traffic)
    return fit / total, under / total, over / total


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("max_hist", [30, 50, 100])
def test_realistic_predictive_vs_reactive(alpha: float, max_hist: int):
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)

    intent = build_intent(CONFIG)
    traffic = generate_realistic_traffic(TEST_DURATION_MIN)

    class Reactive(ScalingDecisionService):
        def decide(self, s):
            self.engine.add_sample(s)
            r = self.engine.evaluate()
            for rule in self.intent.thresholds:
                if rule.min_sessions <= r.current_sessions <= rule.max_sessions:
                    return rule.resources.cpu
            return self.intent.thresholds[-1].resources.cpu

    pred = ScalingDecisionService(intent)
    reac = Reactive(intent)
    reac.engine = ScalingPolicyEngine(
        analyzer=ARIMAAnalyzer(period=12,
                               minimal_reaction_time=5),
        max_history=max_hist
    )

    fit_p, under_p, over_p = backtest(pred, traffic, intent)
    fit_r, under_r, over_r = backtest(reac, traffic, intent)

    print(f"\nALPHA={alpha}, MAX_HIST={max_hist}")
    print(f"PREDICTIVE: fit={fit_p:.2%}, under={under_p:.2%}, over={over_p:.2%}")
    print(f"REACTIVE : fit={fit_r:.2%}, under={under_r:.2%}, over={over_r:.2%}")

    assert under_p < under_r + 0.01, "Predictive under-provision should be lower or equal"
    assert fit_p >= fit_r - 0.01, "Predictive fit should be better or equal"
