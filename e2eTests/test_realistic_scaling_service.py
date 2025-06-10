import random
from typing import List, Optional

import numpy as np
import pytest

from app.domain.Intent import Intent, ThresholdPolicy
from app.domain.resourceSpec import ResourceSpec
from app.domain.ueSessionInfo import UeSessionInfo
from app.service.scaling_decision_service import ScalingDecisionService

# -----------------------------------------------------------------------------
# Helpers ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

PROBE_INTERVAL_SEC = 5
APPLY_DELAY_PROBES = 6
TEST_DURATION_MIN = 20

SLOPE_THRESHOLD = 0.1
DY_THRESHOLD = 1.0
CONFIG = {
    "slope_threshold": SLOPE_THRESHOLD,
    "dy_threshold": DY_THRESHOLD,
    "thresholds": [
        {"min_sessions": 0, "max_sessions": 3, "resources": {"cpu": "100m", "memory": "64Mi"}},
        {"min_sessions": 4, "max_sessions": 7, "resources": {"cpu": "150m", "memory": "64Mi"}},
        {"min_sessions": 8, "max_sessions": 11, "resources": {"cpu": "200m", "memory": "64Mi"}},
        {"min_sessions": 12, "max_sessions": 99999, "resources": {"cpu": "250m", "memory": "64Mi"}},
    ]
}


class MockCluster:

    def __init__(self, apply_delay: int = APPLY_DELAY_PROBES):
        self.pending_cpu: Optional[str] = None
        self.current_cpu: Optional[str] = None
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


def build_intent_from_config(cfg: dict) -> Intent:
    return Intent(
        dy_threshold=cfg["dy_threshold"],
        slope_threshold=cfg["slope_threshold"],
        thresholds=[
            ThresholdPolicy(
                min_sessions=r["min_sessions"],
                max_sessions=r["max_sessions"],
                resources=ResourceSpec(**r["resources"]),
            )
            for r in cfg["thresholds"]
        ],
        cooldown_sec=0
    )


def rule_for(count: int, intent: Intent):
    return next(r for r in intent.thresholds if r.min_sessions <= count <= r.max_sessions)


# -----------------------------------------------------------------------------
# Realistic traffic generator --------------------------------------------------
# -----------------------------------------------------------------------------

def generate_traffic(duration_min: int) -> List[int]:
    total = duration_min * 60 // PROBE_INTERVAL_SEC
    x = np.arange(total)
    trend = 1 + 0.2 * x
    noise = np.random.normal(0, 0.5, total)
    series = np.clip(trend + noise, 0, None)
    half = total // 2
    plateau = np.full(total // 4, series.max())
    descend = np.clip(series.max() - 0.2 * np.arange(total // 4), 0, None)
    return np.concatenate([series[:half], plateau, descend]).astype(int).tolist()

def backtest(service: ScalingDecisionService, traffic: List[int], intent: Intent):
    cluster = MockCluster()
    under = over = fit = 0
    for sess in traffic:
        cpu = service.decide(UeSessionInfo(0, sess))
        if cpu:
            cluster.apply_cpu(cpu)
        cluster.tick()

        effective = cluster.current_cpu
        expected_cpu = rule_for(sess, intent).resources.cpu

        if effective == expected_cpu:
            fit += 1
        elif effective is None or expected_cpu > effective:
            under += 1
        else:
            over += 1

    total = len(traffic)
    return fit / total, under / total, over / total


# -----------------------------------------------------------------------------
# Main realistic test ----------------------------------------------------------
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42])
def test_scaling_with_propagation_delay(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    intent = build_intent_from_config(CONFIG)
    service_pred = ScalingDecisionService(intent)
    traffic = generate_traffic(TEST_DURATION_MIN)

    fit_p, under_p, over_p = backtest(service_pred, traffic, intent)

    class Reactive(ScalingDecisionService):
        def decide(self, s):
            self.engine.add_sample(s)
            r = self.engine.evaluate()
            for rule in self.intent.thresholds:
                if rule.min_sessions <= r.current_sessions <= rule.max_sessions:
                    return rule.resources.cpu
            return self.intent.thresholds[-1].resources.cpu

    service_reac = Reactive(intent)
    fit_r, under_r, over_r = backtest(service_reac, traffic, intent)

    print(f"\nPREDICTIVE: fit={fit_p:.2%}, under={under_p:.2%}, over={over_p:.2%}")
    print(f"REACTIVE : fit={fit_r:.2%}, under={under_r:.2%}, over={over_r:.2%}")

    assert under_p < under_r, f"Prediction did not worse under-provision: {under_p:.2%} vs {under_r:.2%}"
    assert fit_p >= fit_r, f"Prediciton worsen CPU-fit: {fit_p:.2%} vs {fit_r:.2%}"