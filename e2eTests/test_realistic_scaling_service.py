import random
from typing import List, Optional

import numpy as np

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


def generate_linear_noise_traffic(duration_min: int) -> List[int]:
    x = np.arange(0, duration_min * 60, PROBE_INTERVAL_SEC)
    trend = 0.5 * x / 60  # ok. 0.5 użytkownika na sekundę
    noise = np.random.normal(0, 2, size=len(x))
    return np.clip((trend + noise).astype(int), 0, None).tolist()


def generate_sine_traffic(duration_min: int) -> List[int]:
    x = np.arange(0, duration_min * 60, PROBE_INTERVAL_SEC)
    base = 30 + 10 * np.sin(2 * np.pi * x / 300)
    noise = np.random.normal(0, 2, size=len(x))
    return np.clip((base + noise).astype(int), 0, None).tolist()


def generate_spikey_traffic(duration_min: int) -> List[int]:
    base = generate_linear_noise_traffic(duration_min)
    for i in range(0, len(base), 50):
        base[i] += random.randint(15, 30)
    return base


def generate_broken_traffic(duration_min: int) -> List[int]:
    x = np.arange(0, duration_min * 60, PROBE_INTERVAL_SEC)
    y = np.random.normal(30, 10, size=len(x)).astype(int)
    y[::40] = 0  # dropout
    y[::60] += 50  # spike
    return np.clip(y, 0, None).tolist()


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


# --- PARAMETRIZED TEST ---
def _run_test_for_generator(generator_func):
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)

    intent = build_intent_from_config(CONFIG)
    traffic = generator_func(TEST_DURATION_MIN)

    pred = ScalingDecisionService(intent)
    fit_p, under_p, over_p = backtest(pred, traffic, intent)

    class Reactive(ScalingDecisionService):
        def decide(self, s):
            self.engine.add_sample(s)
            r = self.engine.evaluate()
            for rule in self.intent.thresholds:
                if rule.min_sessions <= r.current_sessions <= rule.max_sessions:
                    return rule.resources.cpu
            return self.intent.thresholds[-1].resources.cpu

    reac = Reactive(intent)
    fit_r, under_r, over_r = backtest(reac, traffic, intent)

    print(f"\n{generator_func.__name__}")
    print(f"PREDICTIVE: fit={fit_p:.2%}, under={under_p:.2%}, over={over_p:.2%}")
    print(f"REACTIVE : fit={fit_r:.2%}, under={under_r:.2%}, over={over_r:.2%}")

    assert under_p < under_r, f"Prediction did not worse under-provision: {under_p:.2%} vs {under_r:.2%}"
    assert fit_p >= fit_r, f"Prediction worsen CPU-fit: {fit_p:.2%} vs {fit_r:.2%}"
    assert under_p <= under_r + 0.01

def test_linear_noise():
    _run_test_for_generator(generate_linear_noise_traffic)


def test_sine():
    _run_test_for_generator(generate_sine_traffic)


def test_spikey():
    _run_test_for_generator(generate_spikey_traffic)


def test_broken():
    _run_test_for_generator(generate_broken_traffic)


def test_default():
    _run_test_for_generator(generate_traffic)
