from dataclasses import dataclass
from typing import List

from app.domain.thresholdPolicy import ThresholdPolicy


@dataclass
class Intent:
    dy_threshold: float
    slope_threshold: float
    thresholds: List[ThresholdPolicy]
    margin_up: int = 3
    margin_down: int = 1
    cooldown_sec: int = 10


"""
thresholds": [
  { "min_sessions": 0, "max_sessions": 4, "resources": {"cpu": "100m", "memory": "64Mi"} },
  { "min_sessions": 5, "max_sessions": 9, "resources": {"cpu": "200m", "memory": "128Mi"} }
]
"""
