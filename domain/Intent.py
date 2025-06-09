from dataclasses import dataclass


@dataclass
class Intent:
    thresholds: list[ThresholdPolicy]
    slope_threshold: float
    dy_threshold: int

""""
thresholds": [
  { "min_sessions": 0, "max_sessions": 4, "resources": {"cpu": "100m", "memory": "64Mi"} },
  { "min_sessions": 5, "max_sessions": 9, "resources": {"cpu": "200m", "memory": "128Mi"} }
]
"""