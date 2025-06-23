from dataclasses import dataclass


@dataclass
class TrendResult:
    delta: float
    slope: float
    current_sessions: int
