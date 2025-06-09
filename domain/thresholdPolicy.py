from dataclasses import dataclass

from domain.resourceSpec import ResourceSpec


@dataclass
class ThresholdPolicy:
    min_sessions: int
    max_sessions: int
    resources: ResourceSpec


