from dataclasses import dataclass


@dataclass
class UeSessionInfo:
    timestamp: float
    session_count: int
