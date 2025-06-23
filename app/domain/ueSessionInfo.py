from dataclasses import dataclass
from datetime import time


@dataclass
class UeSessionInfo:
    timestamp: float
    session_count: int
