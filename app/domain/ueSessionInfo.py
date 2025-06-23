from dataclasses import dataclass
from datetime import time


@dataclass
class UeSessionInfo:
    timestamp: float = time.t
    session_count: int
