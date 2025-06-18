from abc import ABC, abstractmethod
from typing import List
from app.domain.trendResult import TrendResult
from app.domain.ueSessionInfo import UeSessionInfo


class TrendAnalyzer(ABC):
    @abstractmethod
    def evaluate(self, history: List[UeSessionInfo]) -> TrendResult:
        pass
