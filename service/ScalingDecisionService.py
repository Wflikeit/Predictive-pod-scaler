from typing import Optional
from domain.Intent import Intent
from domain.ueSessionInfo import UeSessionInfo
from service.ScalingPolicyEngine import ScalingPolicyEngine

class ScalingDecisionService:
    def __init__(self, intent: Intent):
        self.intent = intent
        self.engine = ScalingPolicyEngine(intent)

    def decide(self, history: list[UeSessionInfo]) -> Optional[str]:
        """
        Makes a scaling decision based on the history of UE session metrics.

        Args:
            history (list[UeSessionInfo]): List of recent UE session metric samples.

        Returns:
            Optional[str]: Target CPU limit (e.g., "200m") if scaling is required,
                           otherwise None.
        """
        return self.engine.evaluate(history)
