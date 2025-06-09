from typing import Optional
from domain.Intent import Intent
from domain.ueSessionInfo import UeSessionInfo

class ScalingPolicyEngine:
    def __init__(self, intent: Intent):
        self.intent = intent

    def evaluate(self, history: list[UeSessionInfo]) -> Optional[str]:
        """
        Evaluates the current metric history against intent-defined policies and
        determines whether scaling is needed.

        Args:
            history (list[UeSessionInfo]): List of UE session samples over time.

        Returns:
            Optional[str]: CPU limit (e.g., "150m") if a scaling action should be taken,
                           otherwise None.
        """
        if not history:
            return None

        latest = history[-1].session_count

        for rule in self.intent.rules:
            if rule.min_sessions <= latest < rule.max_sessions:
                return rule.cpu_limit

        return None
