import json
from domain.resourceSpec import Intent, ThresholdPolicy, ResourceSpec

def load_intent(path: str) -> Intent:
    with open(path, 'r') as f:
        data = json.load(f)

    thresholds = [
        ThresholdPolicy(
            min_sessions=policy['min_sessions'],
            max_sessions=policy['max_sessions'],
            resources=ResourceSpec(
                cpu=policy['resources']['cpu'],
                memory=policy['resources']['memory']
            )
        )
        for policy in data['thresholds']
    ]

    return Intent(
        thresholds=thresholds,
        slope_threshold=data['slope_threshold'],
        dy_threshold=data['dy_threshold']
    )
