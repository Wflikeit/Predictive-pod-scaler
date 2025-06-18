from app.domain.intent_loader import load_intent
from app.core.scaling_decision_service import ScalingDecisionService


def main():
    intent = load_intent("../intent.json")
    decision_service = ScalingDecisionService(intent)


if __name__ == "__main__":
    main()
