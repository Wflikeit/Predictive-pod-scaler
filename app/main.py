from app.core.scaling_decision_service import ScalingDecisionService
from app.domain.intent_loader import load_intent
from app.infra.kubernetes_client import KubernetesClient
from app.infra.prometheus_client import PrometheusClient


def main():
    # --- Infrastructure adapters (np. bramki) ---
    scaler = KubernetesClient()

    print(scaler.test_connection())

    prometheus = PrometheusClient(
        base_url="http://192.168.1.20:9090",
        metric="amf_session",
        namespace="default",
        service="open5gs-amf-metrics",
    )
    print(prometheus.get_metric())

    # --- Application input (intent) ---
    intent = load_intent("intent.json")

    # --- Application service ---
    decision_service = ScalingDecisionService(
        intent=intent,
        metrics_client=prometheus,
        scaler=scaler
    )

    # --- Core orchestration ---
    # decision_service.apply_scaling_if_needed()


if __name__ == "__main__":
    main()
