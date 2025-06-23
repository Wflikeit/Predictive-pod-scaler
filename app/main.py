import time
from app.core.scaling_decision_service import ScalingDecisionService
from app.domain.intent_loader import load_intent
from app.infra.kubernetes_client import KubernetesClient
from app.infra.prometheus_client import PrometheusClient

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def main():
    # --- Infrastructure ---
    scaler = KubernetesClient()
    prometheus = PrometheusClient(
        base_url="http://192.168.1.20:9090",
        metric="amf_session",
        namespace="default",
        service="open5gs-amf-metrics",
    )

    # --- Load Intent ---
    intent = load_intent("intent.json")

    # --- Initialize service ---
    decision_service = ScalingDecisionService(
        intent=intent,
        metrics_client=prometheus,
        scaler=scaler
    )

    # --- Main loop ---
    INTERVAL_SEC = 15

    print("[scaler] Starting scaling loop...")
    try:
        while True:
            decision_service.apply_scaling_if_needed()
            time.sleep(INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\n[scaler] Exiting on Ctrl+C")


if __name__ == "__main__":
    main()
