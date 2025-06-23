import subprocess
import time
import logging
import numpy as np

# ========================
# KONFIGURACJA
# ========================
DURATION_MINUTES = 40            # Czas trwania eksperymentu
PERIOD_MINUTES = 10              # Okres sinusoidy
PROBE_INTERVAL_SEC = 35          # Odstęp między próbami (Prometheus scrape + bufor)
MAX_SESSIONS = 50                # Maksymalna liczba sesji (count)
DEVIATION = 0.25                 # Odchylenie losowe (np. ±25%)

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Podstawowa komenda Helm
HELM_BASE_COMMAND = [
    "helm", "upgrade", "ueransim-ues-additional", "oci://registry-1.docker.io/gradiant/ueransim-ues",
    "--set", "gnb.hostname=ueransim-gnb",
    "--set", 'initialMSISDN="0000000005"'
]

# ========================
# GENERATOR RUCHU
# ========================
def generate_realistic_traffic(duration_min, period_min=1, deviation=0.2, max_session=50):
    steps = int(duration_min * 60 / PROBE_INTERVAL_SEC)
    total_periods = duration_min / period_min

    # Sinusoida 0-1 (pełne cykle)
    base = (np.sin(np.linspace(0, 2 * np.pi * total_periods, steps)) + 1) / 2

    # Dodaj losowy jitter ±deviation
    noise = (np.random.rand(steps) - 0.5) * 2 * deviation
    signal = np.clip((base * (1 + noise)) * max_session, 0, max_session)

    return np.round(signal).astype(int).tolist()

# ========================
# WYKONANIE SYMULACJI
# ========================
def run_simulation(session_counts):
    for i, count in enumerate(session_counts):
        cmd = HELM_BASE_COMMAND + ["--set", f"count={count}"]
        logging.info(f"[{i + 1}/{len(session_counts)}] Helm upgrade with count={count}")

        try:
            subprocess.run(cmd, check=True)
            logging.info("✔️  Update successful")
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Helm command failed: {e}")

        if i < len(session_counts) - 1:
            time.sleep(PROBE_INTERVAL_SEC)

# ========================
# ENTRYPOINT
# ========================
if __name__ == "__main__":
    logging.info("🔄 Generating realistic traffic pattern...")
    pattern = generate_realistic_traffic(
        duration_min=DURATION_MINUTES,
        period_min=PERIOD_MINUTES,
        deviation=DEVIATION,
        max_session=MAX_SESSIONS
    )
    logging.info(f"✅ Generated {len(pattern)} steps of session counts")
    run_simulation(pattern)
