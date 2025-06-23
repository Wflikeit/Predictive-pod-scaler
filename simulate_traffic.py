import subprocess
import time
import logging

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Stałe
HELM_CMD_BASE = [
    "helm", "upgrade", "ueransim-ues-additional", "oci://registry-1.docker.io/gradiant/ueransim-ues",
    "--set", "gnb.hostname=ueransim-gnb",
    "--set", 'initialMSISDN="0000000005"'
]

WAIT_SECONDS = 35
ITERATIONS = 40
START_COUNT = 16

def run_simulation():
    for i in range(ITERATIONS):
        current_count = START_COUNT + i
        full_cmd = HELM_CMD_BASE + ["--set", f"count={current_count}"]

        logging.info(f"Running Helm upgrade with count={current_count}")
        try:
            subprocess.run(full_cmd, check=True)
            logging.info(f"Successfully applied count={current_count}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Helm command failed with count={current_count}: {e}")

        if i < ITERATIONS - 1:
            logging.info(f"Waiting {WAIT_SECONDS} seconds before next iteration...")
            time.sleep(WAIT_SECONDS)

if __name__ == "__main__":
    run_simulation()
