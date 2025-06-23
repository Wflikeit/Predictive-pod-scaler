from __future__ import annotations

import logging
import time
from typing import Any, Final

import requests

from app.domain.ueSessionInfo import UeSessionInfo

logger = logging.getLogger(__name__)


class PrometheusClient:
    _QUERY_PATH: Final[str] = "/api/v1/query"

    def __init__(
            self,
            base_url: str,
            metric: str,
            namespace: str,
            service: str,
            timeout: float = 2.0,
    ) -> None:
        self.base = base_url.rstrip("/")
        self.metric = metric
        self.ns = namespace
        self.svc = service
        self.timeout = timeout

    def fetch_sessions(self) -> UeSessionInfo:
        query = f'{self.metric}{{service="{self.svc}",namespace="{self.ns}"}}'
        params = {"query": query}

        try:
            r = requests.get(f"{self.base}{self._QUERY_PATH}", params=params, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError("Failed to query Prometheus") from exc

        data: dict[str, Any] = r.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Prometheus error: {data}")

        try:
            result = data["data"]["result"]
            if not result:
                raise RuntimeError("No session data returned by Prometheus")

            value_str = result[0]["value"][1]
            session_count = int(value_str)
            timestamp = time.time()
            logger.info(f"[{timestamp:.3f}] Fetched {session_count} sessions from Prometheus")

            return UeSessionInfo(session_count=session_count, timestamp=timestamp)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Invalid Prometheus response: {data}") from exc