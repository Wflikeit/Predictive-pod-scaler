from __future__ import annotations

from typing import Any, Final

import requests


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

    def get_metric(self) -> int:
        query = f'{self.metric}{{service="{self.svc}",namespace="{self.ns}"}}'
        params = {"query": query}

        r = requests.get(
            f"{self.base}{self._QUERY_PATH}",
            params=params,
            timeout=self.timeout,
        )
        r.raise_for_status()

        data: dict[str, Any] = r.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Prometheus error: {data}")

        try:
            value_str = data["data"]["result"][0]["value"][1]
            return int(float(value_str))
        except (KeyError, IndexError, ValueError) as exc:
            raise RuntimeError(f"Invalid Prometheus response: {data}") from exc
