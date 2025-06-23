from __future__ import annotations
from typing import Optional, Mapping, Any
from kubernetes import client, config
from kubernetes.client import V1Pod, V1PodList


def _load_config() -> None:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


class KubernetesClient:
    def __init__(self, namespace: str = "default") -> None:
        _load_config()
        self.v1 = client.CoreV1Api()
        self.ns = namespace

    def _pick_pod(self, pod_label: str) -> Optional[V1Pod]:
        pods = self.v1.list_namespaced_pod(
            namespace=self.ns,
            label_selector=pod_label,
        ).items
        return pods[0] if pods else None

    @staticmethod
    def _container_index(pod: V1Pod, container_name: str) -> int:
        for idx, c in enumerate(pod.spec.containers):
            if c.name == container_name:
                return idx
        raise ValueError(f"Container '{container_name}' not found in pod {pod.metadata.name}")

    def scale_cpu_if_needed(
        self,
        pod_label: str,
        container: str,
        cpu_target_millicores: int,
    ) -> bool:
        pod = self._pick_pod(pod_label)
        if not pod:
            raise RuntimeError(f"No pod for label '{pod_label}'")

        idx = self._container_index(pod, container)
        current = pod.spec.containers[idx].resources
        current_cpu = (
            current.limits.get("cpu") if current and current.limits else None
        )

        target_cpu_str = f"{cpu_target_millicores}m"
        if current_cpu == target_cpu_str:
            return False

        patch: Mapping[str, Any] = {
            "spec": {
                "containers": [
                    {
                        "name": container,
                        "resources": {"limits": {"cpu": target_cpu_str}},
                    }
                ]
            }
        }

        self.v1.patch_namespaced_pod_resize(
            name=pod.metadata.name,
            namespace=self.ns,
            body=patch,
        )

        print(f"[scaler] pod={pod.metadata.name} cpu {current_cpu} -> {target_cpu_str}")
        return True

    def test_connection(self) -> list[str]:
        try:
            pod_list: V1PodList = self.v1.list_pod_for_all_namespaces()
            names = [f"{pod.metadata.namespace}/{pod.metadata.name}" for pod in pod_list.items]
            print(f"[test] OK: znaleziono {len(names)} podów")
            return names
        except Exception as e:
            print(f"[test] Błąd połączenia z Kubernetes API: {e}")
            return []
