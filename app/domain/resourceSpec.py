from dataclasses import dataclass


@dataclass
class ResourceSpec:
    cpu: str
    memory: str
