"""Small runtime helpers for local execution and future Ray integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ResourcePool:
    name: str
    world_size: int = 1


@dataclass
class LocalWorkerGroup:
    role: str
    workers: List[Any]
    resource_pool: ResourcePool = field(default_factory=lambda: ResourcePool(name="local"))

    def first(self) -> Any:
        return self.workers[0]

    def call(self, method_name: str, *args, **kwargs) -> Any:
        if len(self.workers) == 1:
            return getattr(self.workers[0], method_name)(*args, **kwargs)
        return [getattr(worker, method_name)(*args, **kwargs) for worker in self.workers]


class RayWorkerGroup:
    """Optional Ray-backed worker group."""

    def __init__(self, role: str, worker_factory: Callable[[], Any], count: int = 1, address: Optional[str] = None):
        try:
            import ray
        except ImportError as exc:  # pragma: no cover - exercised when ray is installed
            raise RuntimeError("Ray support requires the optional 'train' dependencies.") from exc
        if not ray.is_initialized():
            init_kwargs: Dict[str, Any] = {}
            if address:
                init_kwargs["address"] = address
            ray.init(**init_kwargs)
        self.role = role
        self.resource_pool = ResourcePool(name="ray", world_size=count)
        self._workers = [ray.remote(worker_factory.__class__)]  # pragma: no cover
        raise NotImplementedError("Full Ray worker groups are intentionally deferred to the next integration pass.")


__all__ = ["LocalWorkerGroup", "RayWorkerGroup", "ResourcePool"]
