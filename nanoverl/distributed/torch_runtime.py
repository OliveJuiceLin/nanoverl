"""Small torch.distributed helpers for the first FSDP training path."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class TorchDistributedRuntime:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @classmethod
    def from_environment(cls) -> "TorchDistributedRuntime":
        # This constructor is new in Phase 3 because the earlier local trainer had
        # no concept of per-rank ownership. The FSDP path needs one explicit place
        # to read rank topology without threading env lookups through the whole repo.
        return cls(
            rank=_env_int("RANK", 0),
            local_rank=_env_int("LOCAL_RANK", 0),
            world_size=_env_int("WORLD_SIZE", 1),
            backend=os.environ.get("TORCH_DISTRIBUTED_BACKEND"),
        )

    def barrier(self) -> None:
        """
        Function:
            - If distributed is not enabled, does nothing.
            - If distributed is enabled, blocks until all processes reach this point.
        """
        if not self.enabled:
            return
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def all_gather_objects(self, value: Any) -> List[Any]:
        # This method is new in Phase 3 because validation keeps a Python-level
        # summary path, so the first distributed version needs object gathering
        # without introducing a second tensor-only metrics pipeline.
        if not self.enabled:
            return [value]
        import torch.distributed as dist

        gathered_values: List[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_values, value)
        return gathered_values

    def broadcast_object(self, value: Any, src: int = 0) -> Any:
        if not self.enabled:
            return value
        import torch.distributed as dist

        payload = [value]
        dist.broadcast_object_list(payload, src=src)
        return payload[0]


__all__ = ["TorchDistributedRuntime"]
