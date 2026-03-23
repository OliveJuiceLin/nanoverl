"""Rollout engine interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from nanoverl.core.batch import RLBatch


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    do_sample: bool = True
    n: int = 1


class RolloutEngine:
    """Base rollout engine."""

    def generate(self, batch: RLBatch, sampling: SamplingParams) -> RLBatch:
        raise NotImplementedError

    def sync_policy(self, policy_state: Dict[str, Any]) -> None:
        del policy_state

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        del state


__all__ = ["RolloutEngine", "SamplingParams"]
