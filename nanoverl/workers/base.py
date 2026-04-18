"""Explicit worker interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from nanoverl.core.batch import RLBatch


@dataclass
class LogProbResult:
    log_probs: List[List[float]]
    entropy: List[List[float]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValueResult:
    values: List[List[float]]
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class UpdateResult:
    metrics: Dict[str, float] = field(default_factory=dict)
    step_metrics: List[Dict[str, float]] = field(default_factory=list)


class PolicyWorker:
    def compute_log_probs(self, batch: RLBatch) -> LogProbResult:
        raise NotImplementedError

    def update(self, batch: RLBatch) -> UpdateResult:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        del state


class ReferenceWorker:
    def compute_log_probs(self, batch: RLBatch) -> LogProbResult:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        del state


class ValueWorker:
    def compute_values(self, batch: RLBatch) -> ValueResult:
        raise NotImplementedError

    def update(self, batch: RLBatch) -> UpdateResult:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        del state


__all__ = [
    "LogProbResult",
    "PolicyWorker",
    "ReferenceWorker",
    "UpdateResult",
    "ValueResult",
    "ValueWorker",
]
