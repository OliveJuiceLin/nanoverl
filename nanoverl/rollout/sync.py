"""Policy-to-rollout weight synchronization helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PolicySyncResult:
    reason: str
    count: int
    seconds: float
    metrics: Dict[str, float]


class PolicySyncer:
    """Single owner of actor -> rollout policy synchronization."""

    def __init__(self) -> None:
        self.policy_sync_count = 0

    def sync(self, policy_worker: Any, rollout_engine: Any, reason: str) -> PolicySyncResult:
        started = time.time()
        policy_state = policy_worker.policy_state_dict()
        rollout_engine.sync_policy(policy_state)
        seconds = time.time() - started
        self.policy_sync_count += 1
        return PolicySyncResult(
            reason=reason,
            count=self.policy_sync_count,
            seconds=seconds,
            metrics={
                "rollout/policy_sync_count": float(self.policy_sync_count),
                "rollout/policy_sync_seconds": seconds,
            },
        )


__all__ = ["PolicySyncResult", "PolicySyncer"]
