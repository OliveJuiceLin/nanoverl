"""Algorithm plugin interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol

from nanoverl.config import SamplingConfig, TrainerConfig
from nanoverl.core.batch import RLBatch
from nanoverl.distributed import TorchDistributedRuntime
from nanoverl.reward import RewardManager
from nanoverl.rollout import RolloutEngine, SamplingParams
from nanoverl.rollout.sync import PolicySyncResult


def sampling_to_params(config: SamplingConfig) -> SamplingParams:
    return SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        do_sample=config.do_sample,
        n=config.n,
    )


class OptimizerMetricRecorder(Protocol):
    def __call__(self, prefix: str, step_metrics: list[Dict[str, float]]) -> None:
        ...


def _noop_policy_sync(reason: str) -> PolicySyncResult:
    return PolicySyncResult(reason=reason, count=0, seconds=0.0, metrics={})


@dataclass
class AlgorithmStepContext:
    config: TrainerConfig
    policy_worker: Any
    reference_worker: Any | None
    value_worker: Any | None
    rollout_engine: RolloutEngine
    reward_manager: RewardManager
    runtime: TorchDistributedRuntime
    global_step: int
    train_epoch: int
    prepare_rollout_batch: Callable[[RLBatch, SamplingParams], RLBatch]
    balance_rollout_batch: Callable[[RLBatch], RLBatch]
    record_optimizer_step_metrics: OptimizerMetricRecorder
    dump_train_preview: Callable[[RLBatch, int], None]
    sync_rollout_policy: Callable[[str], PolicySyncResult] = _noop_policy_sync


class RLAlgorithm:
    name = "base"

    def uses_critic(self, config: TrainerConfig) -> bool:
        return config.critic.enable

    def run_step(self, batch: RLBatch, context: AlgorithmStepContext) -> Dict[str, float]:
        raise NotImplementedError


__all__ = ["AlgorithmStepContext", "RLAlgorithm", "sampling_to_params"]
