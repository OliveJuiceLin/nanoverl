"""Debug workers with deterministic behavior."""

from __future__ import annotations

from typing import Dict, List

from nanoverl.algos.kl import compute_kl_penalty
from nanoverl.algos.ppo import compute_value_loss, get_policy_loss_fn
from nanoverl.core.batch import RLBatch
from nanoverl.workers.base import LogProbResult, PolicyWorker, ReferenceWorker, UpdateResult, ValueResult, ValueWorker


def _constant_matrix_like(mask: List[List[int]], value: float) -> List[List[float]]:
    return [[value if keep else 0.0 for keep in row] for row in mask]


class DebugPolicyWorker(PolicyWorker):
    def __init__(self, config):
        self.config = config
        self.version = 0

    def compute_log_probs(self, batch: RLBatch) -> LogProbResult:
        log_probs = [list(row) for row in batch.batch["rollout_log_probs"]]
        entropy = _constant_matrix_like(batch.batch["response_mask"], 0.5)
        return LogProbResult(log_probs=log_probs, entropy=entropy, metrics={"policy_version": float(self.version)})

    def update(self, batch: RLBatch) -> UpdateResult:
        old_log_probs = batch.batch["old_log_probs"]
        shifted_log_probs = [
            [value + self.config.update_step_size if keep else value for value, keep in zip(row, mask_row)]
            for row, mask_row in zip(old_log_probs, batch.batch["response_mask"])
        ]
        policy_loss_fn = get_policy_loss_fn(self.config.policy_loss)
        loss, metrics = policy_loss_fn(
            old_log_probs=old_log_probs,
            log_probs=shifted_log_probs,
            advantages=batch.batch["advantages"],
            response_mask=batch.batch["response_mask"],
            cliprange=self.config.clip_ratio,
            cliprange_low=self.config.clip_ratio_low,
            cliprange_high=self.config.clip_ratio_high,
            clip_ratio_c=self.config.clip_ratio_c,
            loss_agg_mode=self.config.loss_agg_mode,
        )
        if self.config.use_kl_loss and "ref_log_probs" in batch.batch:
            kl_matrix = compute_kl_penalty(shifted_log_probs, batch.batch["ref_log_probs"], mode="low_var_kl")
            kl_mean = sum(sum(row) for row in kl_matrix) / max(
                sum(sum(mask_row) for mask_row in batch.batch["response_mask"]),
                1,
            )
            loss += self.config.kl_loss_coef * kl_mean
            metrics["actor_kl_loss"] = kl_mean
        self.version += 1
        metrics.update({"actor_loss": loss, "actor_version": float(self.version)})
        return UpdateResult(metrics=metrics)

    def state_dict(self) -> Dict[str, float]:
        return {"version": float(self.version)}

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.version = int(state.get("version", 0))


class DebugReferenceWorker(ReferenceWorker):
    def __init__(self, config):
        self.config = config

    def compute_log_probs(self, batch: RLBatch) -> LogProbResult:
        log_probs = [
            [value + self.config.fixed_kl_offset if keep else value for value, keep in zip(row, mask_row)]
            for row, mask_row in zip(batch.batch["rollout_log_probs"], batch.batch["response_mask"])
        ]
        return LogProbResult(log_probs=log_probs, metrics={"reference_enabled": 1.0})


class DebugValueWorker(ValueWorker):
    def __init__(self, config):
        self.config = config
        self.value_bias = 0.0

    def compute_values(self, batch: RLBatch) -> ValueResult:
        values = _constant_matrix_like(batch.batch["response_mask"], self.value_bias)
        return ValueResult(values=values, metrics={"value_bias": self.value_bias})

    def update(self, batch: RLBatch) -> UpdateResult:
        values = batch.batch["values"]
        loss, metrics = compute_value_loss(
            values=values,
            returns=batch.batch["returns"],
            response_mask=batch.batch["response_mask"],
            cliprange_value=self.config.cliprange_value,
            loss_agg_mode=self.config.loss_agg_mode,
        )
        valid_returns = []
        for return_row, mask_row in zip(batch.batch["returns"], batch.batch["response_mask"]):
            valid_returns.extend(value for value, keep in zip(return_row, mask_row) if keep)
        target_mean = sum(valid_returns) / max(len(valid_returns), 1)
        self.value_bias = (self.value_bias + target_mean) / 2.0
        metrics.update({"critic_loss": loss, "critic_value_bias": self.value_bias})
        return UpdateResult(metrics=metrics)

    def state_dict(self) -> Dict[str, float]:
        return {"value_bias": self.value_bias}

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.value_bias = float(state.get("value_bias", 0.0))


__all__ = ["DebugPolicyWorker", "DebugReferenceWorker", "DebugValueWorker"]
