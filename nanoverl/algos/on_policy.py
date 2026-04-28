"""Built-in on-policy RL algorithms."""

from __future__ import annotations

import time
from typing import Dict

from nanoverl.algos.advantages import get_advantage_estimator
from nanoverl.algos.base import AlgorithmStepContext, RLAlgorithm, sampling_to_params
from nanoverl.algos.kl import apply_kl_penalty
from nanoverl.algos.registry import register_algorithm
from nanoverl.config import TrainerConfig
from nanoverl.core.batch import RLBatch
from nanoverl.logging.metrics import compute_data_metrics, compute_throughput_metrics, compute_timing_metrics


class OnPolicyAlgorithm(RLAlgorithm):
    advantage_estimator = "gae"
    uses_value_worker = True

    def uses_critic(self, config: TrainerConfig) -> bool:
        return config.critic.enable and self.uses_value_worker

    def _uses_critic(self, context: AlgorithmStepContext) -> bool:
        return context.value_worker is not None and self.uses_critic(context.config)

    def _extract_rewards(self, batch: RLBatch, context: AlgorithmStepContext) -> None:
        reward_result = context.reward_manager.compute(batch)
        batch.batch["token_level_scores"] = reward_result.token_level_scores
        if reward_result.extra:
            for key, values in reward_result.extra.items():
                batch.non_tensor[key] = values

    def _shape_rewards(self, batch: RLBatch, context: AlgorithmStepContext) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if context.config.algorithm.use_kl_in_reward and context.reference_worker is not None:
            shaped_rewards, mean_kl = apply_kl_penalty(
                token_level_scores=batch.batch["token_level_scores"],
                old_log_probs=batch.batch["old_log_probs"],
                ref_log_probs=batch.batch["ref_log_probs"],
                response_mask=batch.batch["response_mask"],
                beta=context.config.algorithm.kl_coef,
                mode=context.config.algorithm.kl_penalty,
            )
            batch.batch["token_level_rewards"] = shaped_rewards
            metrics["reward/kl_penalty"] = mean_kl
        else:
            batch.batch["token_level_rewards"] = [list(row) for row in batch.batch["token_level_scores"]]
        return metrics

    def _compute_advantages(self, batch: RLBatch, context: AlgorithmStepContext) -> None:
        advantage_fn = get_advantage_estimator(self.advantage_estimator)
        advantages, returns = advantage_fn(batch, context.config.algorithm)
        batch.batch["advantages"] = advantages
        batch.batch["returns"] = returns

    def run_step(self, batch: RLBatch, context: AlgorithmStepContext) -> Dict[str, float]:
        step_started = time.time()
        metrics: Dict[str, float] = {}
        timing: Dict[str, float] = {}

        rollout_params = sampling_to_params(context.config.rollout.train)
        rollout_batch = context.prepare_rollout_batch(batch, rollout_params)

        t0 = time.time()
        rollout_batch = context.rollout_engine.generate(rollout_batch, rollout_params)
        timing["rollout"] = time.time() - t0
        rollout_batch = context.balance_rollout_batch(rollout_batch)

        t0 = time.time()
        self._extract_rewards(rollout_batch, context)
        timing["reward_extract"] = time.time() - t0

        t0 = time.time()
        old_log_probs = context.policy_worker.compute_log_probs(rollout_batch)
        rollout_batch.batch["old_log_probs"] = old_log_probs.log_probs
        metrics.update({"actor/%s" % key: value for key, value in old_log_probs.metrics.items()})
        timing["policy_eval"] = time.time() - t0

        if context.reference_worker is not None:
            t0 = time.time()
            ref_log_probs = context.reference_worker.compute_log_probs(rollout_batch)
            rollout_batch.batch["ref_log_probs"] = ref_log_probs.log_probs
            metrics.update({"ref/%s" % key: value for key, value in ref_log_probs.metrics.items()})
            timing["reference_eval"] = time.time() - t0

        if self._uses_critic(context):
            t0 = time.time()
            values = context.value_worker.compute_values(rollout_batch)
            rollout_batch.batch["values"] = values.values
            metrics.update({"critic/%s" % key: value for key, value in values.metrics.items()})
            timing["value_eval"] = time.time() - t0
        else:
            rollout_batch.batch["values"] = [[0.0 for _ in row] for row in rollout_batch.batch["response_mask"]]

        t0 = time.time()
        metrics.update(self._shape_rewards(rollout_batch, context))
        timing["reward_shape"] = time.time() - t0

        t0 = time.time()
        self._compute_advantages(rollout_batch, context)
        timing["advantage"] = time.time() - t0

        if self._uses_critic(context):
            t0 = time.time()
            critic_update = context.value_worker.update(rollout_batch)
            metrics.update({"critic/%s" % key: value for key, value in critic_update.metrics.items()})
            context.record_optimizer_step_metrics("critic", critic_update.step_metrics)
            timing["critic_update"] = time.time() - t0

        if context.global_step >= context.config.trainer.critic_warmup:
            t0 = time.time()
            actor_update = context.policy_worker.update(rollout_batch)
            metrics.update({"actor/%s" % key: value for key, value in actor_update.metrics.items()})
            context.record_optimizer_step_metrics("actor", actor_update.step_metrics)
            context.rollout_engine.sync_policy(context.policy_worker.state_dict())
            timing["actor_update"] = time.time() - t0

        timing["step"] = time.time() - step_started
        metrics.update(
            compute_data_metrics(
                rollout_batch,
                use_critic=self._uses_critic(context),
                response_length_limit=context.config.rollout.response_length,
            )
        )
        metrics.update(compute_timing_metrics(timing))
        metrics.update(
            compute_throughput_metrics(
                rollout_batch,
                timing["step"],
                world_size=context.runtime.world_size,
            )
        )
        metrics["training/global_step"] = float(context.global_step + 1)
        metrics["training/epoch"] = float(context.train_epoch)
        context.dump_train_preview(rollout_batch, context.global_step + 1)
        return metrics


@register_algorithm("ppo")
class PPOAlgorithm(OnPolicyAlgorithm):
    advantage_estimator = "gae"


@register_algorithm("grpo")
class GRPOAlgorithm(OnPolicyAlgorithm):
    advantage_estimator = "grpo"
    uses_value_worker = False


@register_algorithm("rloo")
class RLOOAlgorithm(OnPolicyAlgorithm):
    advantage_estimator = "rloo"
    uses_value_worker = False


__all__ = ["GRPOAlgorithm", "OnPolicyAlgorithm", "PPOAlgorithm", "RLOOAlgorithm"]
