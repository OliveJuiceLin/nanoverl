"""Synchronous RL trainer."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Dict, Optional

from nanoverl.algos.advantages import compute_gae_advantages, compute_grpo_advantages
from nanoverl.algos.kl import apply_kl_penalty
from nanoverl.checkpoint.manager import CheckpointManager
from nanoverl.config import SamplingConfig, TrainerConfig
from nanoverl.core.batch import RLBatch
from nanoverl.data.dataset import JsonDataset, StatefulDataLoader
from nanoverl.logging.metrics import compute_data_metrics, compute_throughput_metrics, compute_timing_metrics
from nanoverl.logging.trackers import TrackingManager
from nanoverl.reward import RewardManager, load_reward_function
from nanoverl.rollout import DebugRolloutEngine, SamplingParams
from nanoverl.trainer.validation import summarize_validation
from nanoverl.workers import create_policy_worker, create_reference_worker, create_value_worker


def _sampling_to_params(config: SamplingConfig) -> SamplingParams:
    return SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        do_sample=config.do_sample,
        n=config.n,
    )


def build_trainer(config: TrainerConfig) -> "RLTrainer":
    train_dataset = JsonDataset(config.data.train_path)
    val_dataset = JsonDataset(config.data.val_path) if config.data.val_path else None
    train_loader = StatefulDataLoader(
        train_dataset,
        batch_size=config.data.train_batch_size,
        prompt_key=config.data.prompt_key,
        shuffle=config.data.shuffle,
        seed=config.data.seed,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = StatefulDataLoader(
            val_dataset,
            batch_size=config.data.val_batch_size,
            prompt_key=config.data.prompt_key,
            shuffle=False,
            seed=config.data.seed,
            drop_last=False,
        )

    policy_worker = create_policy_worker(config.actor.backend, config.actor)
    reference_worker = create_reference_worker(config.reference.backend, config.reference) if config.reference.enable else None
    value_worker = create_value_worker(config.critic.backend, config.critic) if config.critic.enable else None

    if config.rollout.backend != "debug":
        raise ValueError("Only the built-in debug rollout engine is available in the current scaffold.")
    rollout_engine = DebugRolloutEngine(max_response_length=config.rollout.response_length)

    reward_fn = load_reward_function(config.reward.function_path, config.reward.function_name)
    reward_manager = RewardManager(reward_fn)
    tracker = TrackingManager(
        project_name=config.trainer.project_name,
        experiment_name=config.trainer.experiment_name,
        backends=config.trainer.loggers,
        config=config.to_dict(),
    )
    checkpoint_manager = CheckpointManager(config.trainer.default_local_dir)

    return RLTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        policy_worker=policy_worker,
        reference_worker=reference_worker,
        value_worker=value_worker,
        rollout_engine=rollout_engine,
        reward_manager=reward_manager,
        tracker=tracker,
        checkpoint_manager=checkpoint_manager,
    )


class RLTrainer:
    """Driver-owned synchronous RL trainer."""

    def __init__(
        self,
        config: TrainerConfig,
        train_loader: StatefulDataLoader,
        val_loader: Optional[StatefulDataLoader],
        policy_worker,
        reference_worker,
        value_worker,
        rollout_engine,
        reward_manager: RewardManager,
        tracker: TrackingManager,
        checkpoint_manager: CheckpointManager,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.policy_worker = policy_worker
        self.reference_worker = reference_worker
        self.value_worker = value_worker
        self.rollout_engine = rollout_engine
        self.reward_manager = reward_manager
        self.tracker = tracker
        self.checkpoint_manager = checkpoint_manager
        self.global_step = 0
        self.last_validation_metrics: Dict[str, float] = {}

    def close(self) -> None:
        self.tracker.close()

    def _ensure_uids(self, batch: RLBatch) -> RLBatch:
        if "uid" in batch.non_tensor:
            return batch
        batch = batch.clone()
        batch.non_tensor["uid"] = [str(uuid.uuid4()) for _ in range(len(batch))]
        return batch

    def _prepare_rollout_batch(self, batch: RLBatch, sampling: SamplingParams) -> RLBatch:
        batch = self._ensure_uids(batch)
        repeated = batch.repeat(sampling.n, interleave=True)
        repeated.non_tensor["rollout_index"] = [index % sampling.n for index in range(len(repeated))]
        return repeated

    def _extract_rewards(self, batch: RLBatch) -> None:
        reward_result = self.reward_manager.compute(batch)
        batch.batch["token_level_scores"] = reward_result.token_level_scores
        if reward_result.extra:
            for key, values in reward_result.extra.items():
                batch.non_tensor[key] = values

    def _shape_rewards(self, batch: RLBatch) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.config.algorithm.use_kl_in_reward and self.reference_worker is not None:
            shaped_rewards, mean_kl = apply_kl_penalty(
                token_level_scores=batch.batch["token_level_scores"],
                old_log_probs=batch.batch["old_log_probs"],
                ref_log_probs=batch.batch["ref_log_probs"],
                response_mask=batch.batch["response_mask"],
                beta=self.config.algorithm.kl_coef,
                mode=self.config.algorithm.kl_penalty,
            )
            batch.batch["token_level_rewards"] = shaped_rewards
            metrics["reward/kl_penalty"] = mean_kl
        else:
            batch.batch["token_level_rewards"] = [list(row) for row in batch.batch["token_level_scores"]]
        return metrics

    def _compute_advantages(self, batch: RLBatch) -> None:
        estimator = self.config.algorithm.advantage_estimator
        if estimator == "gae":
            advantages, returns = compute_gae_advantages(
                token_level_rewards=batch.batch["token_level_rewards"],
                values=batch.batch["values"],
                response_mask=batch.batch["response_mask"],
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
            )
        elif estimator == "grpo":
            advantages, returns = compute_grpo_advantages(
                token_level_rewards=batch.batch["token_level_rewards"],
                response_mask=batch.batch["response_mask"],
                group_ids=batch.non_tensor["uid"],
                normalize_by_std=self.config.algorithm.norm_adv_by_std_in_grpo,
            )
        else:
            raise ValueError("Unsupported advantage estimator: %s" % estimator)
        batch.batch["advantages"] = advantages
        batch.batch["returns"] = returns

    def _checkpoint_payload(self) -> Dict[str, object]:
        return {
            "global_step": self.global_step,
            "train_loader_state": self.train_loader.state_dict(),
            "policy_state": self.policy_worker.state_dict(),
            "reference_state": self.reference_worker.state_dict() if self.reference_worker is not None else None,
            "value_state": self.value_worker.state_dict() if self.value_worker is not None else None,
            "rollout_state": self.rollout_engine.state_dict(),
            "config": self.config.to_dict(),
        }

    def save_checkpoint(self) -> None:
        self.checkpoint_manager.save(self.global_step, self._checkpoint_payload())

    def load_checkpoint(self) -> bool:
        payload = self.checkpoint_manager.load_latest()
        if payload is None:
            return False
        self.global_step = int(payload["global_step"])
        self.train_loader.load_state_dict(payload["train_loader_state"])
        self.policy_worker.load_state_dict(payload["policy_state"])
        if self.reference_worker is not None and payload.get("reference_state") is not None:
            self.reference_worker.load_state_dict(payload["reference_state"])
        if self.value_worker is not None and payload.get("value_state") is not None:
            self.value_worker.load_state_dict(payload["value_state"])
        self.rollout_engine.load_state_dict(payload.get("rollout_state", {}))
        self.rollout_engine.sync_policy(self.policy_worker.state_dict())
        return True

    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        scores = []
        data_sources = []
        state = self.val_loader.state_dict()
        self.val_loader.sampler.position = 0
        while True:
            batch = self.val_loader.next_batch()
            if batch is None:
                break
            batch = self._prepare_rollout_batch(batch, _sampling_to_params(self.config.rollout.validation))
            batch = self.rollout_engine.generate(batch, _sampling_to_params(self.config.rollout.validation))
            reward_result = self.reward_manager.compute(batch)
            scores.extend(sum(row) for row in reward_result.token_level_scores)
            data_sources.extend(batch.non_tensor.get("data_source", ["unknown"] * len(batch)))
        self.val_loader.load_state_dict(state)
        metrics = summarize_validation(scores, data_sources)
        self.last_validation_metrics = metrics
        return metrics

    def train_step(self, batch: RLBatch) -> Dict[str, float]:
        step_started = time.time()
        metrics: Dict[str, float] = {}
        timing: Dict[str, float] = {}

        rollout_params = _sampling_to_params(self.config.rollout.train)
        rollout_batch = self._prepare_rollout_batch(batch, rollout_params)

        t0 = time.time()
        rollout_batch = self.rollout_engine.generate(rollout_batch, rollout_params)
        timing["rollout"] = time.time() - t0

        t0 = time.time()
        self._extract_rewards(rollout_batch)
        timing["reward_extract"] = time.time() - t0

        t0 = time.time()
        old_log_probs = self.policy_worker.compute_log_probs(rollout_batch)
        rollout_batch.batch["old_log_probs"] = old_log_probs.log_probs
        rollout_batch.batch["entropy"] = old_log_probs.entropy
        metrics.update({"actor/%s" % key: value for key, value in old_log_probs.metrics.items()})
        timing["policy_eval"] = time.time() - t0

        if self.reference_worker is not None:
            t0 = time.time()
            ref_log_probs = self.reference_worker.compute_log_probs(rollout_batch)
            rollout_batch.batch["ref_log_probs"] = ref_log_probs.log_probs
            metrics.update({"ref/%s" % key: value for key, value in ref_log_probs.metrics.items()})
            timing["reference_eval"] = time.time() - t0

        if self.value_worker is not None:
            t0 = time.time()
            values = self.value_worker.compute_values(rollout_batch)
            rollout_batch.batch["values"] = values.values
            metrics.update({"critic/%s" % key: value for key, value in values.metrics.items()})
            timing["value_eval"] = time.time() - t0
        else:
            rollout_batch.batch["values"] = [[0.0 for _ in row] for row in rollout_batch.batch["response_mask"]]

        t0 = time.time()
        reward_metrics = self._shape_rewards(rollout_batch)
        metrics.update(reward_metrics)
        timing["reward_shape"] = time.time() - t0

        t0 = time.time()
        self._compute_advantages(rollout_batch)
        timing["advantage"] = time.time() - t0

        if self.value_worker is not None:
            t0 = time.time()
            critic_update = self.value_worker.update(rollout_batch)
            metrics.update(critic_update.metrics)
            timing["critic_update"] = time.time() - t0

        if self.global_step >= self.config.trainer.critic_warmup:
            t0 = time.time()
            actor_update = self.policy_worker.update(rollout_batch)
            metrics.update(actor_update.metrics)
            self.rollout_engine.sync_policy(self.policy_worker.state_dict())
            timing["actor_update"] = time.time() - t0

        timing["step"] = time.time() - step_started
        metrics.update(compute_data_metrics(rollout_batch, use_critic=self.value_worker is not None))
        metrics.update(compute_timing_metrics(timing))
        metrics.update(compute_throughput_metrics(rollout_batch, timing["step"], world_size=1))
        metrics["training/global_step"] = float(self.global_step)
        metrics["training/epoch"] = float(self.train_loader.epoch)
        return metrics

    def fit(self) -> Dict[str, float]:
        self.load_checkpoint()
        self.rollout_engine.sync_policy(self.policy_worker.state_dict())

        if self.config.trainer.validate_before_train:
            val_metrics = self.validate()
            if val_metrics:
                self.tracker.log(val_metrics, step=self.global_step)
            if self.config.trainer.validate_only:
                return val_metrics

        max_steps = self.config.total_training_steps(len(self.train_loader))
        while self.global_step < max_steps:
            batch = self.train_loader.next_batch()
            if batch is None:
                self.train_loader.reset_for_new_epoch()
                batch = self.train_loader.next_batch()
                if batch is None:
                    break

            metrics = self.train_step(batch)
            self.global_step += 1

            if self.config.trainer.test_freq > 0 and self.val_loader is not None:
                if self.global_step % self.config.trainer.test_freq == 0 or self.global_step == max_steps:
                    metrics.update(self.validate())

            self.tracker.log(metrics, step=self.global_step)

            if self.config.trainer.save_freq > 0:
                if self.global_step % self.config.trainer.save_freq == 0 or self.global_step == max_steps:
                    self.save_checkpoint()

        if self.config.trainer.save_freq == 0:
            self.save_checkpoint()
        return self.last_validation_metrics


__all__ = ["RLTrainer", "build_trainer"]
