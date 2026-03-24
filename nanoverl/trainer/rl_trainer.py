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
        """
        Function:
            - 为批次中的每个样本分配一个唯一的 UID，确保在整个训练过程中可以跟踪和区分每个样本的轨迹和奖励。
        Example:
            batch = RLBatch(batch={"x": [1, 2, 3]})
            batch = self._ensure_uids(batch)
            # batch.non_tensor["uid"] = ["uuid1", "uuid2", "uuid3"]  (每个样本都有一个唯一的 UUID)
        """
        if "uid" in batch.non_tensor:
            return batch
        batch = batch.clone()
        batch.non_tensor["uid"] = [str(uuid.uuid4()) for _ in range(len(batch))]
        return batch

    def _prepare_rollout_batch(self, batch: RLBatch, sampling: SamplingParams) -> RLBatch:
        """
        Function:
            - 确保批次中的每个样本都有一个唯一的 UID，以便在后续的 rollout 和奖励计算过程中进行跟踪。
            - 根据采样参数重复批次中的每个样本，以便在 rollout 过程中生成多个响应版本。
            - 加入 rollout_index 来区分同一原始样本的不同采样版本

        """
        batch = self._ensure_uids(batch)
        repeated = batch.repeat(sampling.n, interleave=True)
        # 这里的 rollout_index 是为了在后续的奖励计算中区分同一原始样本（同一组）的不同采样版本，方便进行奖励分配和优势计算。
        # 例如index = 0, 1, 2, 3, 4 对应原始样本的不同版本，当 n=2 时，index % n 得[0, 1, 0, 1, 0]，表示每两个样本是一组采样版本。
        repeated.non_tensor["rollout_index"] = [index % sampling.n for index in range(len(repeated))]
        return repeated

    def _extract_rewards(self, batch: RLBatch) -> None:
        reward_result = self.reward_manager.compute(batch) # 一个 dataclass, 包含 token_level_scores 和 extra 两个字段
        batch.batch["token_level_scores"] = reward_result.token_level_scores
        if reward_result.extra:
            for key, values in reward_result.extra.items():
                batch.non_tensor[key] = values

    def _shape_rewards(self, batch: RLBatch) -> Dict[str, float]:
        """
        Function:
            - 根据配置中的 KL 惩罚设置，对原始的 token_level_scores 进行奖励 shaping，生成 token_level_rewards。
            - 如果启用了 KL 惩罚，并且 reference_worker 存在，则调用 apply_kl_penalty 函数来计算带有 KL 惩罚的奖励，并将其存储在 batch.batch["token_level_rewards"] 中。同时，将平均 KL 惩罚值添加到 metrics 中，以便进行日志记录和分析。
            - 如果没有启用 KL 惩罚或者 reference_worker 不存在，则直接将原始的 token_level_scores 作为 token_level_rewards 存储在 batch.batch["token_level_rewards"] 中。
        """
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
            # 直接将原始的 token_level_scores 作为 token_level_rewards 存储在 batch.batch["token_level_rewards"] 中。
            # 这里使用了一个列表推导式来创建一个新的列表，其中每一行都是原始 token_level_scores 中对应行的一个列表副本。
            # 这是为了确保 token_level_rewards 是一个独立的列表，而不是直接引用 token_level_scores，以避免在后续的奖励 shaping 或优势计算过程中对原始 scores 造成意外修改。
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
        payload = self.checkpoint_manager.load_latest() # load(checkpoint_dir) | none
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
        state = self.val_loader.state_dict() # 记录当前的读取位置
        self.val_loader.sampler.position = 0 # 把验证集的指针重置到开头，以确保每次验证都从头开始读取数据
        while True:
            batch = self.val_loader.next_batch()
            if batch is None:
                break
            batch = self._prepare_rollout_batch(batch, _sampling_to_params(self.config.rollout.validation))
            batch = self.rollout_engine.generate(batch, _sampling_to_params(self.config.rollout.validation))
            reward_result = self.reward_manager.compute(batch)
            scores.extend(sum(row) for row in reward_result.token_level_scores)
            data_sources.extend(batch.non_tensor.get("data_source", ["unknown"] * len(batch)))
        self.val_loader.load_state_dict(state) # 恢复验证集的读取位置
        metrics = summarize_validation(scores, data_sources)
        self.last_validation_metrics = metrics
        return metrics

    def train_step(self, batch: RLBatch) -> Dict[str, float]:
        # 注意，当前的 train_step 并不完全，例如并没有多次利用 rollout 数据进行更新，也没有实现一些算法特有的细节（如 PPO 的剪切机制）。
        # 这些都可以在后续的迭代中逐步完善。 

        step_started = time.time()
        metrics: Dict[str, float] = {}
        timing: Dict[str, float] = {}

        rollout_params = _sampling_to_params(self.config.rollout.train)
        rollout_batch = self._prepare_rollout_batch(batch, rollout_params)

        t0 = time.time()
        # 生成相应文本
        rollout_batch = self.rollout_engine.generate(rollout_batch, rollout_params)
        timing["rollout"] = time.time() - t0

        t0 = time.time()
        # 使用 reward_fn 计算句子的奖励
        self._extract_rewards(rollout_batch)
        timing["reward_extract"] = time.time() - t0

        # 每个worker执行各自的工作：
        # 计时、计算、记录数值、记录指标
        t0 = time.time()
        old_log_probs = self.policy_worker.compute_log_probs(rollout_batch)
        rollout_batch.batch["old_log_probs"] = old_log_probs.log_probs
        rollout_batch.batch["entropy"] = old_log_probs.entropy
        metrics.update({"actor/%s" % key: value for key, value in old_log_probs.metrics.items()})
        timing["policy_eval"] = time.time() - t0

        if self.reference_worker is not None:
            # 如果 reference_worker 存在，计算参考模型的 log_probs，并将其添加到 rollout_batch 中，以便后续的奖励 shaping 和优势计算使用。同时，将参考模型评估的相关指标添加到 metrics 中，以便进行日志记录和分析。
            t0 = time.time()
            ref_log_probs = self.reference_worker.compute_log_probs(rollout_batch)
            rollout_batch.batch["ref_log_probs"] = ref_log_probs.log_probs
            metrics.update({"ref/%s" % key: value for key, value in ref_log_probs.metrics.items()})
            timing["reference_eval"] = time.time() - t0

        if self.value_worker is not None:
            # 如果 value_worker 存在，计算状态值函数的估计值，并将其添加到 rollout_batch 中，以便后续的优势计算使用。同时，将价值函数评估的相关指标添加到 metrics 中，以便进行日志记录和分析。
            t0 = time.time()
            values = self.value_worker.compute_values(rollout_batch)
            rollout_batch.batch["values"] = values.values
            metrics.update({"critic/%s" % key: value for key, value in values.metrics.items()})
            timing["value_eval"] = time.time() - t0
        else:
            rollout_batch.batch["values"] = [[0.0 for _ in row] for row in rollout_batch.batch["response_mask"]]

        t0 = time.time()
        # 得出token_level_rewards字段
        reward_metrics = self._shape_rewards(rollout_batch)
        metrics.update(reward_metrics)
        timing["reward_shape"] = time.time() - t0

        t0 = time.time()
        # 根据 token_level_rewards 和 values 计算优势函数，并将优势值添加到 rollout_batch 中，以便后续的策略更新使用。同时，将优势计算的相关指标添加到 metrics 中，以便进行日志记录和分析。
        self._compute_advantages(rollout_batch)
        timing["advantage"] = time.time() - t0

        # ===================== 进入更新阶段 =====================
        # 如果 value_worker 存在，首先更新价值函数，并将相关指标添加到 metrics 中，以便进行日志记录和分析。
        if self.value_worker is not None:
            t0 = time.time()
            critic_update = self.value_worker.update(rollout_batch)
            metrics.update(critic_update.metrics)
            timing["critic_update"] = time.time() - t0
        # 如果全局步骤数已经超过了 critic_warmup 的设置，则更新策略，并将相关指标添加到 metrics 中，以便进行日志记录和分析。同时，将更新后的策略状态同步到 rollout_engine 中，以确保在后续的 rollout 过程中使用最新的策略进行生成。
        # 在 critic_warmup 阶段，训练过程只更新价值函数，而不更新策略。这是为了让价值函数先行学习一个相对稳定的状态值估计，从而在后续的策略更新中提供更准确的优势估计，帮助策略更有效地学习。
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

            # 根据配置中的 save_freq 设置，定期保存训练检查点。当全局步骤数达到指定的保存频率时，调用 save_checkpoint 方法将当前的训练状态保存到磁盘
            if self.config.trainer.save_freq > 0:
                if self.global_step % self.config.trainer.save_freq == 0 or self.global_step == max_steps:
                    self.save_checkpoint()

        if self.config.trainer.save_freq == 0:
            self.save_checkpoint()
        return self.last_validation_metrics


__all__ = ["RLTrainer", "build_trainer"]
