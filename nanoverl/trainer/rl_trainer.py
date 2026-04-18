"""Synchronous RL trainer."""

from __future__ import annotations

import math
import time
import uuid
from typing import Dict, Optional

from nanoverl.algos.advantages import compute_gae_advantages, compute_grpo_advantages
from nanoverl.algos.kl import apply_kl_penalty
from nanoverl.checkpoint.manager import CheckpointManager
from nanoverl.config import SamplingConfig, TrainerConfig
from nanoverl.core.batch import RLBatch
from nanoverl.data.dataset import JsonDataset, StatefulDataLoader
from nanoverl.distributed import TorchDistributedRuntime
from nanoverl.logging.metrics import compute_data_metrics, compute_throughput_metrics, compute_timing_metrics
from nanoverl.logging.trackers import TrackingManager
from nanoverl.reward import RewardManager, load_reward_function
from nanoverl.rollout import DebugRolloutEngine, HFRolloutEngine, SamplingParams, VLLMRolloutEngine
from nanoverl.trainer.artifacts import ArtifactWriter, build_batch_preview_rows
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
    runtime: TorchDistributedRuntime = TorchDistributedRuntime.from_environment()
    train_batch_size = config.data.train_batch_size
    val_batch_size = config.data.val_batch_size

    if runtime.enabled: # 如果开启了分布式训练，进行以下检查和调整单机数据量
        # 确保全局批次大小（Global Batch Size）能够均匀地分配到每一个计算设备上
        if train_batch_size % runtime.world_size != 0:
            raise ValueError("data.train_batch_size must be divisible by WORLD_SIZE for distributed training.")
        if config.data.val_path and val_batch_size % runtime.world_size != 0:
            raise ValueError("data.val_batch_size must be divisible by WORLD_SIZE for distributed validation.")
        # 将变量从“全局总计”更新为“单机/单卡”的大小。后续的 DataLoader 将使用这个计算后的局部值。
        train_batch_size //= runtime.world_size
        val_batch_size //= runtime.world_size

    # ========== 构建数据加载器 ==========
    train_dataset = JsonDataset(config.data.train_path)
    val_dataset = JsonDataset(config.data.val_path) if config.data.val_path else None
    train_loader = StatefulDataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=config.data.shuffle,
        seed=config.data.seed,
        drop_last=True,
        rank=runtime.rank,
        world_size=runtime.world_size,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = StatefulDataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            seed=config.data.seed,
            drop_last=False,
            rank=runtime.rank,
            world_size=runtime.world_size,
        )
    # ========== 构建 worker 和 rollout engine ==========
    policy_worker = create_policy_worker(config.actor.backend, config.model, config.actor)
    reference_worker = (
        create_reference_worker(config.reference.backend, config.model, config.reference)
        if config.reference.enable
        else None
    )
    use_critic = config.critic.enable and config.algorithm.advantage_estimator != "grpo"
    value_worker = create_value_worker(config.critic.backend, config.model, config.critic) if use_critic else None

    if config.rollout.backend == "debug":
        rollout_engine = DebugRolloutEngine(max_response_length=config.rollout.response_length)
    elif config.rollout.backend == "hf":
        rollout_engine = HFRolloutEngine(config.model, config.data, config.rollout)
    elif config.rollout.backend == "vllm":
        rollout_engine = VLLMRolloutEngine(config.model, config.data, config.rollout)
    else:
        raise ValueError("Unknown rollout backend: %s" % config.rollout.backend)
    # ========== 构建奖励函数管理器、日志跟踪器和检查点管理器 ==========
    reward_fn = load_reward_function(config.reward.function_path, config.reward.function_name)
    reward_manager = RewardManager(reward_fn)
    tracker = TrackingManager(
        project_name=config.trainer.project_name,
        experiment_name=config.trainer.experiment_name,
        backends=config.trainer.loggers if runtime.is_main_process else (), # 只在主进程中启用日志记录，以避免重复记录和竞争条件
        config=config.to_dict(),
    )
    checkpoint_manager = CheckpointManager(config.trainer.default_local_dir)
    artifact_writer = None
    if runtime.is_main_process and (config.trainer.train_dump_freq > 0 or config.trainer.validation_dump_freq > 0):
        artifact_writer = ArtifactWriter(config.trainer.default_local_dir, config.trainer.experiment_name)

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
        artifact_writer=artifact_writer,
        runtime=runtime,
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
        artifact_writer: Optional[ArtifactWriter] = None,
        runtime: Optional[TorchDistributedRuntime] = None,
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
        self.artifact_writer = artifact_writer
        self.runtime = runtime or TorchDistributedRuntime()
        self.global_step = 0
        self.log_step = 0
        self.actor_optimizer_step = 0
        self.critic_optimizer_step = 0
        self.last_validation_metrics: Dict[str, float] = {}

    def close(self) -> None:
        self.tracker.close()

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        self.tracker.log(metrics, step=self.log_step)
        self.log_step += 1

    def _record_optimizer_step_metrics(
        self,
        prefix: str,
        step_metrics: list[Dict[str, float]],
    ) -> None:
        """
        处理强化学习中“全局步数”与“实际优化器更新步数”的记录问题
        Rollout 生成一次数据（即 1 个 Global Step）
        针对这一批数据进行多次组装（Mini-batches）和多次循环（Epochs），这意味着在一个 Global Step 内，优化器（Optimizer）实际上会更新很多次
        """
        counter_attr = "%s_optimizer_step" % prefix
        for metrics in step_metrics: # 代表着针对同一批数据的不同 Mini-batch 和 Epoch 的优化器更新指标
            next_optimizer_step = int(getattr(self, counter_attr)) + 1
            setattr(self, counter_attr, next_optimizer_step)
            if not self.config.trainer.log_optimizer_steps:
                # 检查配置中是否允许记录如此细粒度（Optimizer Step 级别）的日志。如果不允许，则跳过后面的日志组装和发送（仅保持内部计数）。
                continue
            log_data = {
                "%s/%s" % (prefix, key): value
                for key, value in metrics.items()
            }
            log_data["%s/optimizer_step" % prefix] = float(next_optimizer_step)
            log_data["training/global_step"] = float(self.global_step + 1)
            log_data["training/epoch"] = float(self.train_loader.epoch)
            self._log_metrics(log_data)

    def _uses_critic(self) -> bool:
        # This method is new in Phase 2 so PPO and GRPO can share one trainer loop
        # without introducing a separate actor-only trainer for GRPO.
        return self.value_worker is not None and self.config.algorithm.advantage_estimator != "grpo"

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
            - 确保批次中的每个样本(prompt)都有一个唯一的 UID，以便在后续的 rollout 和奖励计算过程中进行跟踪。
            - 根据采样参数重复批次中的每个样本，以便在 rollout 过程中生成多个响应版本。
            - 加入 rollout_index 来区分同一原始样本的不同采样版本

        """
        batch = self._ensure_uids(batch) # uid表示每一个样本的 prompt 的唯一标识
        repeated = batch.repeat(sampling.n, interleave=True)
        # 这里的 rollout_index 是为了在后续的奖励计算中区分同一原始 prompt（同一组）的不同采样版本
        # 例如index = 0, 1, 2, 3, 4 对应原始样本的不同版本，当 n=2 时，index % n 得[0, 1, 0, 1, 0]，表示每两个样本是一组采样版本。
        repeated.non_tensor["rollout_index"] = [index % sampling.n for index in range(len(repeated))]
        return repeated

    def _extract_rewards(self, batch: RLBatch) -> None:
        """
        加入字段 token_level_scores 和 extra 到 batch 中：
        Function:
            - 使用 reward_manager 来计算批次中每个样本的奖励分数，并将这些分数分配到响应文本的 token 上，构建一个 token_level_scores 列表，其中每个元素对应一个响应文本的 token-level 的奖励分数。
            - 如果 reward_manager 计算过程中返回了额外的信息（extra），则将这些信息按照键进行收集，并存储在 batch.non_tensor 中，以便在后续的分析或日志记录中使用。
        """
        reward_result = self.reward_manager.compute(batch) # 一个 dataclass, 包含 token_level_scores 和 extra 两个字段
        batch.batch["token_level_scores"] = reward_result.token_level_scores
        if reward_result.extra:
            for key, values in reward_result.extra.items():
                batch.non_tensor[key] = values

    def _balance_rollout_batch(self, batch: RLBatch) -> RLBatch:
        """
        Function:
            在强化学习微调中（尤其是 GRPO 或 n > 1 的 PPO），模型生成的回复长短不一。如果按原始顺序直接把数据切成 mini-batch 送给 Actor 进行训练，极易出现显存碎片化或负载不均：
            某个 mini-batch 凑巧全是长序列，导致 GPU 显存溢出（OOM）。某个 mini-batch 全是短序列，导致 GPU 算力闲置。
            该函数的作用是在进入打分和训练阶段前，计算每条数据的真实 Token 长度，将它们重新排列。它在保证同一个 prompt 的多个预测版本（同 UID）不被打散的前提下，让切分后的各个 mini-batch 在总长度（Workload）上尽可能平均。
        Note:
            - 当 rollout 次数 n 大于设定的 ppo_mini_batch_size 时，该平衡逻辑和实际截断切分会产生机制上的矛盾（桶多组少）。
            - 建议配置上多注意 ppo_mini_batch_size 大于或等于 n。
        """
        
        # 如果配置中关闭了平衡 (balance_batch=False)，
        # 或者整个 batch 的大小还不如一个 mini-batch 大，
        # 或者缺少长度计算所需的键值，则直接原样返回。
        if not self.config.trainer.balance_batch:
            return batch
        if len(batch) <= self.config.actor.ppo_mini_batch_size:
            return batch
        if "prompts" not in batch.batch or "response_mask" not in batch.batch:
            return batch

        target_partition_rows = max(1, self.config.actor.ppo_mini_batch_size) 
        grouped_indices: Dict[str, list[int]] = {}
        row_group_keys = batch.non_tensor.get("uid")
        if row_group_keys is None:
            row_group_keys = [str(index) for index in range(len(batch))]
        for row_index, group_key in enumerate(row_group_keys):
            grouped_indices.setdefault(str(group_key), []).append(row_index)

        grouped_rows: list[list[int]] = list(grouped_indices.values())
        if len(grouped_rows) <= 1:
            return batch

        grouped_workloads = []
        for row_indices in grouped_rows: # 每一个 prompt 组的索引列表
            workload = 0 # 计算这个 prompt 组的总 token 数，作为它的 workload。workload 的计算方式是：对于这个 prompt 组中的每一行数据，先加上 prompt 的 token 数量，再加上 response 中被 response_mask 标记为 1 的 token 数量。这样得到的 workload 就是这个 prompt 组在训练时对 GPU 资源的消耗程度。
            for row_index in row_indices:
                workload += len(batch.batch["prompts"][row_index])
                workload += sum(int(value) for value in batch.batch["response_mask"][row_index])
            grouped_workloads.append((workload, row_indices))

        grouped_workloads.sort(key=lambda item: item[0], reverse=True) # 按照 workload 从大到小排序，优先处理那些工作量大的 prompt 组，以便更好地平衡后续 mini-batch 的负载。
        partition_count = max(1, math.ceil(len(batch) / target_partition_rows)) # 
        partitions = [{"rows": 0, "workload": 0, "indices": []} for _ in range(partition_count)]

        for workload, row_indices in grouped_workloads: # 为每一组找一个合适的 partition 来放置它
            fitting_partitions = [
                partition
                for partition in partitions
                if partition["rows"] == 0 or partition["rows"] + len(row_indices) <= target_partition_rows
            ] # 找出当前 prompt 组（row_indices）可以放得下的 partition，条件是这个 partition 要么还没有任何行（rows == 0），要么加上当前 prompt 组的行数后不超过 target_partition_rows。
            if fitting_partitions:
                target_partition = min(fitting_partitions, key=lambda partition: (partition["workload"], partition["rows"]))
            else:
                target_partition = min(partitions, key=lambda partition: (partition["workload"], partition["rows"]))
            target_partition["indices"].extend(row_indices)
            target_partition["rows"] += len(row_indices)
            target_partition["workload"] += workload

        reordered_indices = []
        for partition in partitions:
            reordered_indices.extend(partition["indices"])
        if reordered_indices == list(range(len(batch))):
            return batch

        balanced_batch = batch.select(reordered_indices)
        balanced_batch.meta["balanced_by_length"] = True
        return balanced_batch

    def _maybe_dump_train_preview(self, batch: RLBatch, logged_step: int) -> None:
        # This method is new in Phase 2 because the Phase 1 trainer only exposed
        # numeric logs. A tiny batch preview makes rollout and reward debugging much
        # faster without introducing another analytics subsystem.
        if self.artifact_writer is None or self.config.trainer.train_dump_freq <= 0: # 如果不需要写入训练预览，直接返回
            return
        if logged_step % self.config.trainer.train_dump_freq != 0:
            return
        preview_rows = build_batch_preview_rows(batch, self.config.trainer.dump_max_rows)
        self.artifact_writer.write_train_preview(logged_step, preview_rows)

    def _maybe_dump_validation_preview(
        self,
        metrics: Dict[str, float],
        preview_rows: list[Dict[str, object]],
    ) -> None:
        # This method is new in Phase 2 because validation summaries became easier
        # to trust once they were paired with a few representative prompt/response rows.
        if self.artifact_writer is None or self.config.trainer.validation_dump_freq <= 0:
            return
        if self.global_step % self.config.trainer.validation_dump_freq != 0:
            return
        self.artifact_writer.write_validation_preview(self.global_step, metrics, preview_rows)

    def _shape_rewards(self, batch: RLBatch) -> Dict[str, float]:
        """
        Function:
            - 整合 token_level_scores 和 KL 惩罚，生成最终的 token_level_rewards。
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
            "log_step": self.log_step,
            "actor_optimizer_step": self.actor_optimizer_step,
            "critic_optimizer_step": self.critic_optimizer_step,
            "train_loader_state": self.train_loader.state_dict(),
            "policy_state": self.policy_worker.state_dict(),
            "reference_state": self.reference_worker.state_dict() if self.reference_worker is not None else None,
            "value_state": self.value_worker.state_dict() if self.value_worker is not None else None,
            "rollout_state": self.rollout_engine.state_dict(),
            "config": self.config.to_dict(),
        }

    def save_checkpoint(self) -> None:
        if self.runtime.is_main_process:
            self.checkpoint_manager.save(self.global_step, self._checkpoint_payload())
        self.runtime.barrier()

    def load_checkpoint(self) -> bool:
        payload = self.checkpoint_manager.load_latest() # load(checkpoint_dir) | none
        if payload is None:
            return False
        self.global_step = int(payload["global_step"])
        self.log_step = int(payload.get("log_step", self.global_step))
        self.actor_optimizer_step = int(payload.get("actor_optimizer_step", 0))
        self.critic_optimizer_step = int(payload.get("critic_optimizer_step", 0))
        self.train_loader.load_state_dict(payload["train_loader_state"])
        self.policy_worker.load_state_dict(payload["policy_state"])
        if self.reference_worker is not None and payload.get("reference_state") is not None:
            self.reference_worker.load_state_dict(payload["reference_state"])
        if self.value_worker is not None and payload.get("value_state") is not None:
            self.value_worker.load_state_dict(payload["value_state"])
        self.rollout_engine.load_state_dict(payload.get("rollout_state", {}))
        self.rollout_engine.sync_policy(self.policy_worker.state_dict()) # TODO: 这里的同步有些多余，因为在调用这个函数的外部本身就会执行一次同步
        return True

    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        scores = []
        data_sources = []
        reward_extras: Dict[str, list[object]] = {}
        preview_rows: list[Dict[str, object]] = []
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
            for key, values in reward_result.extra.items():
                reward_extras.setdefault(key, []).extend(values)
            remaining_preview_rows = self.config.trainer.dump_max_rows - len(preview_rows)
            if remaining_preview_rows > 0:
                preview_rows.extend(
                    build_batch_preview_rows(
                        batch,
                        remaining_preview_rows,
                        reward_scores=reward_result.token_level_scores,
                        reward_extras=reward_result.extra,
                    )
                )
        gathered_scores = self.runtime.all_gather_objects(scores)
        gathered_data_sources = self.runtime.all_gather_objects(data_sources)
        gathered_reward_extras = self.runtime.all_gather_objects(reward_extras)
        gathered_preview_rows = self.runtime.all_gather_objects(preview_rows)
        self.val_loader.load_state_dict(state) # 恢复验证集的读取位置
        merged_scores = [score for rank_scores in gathered_scores for score in rank_scores]
        merged_data_sources = [source for rank_sources in gathered_data_sources for source in rank_sources]
        merged_reward_extras: Dict[str, list[object]] = {}
        for rank_reward_extras in gathered_reward_extras:
            for key, values in rank_reward_extras.items():
                merged_reward_extras.setdefault(key, []).extend(values)
        merged_preview_rows = [
            row
            for rank_preview_rows in gathered_preview_rows
            for row in rank_preview_rows
        ][: self.config.trainer.dump_max_rows]
        metrics = summarize_validation(merged_scores, merged_data_sources, reward_extras=merged_reward_extras)
        if self.runtime.is_main_process:
            self._maybe_dump_validation_preview(metrics, merged_preview_rows)
        metrics = self.runtime.broadcast_object(metrics, src=0)
        self.last_validation_metrics = metrics
        return metrics

    def train_step(self, batch: RLBatch) -> Dict[str, float]: 

        step_started = time.time()
        metrics: Dict[str, float] = {}
        timing: Dict[str, float] = {}

        rollout_params = _sampling_to_params(self.config.rollout.train)
        rollout_batch = self._prepare_rollout_batch(batch, rollout_params)

        t0 = time.time()
        # 生成response
        rollout_batch = self.rollout_engine.generate(rollout_batch, rollout_params)
        timing["rollout"] = time.time() - t0
        rollout_batch = self._balance_rollout_batch(rollout_batch)

        t0 = time.time()
        # 1. 使用 reward_fn 计算句子的奖励
        self._extract_rewards(rollout_batch)
        timing["reward_extract"] = time.time() - t0

        # 每个worker执行各自的工作：
        # 计时、计算、记录数值、记录指标

        # 2. 计算旧策略的 log 概率 -> 重要性采样
        # 这里 policy_worker 和 rollout 的模型的参数实际上是一样的: 
        t0 = time.time()
        old_log_probs = self.policy_worker.compute_log_probs(rollout_batch) # 这里会发生显存的增长例如大约3k->8k
        rollout_batch.batch["old_log_probs"] = old_log_probs.log_probs
        metrics.update({"actor/%s" % key: value for key, value in old_log_probs.metrics.items()})
        timing["policy_eval"] = time.time() - t0
        # 3. 计算参考模型的 log 概率（如果 reference_worker 存在）-> 奖励 shaping 时会被用做 KL 惩罚作为 token-level 的奖励调整，同时也会被记录到 metrics 中以供分析。
        if self.reference_worker is not None:
            t0 = time.time()
            ref_log_probs = self.reference_worker.compute_log_probs(rollout_batch)
            rollout_batch.batch["ref_log_probs"] = ref_log_probs.log_probs
            metrics.update({"ref/%s" % key: value for key, value in ref_log_probs.metrics.items()})
            timing["reference_eval"] = time.time() - t0
        # 4. 计算状态值函数的估计值（如果 value_worker 存在） -> 优势计算时会被用做状态值估计来计算优势函数，同时也会被记录到 metrics 中以供分析。
        if self._uses_critic():
            t0 = time.time()
            values = self.value_worker.compute_values(rollout_batch)  # type: ignore[union-attr]
            rollout_batch.batch["values"] = values.values
            metrics.update({"critic/%s" % key: value for key, value in values.metrics.items()})
            timing["value_eval"] = time.time() - t0
        else:
            rollout_batch.batch["values"] = [[0.0 for _ in row] for row in rollout_batch.batch["response_mask"]]

        t0 = time.time()
        # 5. 得出token_level_rewards字段
        reward_metrics = self._shape_rewards(rollout_batch)
        metrics.update(reward_metrics)
        timing["reward_shape"] = time.time() - t0

        t0 = time.time()
        # 根据 token_level_rewards 和 values 计算优势函数，并将优势值添加到 rollout_batch 中，以便后续的策略更新使用。同时，将优势计算的相关指标添加到 metrics 中，以便进行日志记录和分析。
        self._compute_advantages(rollout_batch)
        timing["advantage"] = time.time() - t0

        # ===================== 进入更新阶段 =====================
        # 如果 value_worker 存在，首先更新价值函数，并将相关指标添加到 metrics 中，以便进行日志记录和分析。
        if self._uses_critic():
            t0 = time.time()
            critic_update = self.value_worker.update(rollout_batch)  # type: ignore[union-attr]
            metrics.update({"critic/%s" % key: value for key, value in critic_update.metrics.items()})
            self._record_optimizer_step_metrics("critic", critic_update.step_metrics)
            timing["critic_update"] = time.time() - t0
        # 如果全局步骤数已经超过了 critic_warmup 的设置，则更新策略，并将相关指标添加到 metrics 中，以便进行日志记录和分析。同时，将更新后的策略状态同步到 rollout_engine 中，以确保在后续的 rollout 过程中使用最新的策略进行生成。
        # 在 critic_warmup 阶段，训练过程只更新价值函数，而不更新策略。这是为了让价值函数先行学习一个相对稳定的状态值估计，从而在后续的策略更新中提供更准确的优势估计，帮助策略更有效地学习。
        if self.global_step >= self.config.trainer.critic_warmup:
            t0 = time.time()
            actor_update = self.policy_worker.update(rollout_batch) # 会更新self.config.ppo_epochs次
            metrics.update({"actor/%s" % key: value for key, value in actor_update.metrics.items()})
            self._record_optimizer_step_metrics("actor", actor_update.step_metrics)
            self.rollout_engine.sync_policy(self.policy_worker.state_dict()) # 更新结束后同步策略到 rollout_engine 中，以确保在后续的 rollout 过程中使用最新的策略进行生成。
            timing["actor_update"] = time.time() - t0

        timing["step"] = time.time() - step_started
        metrics.update(
            compute_data_metrics(
                rollout_batch,
                use_critic=self._uses_critic(),
                response_length_limit=self.config.rollout.response_length,
            )
        )
        metrics.update(compute_timing_metrics(timing))
        metrics.update(
            compute_throughput_metrics(
                rollout_batch,
                timing["step"],
                world_size=self.runtime.world_size,
            )
        )
        metrics["training/global_step"] = float(self.global_step + 1)
        metrics["training/epoch"] = float(self.train_loader.epoch)
        self._maybe_dump_train_preview(rollout_batch, self.global_step + 1)
        return metrics

    def fit(self) -> Dict[str, float]:
        self.load_checkpoint()
        self.rollout_engine.sync_policy(self.policy_worker.state_dict())

        if self.config.trainer.validate_before_train:
            val_metrics = self.validate()
            if val_metrics:
                self._log_metrics(val_metrics)
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

            metrics = self.train_step(batch) # 来自训练过程的指标
            self.global_step += 1

            if self.config.trainer.test_freq > 0 and self.val_loader is not None:
                if self.global_step % self.config.trainer.test_freq == 0 or self.global_step == max_steps:
                    metrics.update(self.validate()) # 来自验证过程的指标

            self._log_metrics(metrics) # 来自训练过程和验证过程的指标都会被记录到 tracker 中，以便后续的分析和可视化。

            # 根据配置中的 save_freq 设置，定期保存训练检查点。当全局步骤数达到指定的保存频率时，调用 save_checkpoint 方法将当前的训练状态保存到磁盘
            if self.config.trainer.save_freq > 0:
                if self.global_step % self.config.trainer.save_freq == 0 or self.global_step == max_steps:
                    self.save_checkpoint()

        if self.config.trainer.save_freq == 0:
            self.save_checkpoint()
        return self.last_validation_metrics


__all__ = ["RLTrainer", "build_trainer"]
