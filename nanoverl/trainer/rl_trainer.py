"""Synchronous RL trainer."""

from __future__ import annotations

import math
import uuid
from typing import Dict, Optional

from nanoverl.algos import AlgorithmStepContext, RLAlgorithm, create_algorithm
from nanoverl.algos.base import sampling_to_params
from nanoverl.checkpoint.manager import CheckpointManager
from nanoverl.config import TrainerConfig
from nanoverl.core.batch import RLBatch
from nanoverl.data.dataset import JsonDataset, StatefulDataLoader
from nanoverl.distributed import TorchDistributedRuntime
from nanoverl.logging.trackers import TrackingManager
from nanoverl.reward import RewardManager, load_reward_function
from nanoverl.rollout import (
    PolicySyncResult,
    PolicySyncer,
    SamplingParams,
    create_rollout_engine,
)
from nanoverl.trainer.artifacts import ArtifactWriter, build_batch_preview_rows
from nanoverl.trainer.validation import summarize_validation
from nanoverl.workers import create_policy_worker, create_reference_worker, create_value_worker


CHECKPOINT_VERSION = 2


def build_trainer(config: TrainerConfig) -> "RLTrainer":
    runtime: TorchDistributedRuntime = TorchDistributedRuntime.from_environment()
    algorithm = create_algorithm(config.algorithm.name or "ppo")
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
    use_critic = config.critic.enable and algorithm.uses_critic(config)
    value_worker = create_value_worker(config.critic.backend, config.model, config.critic) if use_critic else None

    rollout_engine = create_rollout_engine(config.rollout.backend, config.model, config.data, config.rollout)
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
        algorithm=algorithm,
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
        algorithm: RLAlgorithm,
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
        self.algorithm = algorithm
        self.policy_syncer = PolicySyncer()
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
        return self.value_worker is not None and self.algorithm.uses_critic(self.config)

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

    def _balance_rollout_batch(self, batch: RLBatch) -> RLBatch:
        """
        Function:
            在强化学习微调中（尤其是 GRPO/RLOO 或 n > 1 的 grouped sampling），模型生成的回复长短不一。如果按原始顺序直接把数据切成 mini-batch 送给 Actor 进行训练，极易出现显存碎片化或负载不均：
            某个 mini-batch 凑巧全是长序列，导致 GPU 显存溢出（OOM）。某个 mini-batch 全是短序列，导致 GPU 算力闲置。
            该函数的作用是在进入打分和训练阶段前，计算每条数据的真实 Token 长度，将它们重新排列。它在保证同一个 prompt 的多个预测版本（同 UID）不被打散的前提下，让切分后的各个 mini-batch 在总长度（Workload）上尽可能平均。
        Note:
            - 当 rollout 次数 n 大于设定的 mini_batch_size 时，该平衡逻辑和实际截断切分会产生机制上的矛盾（桶多组少）。
            - 建议配置上多注意 mini_batch_size 大于或等于 n。
        """

        # 如果配置中关闭了平衡 (balance_batch=False)，
        # 或者整个 batch 的大小还不如一个 mini-batch 大，
        # 或者缺少长度计算所需的键值，则直接原样返回。
        if not self.config.trainer.balance_batch:
            return batch
        if len(batch) <= self.config.actor.mini_batch_size:
            return batch
        if "prompts" not in batch.batch or "response_mask" not in batch.batch:
            return batch

        target_partition_rows = max(1, self.config.actor.mini_batch_size)
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

    def _checkpoint_payload(self) -> Dict[str, object]:
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "trainer_state": {
                "global_step": self.global_step,
                "log_step": self.log_step,
                "actor_optimizer_step": self.actor_optimizer_step,
                "critic_optimizer_step": self.critic_optimizer_step,
            },
            "loader_state": {
                "train": self.train_loader.state_dict(),
                "validation": self.val_loader.state_dict() if self.val_loader is not None else None,
            },
            "worker_state": {
                "policy": self.policy_worker.state_dict(),
                "reference": self.reference_worker.state_dict() if self.reference_worker is not None else None,
                "value": self.value_worker.state_dict() if self.value_worker is not None else None,
            },
            "rollout_state": self.rollout_engine.state_dict(),
            "config": self.config.to_dict(),
        }

    def _sync_rollout_policy(self, reason: str) -> PolicySyncResult:
        return self.policy_syncer.sync(self.policy_worker, self.rollout_engine, reason)

    def save_checkpoint(self) -> None:
        if self.runtime.is_main_process:
            self.checkpoint_manager.save(self.global_step, self._checkpoint_payload())
        self.runtime.barrier()

    def load_checkpoint(self) -> bool:
        payload = self.checkpoint_manager.load_latest() # load(checkpoint_dir) | none
        if payload is None:
            return False
        if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported checkpoint payload version: %s" % payload.get("checkpoint_version"))

        trainer_state = payload["trainer_state"]
        loader_state = payload["loader_state"]
        worker_state = payload["worker_state"]

        self.global_step = int(trainer_state["global_step"])
        self.log_step = int(trainer_state["log_step"])
        self.actor_optimizer_step = int(trainer_state["actor_optimizer_step"])
        self.critic_optimizer_step = int(trainer_state["critic_optimizer_step"])
        self.train_loader.load_state_dict(loader_state["train"])
        if self.val_loader is not None and loader_state.get("validation") is not None:
            self.val_loader.load_state_dict(loader_state["validation"])
        self.policy_worker.load_state_dict(worker_state["policy"])
        if self.reference_worker is not None and worker_state.get("reference") is not None:
            self.reference_worker.load_state_dict(worker_state["reference"])
        if self.value_worker is not None and worker_state.get("value") is not None:
            self.value_worker.load_state_dict(worker_state["value"])
        self.rollout_engine.load_state_dict(payload["rollout_state"])
        self._sync_rollout_policy("resume")
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
            batch = self._prepare_rollout_batch(batch, sampling_to_params(self.config.rollout.validation))
            batch = self.rollout_engine.generate(batch, sampling_to_params(self.config.rollout.validation))
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
        context = AlgorithmStepContext(
            config=self.config,
            policy_worker=self.policy_worker,
            reference_worker=self.reference_worker,
            value_worker=self.value_worker,
            rollout_engine=self.rollout_engine,
            reward_manager=self.reward_manager,
            runtime=self.runtime,
            global_step=self.global_step,
            train_epoch=self.train_loader.epoch,
            prepare_rollout_batch=self._prepare_rollout_batch,
            balance_rollout_batch=self._balance_rollout_batch,
            record_optimizer_step_metrics=self._record_optimizer_step_metrics,
            dump_train_preview=self._maybe_dump_train_preview,
            sync_rollout_policy=self._sync_rollout_policy,
        )
        return self.algorithm.run_step(batch, context)

    def fit(self) -> Dict[str, float]:
        resumed = self.load_checkpoint()
        if not resumed:
            self._sync_rollout_policy("startup")

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


__all__ = ["CHECKPOINT_VERSION", "RLTrainer", "build_trainer"]
