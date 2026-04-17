"""Local Hugging Face worker implementations."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from nanoverl.algos.kl import compute_kl_penalty
from nanoverl.algos.ppo import compute_policy_loss, compute_value_loss, masked_mean
from nanoverl.backends.hf import (
    average_or_zero,
    batch_lists_to_tensor,
    build_training_tensors,
    clone_model_state,
    extract_response_stats,
    get_loss_weight,
    get_prompt_lengths,
    get_response_lengths,
    load_backbone_model,
    load_causal_lm,
    load_tokenizer,
    require_hf_dependencies,
    resolve_device,
    tensor_to_list_rows,
)
from nanoverl.workers.base import LogProbResult, PolicyWorker, ReferenceWorker, UpdateResult, ValueResult, ValueWorker


def _infer_hidden_size(backbone_config) -> int:
    for field_name in ("hidden_size", "n_embd"):
        hidden_size = getattr(backbone_config, field_name, None)
        if hidden_size is not None:
            return int(hidden_size)
    raise ValueError("Could not infer hidden size for the local HF critic backbone.")


class HFWorkerBase:
    def __init__(self, model_config, device_name: Optional[str] = None):
        self.model_config = model_config
        self.device = resolve_device(device_name)
        self.tokenizer = load_tokenizer(model_config)

    def _build_model_inputs(self, batch):
        """
        Function:
            - 将 batch 中的 input_ids 和 attention_mask 转换为适合模型输入的张量格式。
        """
        torch, _, _, _ = require_hf_dependencies()
        input_ids = batch_lists_to_tensor(
            batch.batch["input_ids"],
            self.tokenizer.pad_token_id,
            device=self.device,
            dtype=torch.long,
            padding_side="right",
        )
        attention_mask = batch_lists_to_tensor(batch.batch["attention_mask"], 0, device=self.device, dtype=torch.long, padding_side='right')
        return input_ids, attention_mask

    def _compute_response_log_probs_and_entropy(self, model, batch, compute_entropy: bool = False):
        """
        Function:
            - 计算给定 batch 中响应部分的对数概率和熵值。
        Returns:
            - response_log_probs: 一个二维张量，shape为 (batch_size, max_response_length)，其中每个元素表示对应位置的响应 token 的对数概率。
            - response_entropy: 一个二维张量，shape为 (batch_size, max_response_length)，其中每个元素表示对应位置的响应 token 的熵值。
        """
        input_ids, attention_mask = self._build_model_inputs(batch)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        response_log_probs, response_entropy = extract_response_stats(
            logits=outputs.logits,
            input_ids=input_ids,
            prompt_token_counts=get_prompt_lengths(batch),
            response_token_counts=get_response_lengths(batch),
            compute_entropy=compute_entropy,
        )
        del outputs
        return response_log_probs, response_entropy

    def _no_grad(self):
        torch, _, _, _ = require_hf_dependencies()
        return torch.no_grad()

    def _iter_minibatches(self, batch, batch_size: int, shuffle: bool):
        row_indices = list(range(len(batch)))
        if shuffle and not batch.meta.get("balanced_by_length", False):
            random.shuffle(row_indices)
        step = max(1, batch_size)
        for start_index in range(0, len(row_indices), step):
            yield batch.select(row_indices[start_index : start_index + step])

    def _iter_microbatches(self, batch, micro_batch_size: Optional[int]):
        if not micro_batch_size or micro_batch_size >= len(batch):
            yield batch
            return
        row_indices = list(range(len(batch)))
        for start_index in range(0, len(row_indices), micro_batch_size):
            yield batch.select(row_indices[start_index : start_index + micro_batch_size])


class HFReferenceWorker(HFWorkerBase, ReferenceWorker):
    def __init__(self, model_config, ref_config):
        super().__init__(model_config, device_name=ref_config.device)
        self.ref_config = ref_config
        self.model = load_causal_lm(model_config).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def compute_log_probs(self, batch) -> LogProbResult:
        with self._no_grad():
            response_log_probs, _ = self._compute_response_log_probs_and_entropy(self.model, batch, compute_entropy=False)
        response_lengths = get_response_lengths(batch)
        return LogProbResult(
            log_probs=tensor_to_list_rows(response_log_probs, response_lengths),
            metrics={"reference_enabled": 1.0},
        )

    def state_dict(self) -> Dict[str, Any]:
        return {"model_state": clone_model_state(self.model.state_dict())}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("model_state") is not None:
            self.model.load_state_dict(state["model_state"])
        self.model.to(self.device)
        self.model.eval()


class HFPolicyWorker(HFWorkerBase, PolicyWorker):
    def __init__(self, model_config, actor_config):
        super().__init__(model_config, device_name=actor_config.device)
        torch, _, _, _ = require_hf_dependencies()
        self.actor_config = actor_config
        self.model = load_causal_lm(model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=actor_config.lr,
            betas=tuple(actor_config.betas),
            eps=actor_config.eps,
            weight_decay=actor_config.weight_decay,
        )
        # TODO: Use bnb for 8-bit AdamW if needed
        # self.optimizer = bnb.optim.AdamW8bit(
        #     self.model.parameters(),
        #     lr=actor_config.lr,
        #     betas=tuple(actor_config.betas),
        #     eps=actor_config.eps,
        #     weight_decay=actor_config.weight_decay,
        # )
        self.update_steps = 0

    def compute_log_probs(self, batch) -> LogProbResult:
        """
        Function:
            - 计算给定 batch 中响应部分的对数概率和熵值。
        Returns:
            - LogProbResult 对象，其中包含响应的对数概率、熵值以及一些指标（如 policy_update_steps）。
                - log_probs: 一个列表的列表，每个子列表对应 batch 中一个样本的响应部分的对数概率。
                - entropy: 一个列表的列表，每个子列表对应 batch 中一个样本的响应部分的熵值。
                - metrics: 一个字典，包含一些指标信息，例如 policy_update_steps，表示当前策略更新的步数。
        """
        self.model.eval()
        all_log_probs = []
        with self._no_grad():
            for minibatch in self._iter_minibatches(batch, self.actor_config.ppo_mini_batch_size, shuffle=False):
                response_log_probs, _ = self._compute_response_log_probs_and_entropy(
                    self.model, minibatch, compute_entropy=False
                ) # 这里会有显存的上升
                response_lengths = get_response_lengths(minibatch)
                all_log_probs.extend(tensor_to_list_rows(response_log_probs, response_lengths))        
        return LogProbResult(
            log_probs=all_log_probs,
            metrics={"policy_update_steps": float(self.update_steps)},
        )

    def update(self, batch) -> UpdateResult:
        torch, _, _, _ = require_hf_dependencies()
        metric_history: Dict[str, List[float]] = {} # 这个metric_history字典每次附加上去的是每个 microbatch 的指标值，最后在整个 update 结束后会对这些指标值进行平均
        self.model.train()

        for _ in range(self.actor_config.ppo_epochs): # 这里的 epochs 表示要重复利用同一批 rollout 数据进行多少轮策略优化更新。
            for minibatch in self._iter_minibatches(batch, self.actor_config.ppo_mini_batch_size, self.actor_config.shuffle): # 每一轮都会将整个 batch 划分成多个 mini-batch，然后对每个 mini-batch 进行一次完整的前向和反向传播，最后更新模型参数。
                microbatches = tuple(self._iter_microbatches(minibatch, self.actor_config.micro_batch_size))
                if not microbatches:
                    continue

                microbatch_weights: List[float] = [
                    max(get_loss_weight(microbatch, self.actor_config.loss_agg_mode), 1.0) for microbatch in microbatches
                ] # 如果是per_token 的 loss_agg_mode，那么在_compute_policy_loss中计算的损失是micro_batch 上面每个 token 的平均损失，因此为了实现梯度累积这里每一个 micro_batch 的 loss 权重就是这个 micro_batch 中所有 response_mask 中有效 token 的总数。
                total_weight = sum(microbatch_weights)

                self.optimizer.zero_grad() # 在对一个 mini-batch 的所有 micro-batch 进行反向传播之前，先将优化器的梯度缓存清零，以避免梯度累积到之前的 mini-batch 中。
                for microbatch, microbatch_weight in zip(microbatches, microbatch_weights): # 对于每个 mini-batch，我们进一步将其划分成多个 micro-batch，以便在内存受限的情况下进行训练。对于每个 micro-batch，我们计算当前策略的对数概率和熵值，然后根据 PPO 的损失函数计算策略损失，并进行反向传播。最后，我们根据 micro-batch 的权重对损失进行缩放，以确保在更新模型参数时考虑到不同 micro-batch 的重要性。
                    current_log_probs, current_entropy = self._compute_response_log_probs_and_entropy(
                        self.model,
                        microbatch,
                        compute_entropy=self.actor_config.entropy_coeff != 0.0,
                    )
                    field_names = ["old_log_probs", "advantages"]
                    if self.actor_config.use_kl_loss and "ref_log_probs" in microbatch.batch:
                        field_names.append("ref_log_probs")
                    training_tensors = build_training_tensors(microbatch, self.device, field_names=field_names)
                    policy_loss, step_metrics = compute_policy_loss(
                        old_log_probs=training_tensors["old_log_probs"],
                        log_probs=current_log_probs,
                        advantages=training_tensors["advantages"],
                        response_mask=training_tensors["response_mask"],
                        cliprange=self.actor_config.clip_ratio,
                        cliprange_low=self.actor_config.clip_ratio_low,
                        cliprange_high=self.actor_config.clip_ratio_high,
                        clip_ratio_c=self.actor_config.clip_ratio_c,
                        loss_agg_mode=self.actor_config.loss_agg_mode,
                    )
                    if current_entropy is not None:
                        entropy_bonus = masked_mean(current_entropy, training_tensors["response_mask"])
                        total_loss = policy_loss - (self.actor_config.entropy_coeff * entropy_bonus)
                        step_metrics["policy_entropy"] = float(entropy_bonus.detach().cpu())
                    else:
                        total_loss = policy_loss
                        step_metrics["policy_entropy"] = 0.0

                    if self.actor_config.use_kl_loss and "ref_log_probs" in microbatch.batch:
                        kl_penalty = compute_kl_penalty(
                            current_log_probs,
                            training_tensors["ref_log_probs"],
                            mode="low_var_kl",
                        )
                        kl_mean = masked_mean(kl_penalty, training_tensors["response_mask"])
                        total_loss = total_loss + (self.actor_config.kl_loss_coef * kl_mean)
                        step_metrics["actor_kl_loss"] = float(kl_mean.detach().cpu())

                    scaled_loss = total_loss * (microbatch_weight / max(total_weight, 1.0))
                    scaled_loss.backward()
                    step_metrics["actor_loss"] = float(total_loss.detach().cpu())
                    for metric_name, metric_value in step_metrics.items():
                        metric_history.setdefault(metric_name, []).append(float(metric_value))

                if self.actor_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.actor_config.max_grad_norm)
                self.optimizer.step() # 对一个mini-batch的所有 micro-batch 进行反向传播并累积梯度后，进行一次优化器步骤来更新模型参数。是显存增加最多的地方，例如 8k->16K

        self.update_steps += 1 # 实际上这里更新了ppo_epochs*len(batch)/ppo_mini_batch_size 次，但我们把它当做一次整体的策略更新。
        self.model.eval()
        averaged_metrics = {name: average_or_zero(values) for name, values in metric_history.items()}
        averaged_metrics["actor_update_steps"] = float(self.update_steps)
        return UpdateResult(metrics=averaged_metrics)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model_state": clone_model_state(self.model.state_dict()),
            "optimizer_state": self.optimizer.state_dict(),
            "update_steps": self.update_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("model_state") is not None:
            self.model.load_state_dict(state["model_state"])
        if state.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(state["optimizer_state"])
        self.update_steps = int(state.get("update_steps", 0))
        self.model.to(self.device)
        self.model.eval()


class HFValueWorker(HFWorkerBase, ValueWorker):
    def __init__(self, model_config, value_config):
        super().__init__(model_config, device_name=value_config.device)
        torch, _, _, _ = require_hf_dependencies()
        self.value_config = value_config
        self.backbone = load_backbone_model(model_config, path=model_config.critic_path).to(self.device)
        self.value_head = torch.nn.Linear(_infer_hidden_size(self.backbone.config), 1).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.value_head.parameters()),
            lr=value_config.lr,
            betas=tuple(value_config.betas),
            eps=value_config.eps,
            weight_decay=value_config.weight_decay,
        )
        self.update_steps = 0

    def _compute_response_values(self, batch):
        input_ids, attention_mask = self._build_model_inputs(batch)
        hidden_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        token_values = self.value_head(hidden_states).squeeze(-1)

        prompt_token_counts = get_prompt_lengths(batch)
        response_token_counts = get_response_lengths(batch)
        max_response_length = max(response_token_counts, default=0)
        response_values = token_values.new_zeros((token_values.shape[0], max_response_length))
        for row_index, (prompt_token_count, response_token_count) in enumerate(
            zip(prompt_token_counts, response_token_counts)
        ):
            if response_token_count <= 0:
                continue
            response_start = prompt_token_count
            response_stop = response_start + response_token_count
            response_values[row_index, :response_token_count] = token_values[row_index, response_start:response_stop]
        return response_values

    def compute_values(self, batch) -> ValueResult:
        self.backbone.eval()
        self.value_head.eval()
        with self._no_grad():
            response_values = self._compute_response_values(batch)
        return ValueResult(
            values=tensor_to_list_rows(response_values, get_response_lengths(batch)),
            metrics={"critic_update_steps": float(self.update_steps)},
        )

    def update(self, batch) -> UpdateResult:
        torch, _, _, _ = require_hf_dependencies()
        metric_history: Dict[str, List[float]] = {}
        self.backbone.train()
        self.value_head.train()

        for _ in range(self.value_config.ppo_epochs):
            minibatches = self._iter_minibatches(batch, self.value_config.ppo_mini_batch_size, self.value_config.shuffle)
            for minibatch in minibatches:
                microbatches = tuple(self._iter_microbatches(minibatch, self.value_config.micro_batch_size))
                if not microbatches:
                    continue

                microbatch_weights = [
                    max(get_loss_weight(microbatch, self.value_config.loss_agg_mode), 1.0) for microbatch in microbatches
                ]
                total_weight = sum(microbatch_weights)

                self.optimizer.zero_grad()
                for microbatch, microbatch_weight in zip(microbatches, microbatch_weights):
                    predicted_values = self._compute_response_values(microbatch)
                    training_tensors = build_training_tensors(microbatch, self.device, field_names=("returns",))
                    value_loss, step_metrics = compute_value_loss(
                        values=predicted_values,
                        returns=training_tensors["returns"],
                        response_mask=training_tensors["response_mask"],
                        cliprange_value=self.value_config.cliprange_value,
                        loss_agg_mode=self.value_config.loss_agg_mode,
                    )
                    scaled_loss = value_loss * (microbatch_weight / max(total_weight, 1.0))
                    scaled_loss.backward()
                    step_metrics["critic_loss"] = float(value_loss.detach().cpu())
                    for metric_name, metric_value in step_metrics.items():
                        metric_history.setdefault(metric_name, []).append(float(metric_value))

                if self.value_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.backbone.parameters()) + list(self.value_head.parameters()),
                        self.value_config.max_grad_norm,
                    )
                self.optimizer.step()

        self.update_steps += 1
        self.backbone.eval()
        self.value_head.eval()
        averaged_metrics = {name: average_or_zero(values) for name, values in metric_history.items()}
        averaged_metrics["critic_update_steps"] = float(self.update_steps)
        return UpdateResult(metrics=averaged_metrics)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "backbone_state": clone_model_state(self.backbone.state_dict()),
            "value_head_state": clone_model_state(self.value_head.state_dict()),
            "optimizer_state": self.optimizer.state_dict(),
            "update_steps": self.update_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("backbone_state") is not None:
            self.backbone.load_state_dict(state["backbone_state"])
        if state.get("value_head_state") is not None:
            self.value_head.load_state_dict(state["value_head_state"])
        if state.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(state["optimizer_state"])
        self.update_steps = int(state.get("update_steps", 0))
        self.backbone.to(self.device)
        self.value_head.to(self.device)
        self.backbone.eval()
        self.value_head.eval()


__all__ = ["HFPolicyWorker", "HFReferenceWorker", "HFValueWorker"]
