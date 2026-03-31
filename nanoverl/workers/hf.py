"""Local Hugging Face worker implementations."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from nanoverl.backends.hf import (
    average_or_zero,
    batch_lists_to_tensor,
    build_training_tensors,
    clone_model_state,
    extract_response_stats,
    get_default_device,
    get_loss_weight,
    get_prompt_lengths,
    get_response_lengths,
    load_backbone_model,
    load_causal_lm,
    require_hf_dependencies,
    tensor_to_list_rows,
)
from nanoverl.workers.base import LogProbResult, PolicyWorker, ReferenceWorker, UpdateResult, ValueResult, ValueWorker


def _masked_mean(values, mask) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _aggregate_loss(loss_matrix, response_mask, loss_agg_mode: str) -> torch.Tensor:
    torch, _, _, _ = require_hf_dependencies()
    if loss_agg_mode == "token-mean":
        return _masked_mean(loss_matrix, response_mask)

    per_sequence_losses = []
    token_counts = []
    for row_loss, row_mask in zip(loss_matrix, response_mask):
        valid_loss = row_loss[row_mask > 0]
        if valid_loss.numel() == 0:
            continue
        per_sequence_losses.append(valid_loss.sum()) # 每一行的损失总和，即每个序列的总损失。
        token_counts.append(float(valid_loss.numel())) # 每一行中有效 token 的数量，即每个序列中参与计算损失的 token 数量。
    if not per_sequence_losses:
        return torch.tensor(0.0, device=loss_matrix.device)

    stacked_losses = torch.stack(per_sequence_losses)
    if loss_agg_mode == "seq-mean-token-sum":
        return stacked_losses.mean()
    if loss_agg_mode == "seq-mean-token-mean":
        sequence_means = torch.stack([loss / count for loss, count in zip(per_sequence_losses, token_counts)])
        return sequence_means.mean()
    if loss_agg_mode == "seq-mean-token-sum-norm":
        return stacked_losses.mean() / max(token_counts)
    raise ValueError("Unsupported loss_agg_mode: %s" % loss_agg_mode)


def _compute_policy_loss(
    current_log_probs,
    old_log_probs,
    advantages,
    response_mask,
    clip_ratio: float,
    clip_ratio_low: Optional[float],
    clip_ratio_high: Optional[float],
    clip_ratio_c: float,
    loss_agg_mode: str,
):
    # 设定策略更新的允许范围，例如 $0.2$，即新策略的概率只能在旧策略的 $[0.8, 1.2]$ 之间浮动。
    clip_low = clip_ratio if clip_ratio_low is None else clip_ratio_low
    clip_high = clip_ratio if clip_ratio_high is None else clip_ratio_high
    
    # 采样（Rollout）时当时网络生成的概率对数（$\log \pi_{old}$）。
    probability_ratio = (current_log_probs - old_log_probs).clamp(min=-20.0, max=20.0).exp() # 计算当前策略和旧策略的概率比值 $\frac{\pi_{new}}{\pi_{old}}$，通过对数概率的差值取指数实现。这里还对差值进行了裁剪，以避免数值不稳定。
    unclipped_loss = -advantages * probability_ratio
    clipped_ratio = probability_ratio.clamp(min=1.0 - clip_low, max=1.0 + clip_high) # 对概率比值进行裁剪，使其在 $[1 - \epsilon, 1 + \epsilon]$ 的范围内，其中 $\epsilon$ 是 clip_ratio。这是 PPO 的核心机制之一，旨在限制策略更新的幅度，防止过大的更新导致训练不稳定。
    clipped_loss = -advantages * clipped_ratio # 计算裁剪后的损失，即使用裁剪后的概率比值来计算损失。
    clipped_candidate = clipped_loss.maximum(unclipped_loss) # PPO 的损失函数是 unclipped_loss 和 clipped_loss 中的较大者，这样可以确保在概率比值超过裁剪范围时，损失不会继续增加，从而限制了策略更新的幅度。
    
    lower_clipped_loss = (-advantages * clip_ratio_c).minimum(clipped_candidate) # 这个部分是对负优势（即新策略表现更差的情况）进行额外的裁剪，防止策略更新过度惩罚那些表现更差的动作。clip_ratio_c 是一个额外的裁剪参数，用于控制这种情况的损失。
    final_loss = clipped_candidate.where(advantages >= 0.0, lower_clipped_loss)
    masked_loss = final_loss * response_mask # shape为 (batch_size, max_response_length)，其中每个元素表示对应位置的损失值，如果该位置是有效的响应 token 则为计算得到的损失，否则为0。
    aggregate = _aggregate_loss(masked_loss, response_mask, loss_agg_mode) # 根据 loss_agg_mode 的不同，对损失进行不同方式的聚合，例如按 token 平均、按序列平均等。

    valid_token_count = response_mask.sum().clamp_min(1.0)
    clip_fraction = ((clipped_loss > unclipped_loss).float() * response_mask).sum() / valid_token_count
    lower_clip_fraction = (
        (((advantages < 0.0) & (lower_clipped_loss < clipped_candidate)).float() * response_mask).sum()
        / valid_token_count
    )
    approx_kl = _masked_mean(-(current_log_probs - old_log_probs), response_mask)
    return aggregate, {
        "policy_clipfrac": float(clip_fraction.detach().cpu()),
        "policy_clipfrac_lower": float(lower_clip_fraction.detach().cpu()),
        "policy_approx_kl": float(approx_kl.detach().cpu()),
    }


def _compute_value_loss(values, returns, response_mask, cliprange_value: float, loss_agg_mode: str):
    clipped_values = values.clamp(min=returns - cliprange_value, max=returns + cliprange_value)
    squared_error = (values - returns) ** 2
    clipped_squared_error = (clipped_values - returns) ** 2
    masked_loss = squared_error.maximum(clipped_squared_error) * response_mask
    aggregate = _aggregate_loss(masked_loss, response_mask, loss_agg_mode)
    absolute_error = ((returns - values).abs() * response_mask).sum() / response_mask.sum().clamp_min(1.0)
    return aggregate, {"value_abs_error": float(absolute_error.detach().cpu())}


def _compute_kl_penalty(log_probs, ref_log_probs, mode: str = "low_var_kl"):
    if mode in {"kl", "k1"}:
        return log_probs - ref_log_probs
    if mode == "abs":
        return (log_probs - ref_log_probs).abs()
    if mode in {"mse", "k2"}:
        return 0.5 * ((log_probs - ref_log_probs) ** 2)
    if mode in {"low_var_kl", "k3"}:
        diff = log_probs - ref_log_probs
        return (-diff).exp() + diff - 1.0
    raise ValueError("Unsupported KL penalty mode: %s" % mode)


def _infer_hidden_size(backbone_config) -> int:
    for field_name in ("hidden_size", "n_embd"):
        hidden_size = getattr(backbone_config, field_name, None)
        if hidden_size is not None:
            return int(hidden_size)
    raise ValueError("Could not infer hidden size for the local HF critic backbone.")


class HFWorkerBase:
    def __init__(self, model_config):
        self.model_config = model_config
        self.device = get_default_device()

    def _build_model_inputs(self, batch):
        """
        Function:
            - 将 batch 中的 input_ids 和 attention_mask 转换为适合模型输入的张量格式。
        """
        torch, _, _, _ = require_hf_dependencies()
        input_ids = batch_lists_to_tensor(batch.batch["input_ids"], 0, device=self.device, dtype=torch.long, padding_side='right')
        attention_mask = batch_lists_to_tensor(batch.batch["attention_mask"], 0, device=self.device, dtype=torch.long, padding_side='right')
        return input_ids, attention_mask

    def _compute_response_log_probs_and_entropy(self, model, batch):
        """
        Function:
            - 计算给定 batch 中响应部分的对数概率和熵值。
        Returns:
            - response_log_probs: 一个二维张量，shape为 (batch_size, max_response_length)，其中每个元素表示对应位置的响应 token 的对数概率。
            - response_entropy: 一个二维张量，shape为 (batch_size, max_response_length)，其中每个元素表示对应位置的响应 token 的熵值。
        """
        input_ids, attention_mask = self._build_model_inputs(batch)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return extract_response_stats(
            logits=outputs.logits,
            input_ids=input_ids,
            prompt_token_counts=get_prompt_lengths(batch),
            response_token_counts=get_response_lengths(batch),
        )

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
    def __init__(self, model_config, config):
        super().__init__(model_config)
        self.config = config
        self.model = load_causal_lm(model_config).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def compute_log_probs(self, batch) -> LogProbResult:
        with self._no_grad():
            response_log_probs, _ = self._compute_response_log_probs_and_entropy(self.model, batch)
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
    def __init__(self, model_config, config):
        super().__init__(model_config)
        torch, _, _, _ = require_hf_dependencies()
        self.config = config
        self.model = load_causal_lm(model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            betas=tuple(config.betas),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
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
        with self._no_grad():
            response_log_probs, response_entropy = self._compute_response_log_probs_and_entropy(self.model, batch)
        response_lengths = get_response_lengths(batch)
        return LogProbResult(
            log_probs=tensor_to_list_rows(response_log_probs, response_lengths),
            entropy=tensor_to_list_rows(response_entropy, response_lengths),
            metrics={"policy_update_steps": float(self.update_steps)},
        )

    def update(self, batch) -> UpdateResult:
        torch, _, _, _ = require_hf_dependencies()
        metric_history: Dict[str, List[float]] = {}
        self.model.train()

        for _ in range(self.config.ppo_epochs): # 这里的 epochs 表示要重复利用同一批 rollout 数据进行多少轮策略优化更新。
            minibatches = self._iter_minibatches(batch, self.config.ppo_mini_batch_size, self.config.shuffle)
            for minibatch in minibatches: # 每一轮都会将整个 batch 划分成多个 mini-batch，然后对每个 mini-batch 进行一次完整的前向和反向传播，最后更新模型参数。
                microbatches = list(self._iter_microbatches(minibatch, self.config.micro_batch_size))
                if not microbatches:
                    continue

                microbatch_weights: List[float] = [
                    max(get_loss_weight(microbatch, self.config.loss_agg_mode), 1.0) for microbatch in microbatches
                ] # 如果是per_token 的 loss_agg_mode，那么在_compute_policy_loss中计算的损失是micro_batch 上面每个 token 的平均损失，因此为了实现梯度累积这里每一个 micro_batch 的 loss 权重就是这个 micro_batch 中所有 response_mask 中有效 token 的总数。
                total_weight = sum(microbatch_weights)

                self.optimizer.zero_grad() # 在对一个 mini-batch 的所有 micro-batch 进行反向传播之前，先将优化器的梯度缓存清零，以避免梯度累积到之前的 mini-batch 中。
                for microbatch, microbatch_weight in zip(microbatches, microbatch_weights): # 对于每个 mini-batch，我们进一步将其划分成多个 micro-batch，以便在内存受限的情况下进行训练。对于每个 micro-batch，我们计算当前策略的对数概率和熵值，然后根据 PPO 的损失函数计算策略损失，并进行反向传播。最后，我们根据 micro-batch 的权重对损失进行缩放，以确保在更新模型参数时考虑到不同 micro-batch 的重要性。
                    current_log_probs, current_entropy = self._compute_response_log_probs_and_entropy(
                        self.model, microbatch
                    )
                    training_tensors = build_training_tensors(microbatch, self.device)
                    policy_loss, step_metrics = _compute_policy_loss(
                        current_log_probs=current_log_probs,
                        old_log_probs=training_tensors["old_log_probs"],
                        advantages=training_tensors["advantages"],
                        response_mask=training_tensors["response_mask"],
                        clip_ratio=self.config.clip_ratio,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_c=self.config.clip_ratio_c,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )
                    entropy_bonus = _masked_mean(current_entropy, training_tensors["response_mask"])
                    total_loss = policy_loss - (self.config.entropy_coeff * entropy_bonus)
                    step_metrics["policy_entropy"] = float(entropy_bonus.detach().cpu())

                    if self.config.use_kl_loss and "ref_log_probs" in microbatch.batch:
                        kl_penalty = _compute_kl_penalty(
                            current_log_probs,
                            training_tensors["ref_log_probs"],
                            mode="low_var_kl",
                        )
                        kl_mean = _masked_mean(kl_penalty, training_tensors["response_mask"])
                        total_loss = total_loss + (self.config.kl_loss_coef * kl_mean)
                        step_metrics["actor_kl_loss"] = float(kl_mean.detach().cpu())

                    scaled_loss = total_loss * (microbatch_weight / max(total_weight, 1.0))
                    scaled_loss.backward()
                    step_metrics["actor_loss"] = float(total_loss.detach().cpu())
                    for metric_name, metric_value in step_metrics.items():
                        metric_history.setdefault(metric_name, []).append(float(metric_value))

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step() # 对一个mini-batch的所有 micro-batch 进行反向传播并累积梯度后，进行一次优化器步骤来更新模型参数。

        self.update_steps += 1
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
    def __init__(self, model_config, config):
        super().__init__(model_config)
        torch, _, _, _ = require_hf_dependencies()
        self.config = config
        self.backbone = load_backbone_model(model_config, path=model_config.critic_path).to(self.device)
        self.value_head = torch.nn.Linear(_infer_hidden_size(self.backbone.config), 1).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.value_head.parameters()),
            lr=config.lr,
            betas=tuple(config.betas),
            eps=config.eps,
            weight_decay=config.weight_decay,
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

        for _ in range(self.config.ppo_epochs):
            minibatches = self._iter_minibatches(batch, self.config.ppo_mini_batch_size, self.config.shuffle)
            for minibatch in minibatches:
                microbatches = list(self._iter_microbatches(minibatch, self.config.micro_batch_size))
                if not microbatches:
                    continue

                microbatch_weights = [
                    max(get_loss_weight(microbatch, self.config.loss_agg_mode), 1.0) for microbatch in microbatches
                ]
                total_weight = sum(microbatch_weights)

                self.optimizer.zero_grad()
                for microbatch, microbatch_weight in zip(microbatches, microbatch_weights):
                    predicted_values = self._compute_response_values(microbatch)
                    training_tensors = build_training_tensors(microbatch, self.device)
                    value_loss, step_metrics = _compute_value_loss(
                        values=predicted_values,
                        returns=training_tensors["returns"],
                        response_mask=training_tensors["response_mask"],
                        cliprange_value=self.config.cliprange_value,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )
                    scaled_loss = value_loss * (microbatch_weight / max(total_weight, 1.0))
                    scaled_loss.backward()
                    step_metrics["critic_loss"] = float(value_loss.detach().cpu())
                    for metric_name, metric_value in step_metrics.items():
                        metric_history.setdefault(metric_name, []).append(float(metric_value))

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.backbone.parameters()) + list(self.value_head.parameters()),
                        self.config.max_grad_norm,
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
