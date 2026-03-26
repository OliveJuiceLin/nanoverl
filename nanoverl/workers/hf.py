"""Local Hugging Face worker implementations."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from nanoverl.backends.hf import (
    aggregate_weight,
    batch_lists_to_tensor,
    build_response_tensors,
    clone_model_state,
    default_device,
    extract_response_stats,
    load_backbone_model,
    load_causal_lm,
    mean_of_values,
    prompt_lengths,
    require_hf_dependencies,
    response_lengths,
    tensor_to_list_rows,
)
from nanoverl.workers.base import LogProbResult, PolicyWorker, ReferenceWorker, UpdateResult, ValueResult, ValueWorker


def _masked_mean(values, mask):
    torch, _, _, _ = require_hf_dependencies()
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _aggregate_loss(loss_matrix, response_mask, loss_agg_mode: str):
    torch, _, _, _ = require_hf_dependencies()
    if loss_agg_mode == "token-mean":
        return _masked_mean(loss_matrix, response_mask)

    per_sequence = []
    lengths = []
    for row, mask_row in zip(loss_matrix, response_mask):
        valid = row[mask_row > 0]
        if valid.numel() == 0:
            continue
        per_sequence.append(valid.sum())
        lengths.append(float(valid.numel()))
    if not per_sequence:
        return torch.tensor(0.0, device=loss_matrix.device)
    stacked = torch.stack(per_sequence)
    if loss_agg_mode == "seq-mean-token-sum":
        return stacked.mean()
    if loss_agg_mode == "seq-mean-token-mean":
        means = torch.stack([value / length for value, length in zip(per_sequence, lengths)])
        return means.mean()
    if loss_agg_mode == "seq-mean-token-sum-norm":
        scale = max(lengths)
        return stacked.mean() / scale
    raise ValueError("Unsupported loss_agg_mode: %s" % loss_agg_mode)


def _policy_loss(
    current_log_probs,
    old_log_probs,
    advantages,
    response_mask,
    cliprange: float,
    cliprange_low: Optional[float],
    cliprange_high: Optional[float],
    clip_ratio_c: float,
    loss_agg_mode: str,
):
    torch, _, _, _ = require_hf_dependencies()
    clip_low = cliprange if cliprange_low is None else cliprange_low
    clip_high = cliprange if cliprange_high is None else cliprange_high

    ratio = torch.exp((current_log_probs - old_log_probs).clamp(min=-20.0, max=20.0))
    unclipped = -advantages * ratio
    clipped_ratio = ratio.clamp(min=1.0 - clip_low, max=1.0 + clip_high)
    clipped = -advantages * clipped_ratio
    candidate = torch.maximum(unclipped, clipped)
    lower_clipped = torch.minimum(-advantages * clip_ratio_c, candidate)
    final_loss = torch.where(advantages < 0.0, lower_clipped, candidate)
    masked_loss = final_loss * response_mask
    loss = _aggregate_loss(masked_loss, response_mask, loss_agg_mode)

    total_tokens = response_mask.sum().clamp_min(1.0)
    clip_hits = ((clipped > unclipped).float() * response_mask).sum() / total_tokens
    lower_hits = (((advantages < 0.0) & (lower_clipped < candidate)).float() * response_mask).sum() / total_tokens
    approx_kl = _masked_mean(-(current_log_probs - old_log_probs), response_mask)
    return loss, {
        "policy_clipfrac": float(clip_hits.detach().cpu()),
        "policy_clipfrac_lower": float(lower_hits.detach().cpu()),
        "policy_approx_kl": float(approx_kl.detach().cpu()),
    }


def _value_loss(values, returns, response_mask, cliprange_value: float, loss_agg_mode: str):
    torch, _, _, _ = require_hf_dependencies()
    clipped_values = torch.clamp(values, min=returns - cliprange_value, max=returns + cliprange_value)
    unclipped = (values - returns) ** 2
    clipped = (clipped_values - returns) ** 2
    loss_matrix = torch.maximum(unclipped, clipped) * response_mask
    loss = _aggregate_loss(loss_matrix, response_mask, loss_agg_mode)
    denom = response_mask.sum().clamp_min(1.0)
    abs_error = (((returns - values).abs()) * response_mask).sum() / denom
    return loss, {"value_abs_error": float(abs_error.detach().cpu())}


def _kl_penalty_tensor(log_probs, ref_log_probs, mode: str = "low_var_kl"):
    torch, _, _, _ = require_hf_dependencies()
    diff = log_probs - ref_log_probs
    if mode in {"kl", "k1"}:
        return diff
    if mode == "abs":
        return diff.abs()
    if mode in {"mse", "k2"}:
        return 0.5 * (diff ** 2)
    if mode in {"low_var_kl", "k3"}:
        return torch.exp(-diff) + diff - 1.0
    raise ValueError("Unsupported KL penalty mode: %s" % mode)


class _BaseHFWorker:
    def __init__(self, model_config):
        self.model_config = model_config
        self.device = default_device()

    def _full_sequence_tensors(self, batch):
        torch, _, _, _ = require_hf_dependencies()
        input_ids = batch_lists_to_tensor(batch.batch["input_ids"], 0, device=self.device, dtype=torch.long)
        attention_mask = batch_lists_to_tensor(batch.batch["attention_mask"], 0, device=self.device, dtype=torch.long)
        return input_ids, attention_mask


class HFReferenceWorker(_BaseHFWorker, ReferenceWorker):
    def __init__(self, model_config, config):
        super().__init__(model_config)
        self.config = config
        self.model = load_causal_lm(model_config).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def compute_log_probs(self, batch) -> LogProbResult:
        with self._inference_mode():
            log_probs, _ = self._compute_response_stats(batch)
        lengths = response_lengths(batch)
        return LogProbResult(
            log_probs=tensor_to_list_rows(log_probs, lengths),
            metrics={"reference_enabled": 1.0},
        )

    def _compute_response_stats(self, batch):
        input_ids, attention_mask = self._full_sequence_tensors(batch)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return extract_response_stats(outputs.logits, input_ids, prompt_lengths(batch), response_lengths(batch))

    def _inference_mode(self):
        torch, _, _, _ = require_hf_dependencies()
        return torch.no_grad()

    def state_dict(self) -> Dict[str, Any]:
        return {"model_state": clone_model_state(self.model.state_dict())}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("model_state") is not None:
            self.model.load_state_dict(state["model_state"])
        self.model.to(self.device)
        self.model.eval()


class HFPolicyWorker(_BaseHFWorker, PolicyWorker):
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
        self.model.eval()
        with self._inference_mode():
            log_probs, entropy = self._compute_response_stats(batch)
        lengths = response_lengths(batch)
        return LogProbResult(
            log_probs=tensor_to_list_rows(log_probs, lengths),
            entropy=tensor_to_list_rows(entropy, lengths),
            metrics={"policy_update_steps": float(self.update_steps)},
        )

    def _compute_response_stats(self, batch):
        input_ids, attention_mask = self._full_sequence_tensors(batch)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return extract_response_stats(outputs.logits, input_ids, prompt_lengths(batch), response_lengths(batch))

    def _inference_mode(self):
        torch, _, _, _ = require_hf_dependencies()
        return torch.no_grad()

    def _iter_minibatches(self, batch):
        indices = list(range(len(batch)))
        if self.config.shuffle:
            random.shuffle(indices)
        step = max(1, self.config.ppo_mini_batch_size)
        for start in range(0, len(indices), step):
            yield batch.select(indices[start : start + step])

    def _iter_microbatches(self, batch):
        if not self.config.micro_batch_size or self.config.micro_batch_size >= len(batch):
            yield batch
            return
        indices = list(range(len(batch)))
        for start in range(0, len(indices), self.config.micro_batch_size):
            yield batch.select(indices[start : start + self.config.micro_batch_size])

    def update(self, batch) -> UpdateResult:
        torch, _, _, _ = require_hf_dependencies()
        metrics_log: Dict[str, List[float]] = {}
        self.model.train()

        # 外层循环是为了重复利用一个 batch 进行多轮 PPO 更新，
        # 内层minibatch 是最小的更新单元(即每个 minibatch 都会进行一次反向传播)，
        # microbatch 是为了在内存受限的情况下进一步分割 minibatch 的，microbatch 内的样本会被聚合成一个 loss 进行反向传播。
        for _ in range(self.config.ppo_epochs):
            for minibatch in self._iter_minibatches(batch):
                mini_weight_total = 0.0
                micro_batches = list(self._iter_microbatches(minibatch))
                if not micro_batches:
                    continue
                micro_weights: List[float] = []
                for micro_batch in micro_batches:
                    micro_weights.append(max(aggregate_weight(micro_batch, self.config.loss_agg_mode), 1.0))
                total_weight = sum(micro_weights)

                self.optimizer.zero_grad()
                for micro_batch, micro_weight in zip(micro_batches, micro_weights):
                    current_log_probs, entropy = self._compute_response_stats(micro_batch)
                    tensors = build_response_tensors(micro_batch, self.device)
                    loss, micro_metrics = _policy_loss(
                        current_log_probs=current_log_probs,
                        old_log_probs=tensors["old_log_probs"],
                        advantages=tensors["advantages"],
                        response_mask=tensors["response_mask"],
                        cliprange=self.config.clip_ratio,
                        cliprange_low=self.config.clip_ratio_low,
                        cliprange_high=self.config.clip_ratio_high,
                        clip_ratio_c=self.config.clip_ratio_c,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )
                    entropy_mean = _masked_mean(entropy, tensors["response_mask"])
                    total_loss = loss - (self.config.entropy_coeff * entropy_mean)
                    micro_metrics["policy_entropy"] = float(entropy_mean.detach().cpu())

                    if self.config.use_kl_loss and "ref_log_probs" in micro_batch.batch:
                        ref_log_probs = tensors["ref_log_probs"]
                        kl_tensor = _kl_penalty_tensor(current_log_probs, ref_log_probs, mode="low_var_kl")
                        kl_mean = _masked_mean(kl_tensor, tensors["response_mask"])
                        total_loss = total_loss + (self.config.kl_loss_coef * kl_mean)
                        micro_metrics["actor_kl_loss"] = float(kl_mean.detach().cpu())

                    scaled_loss = total_loss * (micro_weight / max(total_weight, 1.0))
                    scaled_loss.backward()
                    micro_metrics["actor_loss"] = float(total_loss.detach().cpu())
                    mini_weight_total += micro_weight
                    for key, value in micro_metrics.items():
                        metrics_log.setdefault(key, []).append(float(value))

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

        self.update_steps += 1
        self.model.eval()
        reduced = {key: mean_of_values(values) for key, values in metrics_log.items()}
        reduced["actor_update_steps"] = float(self.update_steps)
        return UpdateResult(metrics=reduced)

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


class _ValueModelModule:
    def __init__(self, model_config):
        torch, _, _, _ = require_hf_dependencies()
        self.backbone = load_backbone_model(model_config, path=model_config.critic_path)
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size for the local HF critic backbone.")
        self.value_head = torch.nn.Linear(hidden_size, 1)


class HFValueWorker(_BaseHFWorker, ValueWorker):
    def __init__(self, model_config, config):
        super().__init__(model_config)
        torch, _, _, _ = require_hf_dependencies()
        self.config = config
        model_module = _ValueModelModule(model_config)
        self.backbone = model_module.backbone.to(self.device)
        self.value_head = model_module.value_head.to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.value_head.parameters()),
            lr=config.lr,
            betas=tuple(config.betas),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        self.update_steps = 0

    def _forward_values(self, batch):
        input_ids, attention_mask = self._full_sequence_tensors(batch)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        values = self.value_head(hidden_states).squeeze(-1)
        prompt_lens = prompt_lengths(batch)
        response_lens = response_lengths(batch)
        torch, _, _, _ = require_hf_dependencies()
        max_response_len = max(response_lens, default=0)
        response_values = values.new_zeros((values.shape[0], max_response_len))
        for row_index, (prompt_len, response_len) in enumerate(zip(prompt_lens, response_lens)):
            if response_len <= 0:
                continue
            start = prompt_len
            end = start + response_len
            response_values[row_index, :response_len] = values[row_index, start:end]
        return response_values

    def compute_values(self, batch) -> ValueResult:
        self.backbone.eval()
        self.value_head.eval()
        with self._inference_mode():
            values = self._forward_values(batch)
        return ValueResult(
            values=tensor_to_list_rows(values, response_lengths(batch)),
            metrics={"critic_update_steps": float(self.update_steps)},
        )

    def _inference_mode(self):
        torch, _, _, _ = require_hf_dependencies()
        return torch.no_grad()

    def _iter_minibatches(self, batch):
        indices = list(range(len(batch)))
        if self.config.shuffle:
            random.shuffle(indices)
        step = max(1, self.config.ppo_mini_batch_size)
        for start in range(0, len(indices), step):
            yield batch.select(indices[start : start + step])

    def _iter_microbatches(self, batch):
        if not self.config.micro_batch_size or self.config.micro_batch_size >= len(batch):
            yield batch
            return
        indices = list(range(len(batch)))
        for start in range(0, len(indices), self.config.micro_batch_size):
            yield batch.select(indices[start : start + self.config.micro_batch_size])

    def update(self, batch) -> UpdateResult:
        torch, _, _, _ = require_hf_dependencies()
        metrics_log: Dict[str, List[float]] = {}
        self.backbone.train()
        self.value_head.train()

        for _ in range(self.config.ppo_epochs):
            for minibatch in self._iter_minibatches(batch):
                micro_batches = list(self._iter_microbatches(minibatch))
                if not micro_batches:
                    continue
                micro_weights = [max(aggregate_weight(micro_batch, self.config.loss_agg_mode), 1.0) for micro_batch in micro_batches]
                total_weight = sum(micro_weights)
                self.optimizer.zero_grad()
                for micro_batch, micro_weight in zip(micro_batches, micro_weights):
                    value_predictions = self._forward_values(micro_batch)
                    tensors = build_response_tensors(micro_batch, self.device)
                    loss, micro_metrics = _value_loss(
                        values=value_predictions,
                        returns=tensors["returns"],
                        response_mask=tensors["response_mask"],
                        cliprange_value=self.config.cliprange_value,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )
                    scaled_loss = loss * (micro_weight / max(total_weight, 1.0))
                    scaled_loss.backward()
                    micro_metrics["critic_loss"] = float(loss.detach().cpu())
                    for key, value in micro_metrics.items():
                        metrics_log.setdefault(key, []).append(float(value))

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.backbone.parameters()) + list(self.value_head.parameters()),
                        self.config.max_grad_norm,
                    )
                self.optimizer.step()

        self.update_steps += 1
        self.backbone.eval()
        self.value_head.eval()
        reduced = {key: mean_of_values(values) for key, values in metrics_log.items()}
        reduced["critic_update_steps"] = float(self.update_steps)
        return UpdateResult(metrics=reduced)

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
