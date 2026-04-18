"""PPO loss functions and shared masked aggregations."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


Matrix = Sequence[Sequence[float]]


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "shape")


def _masked_values(values: Matrix, mask: Sequence[Sequence[int]]) -> List[float]:
    flattened: List[float] = []
    for row, mask_row in zip(values, mask):
        for value, keep in zip(row, mask_row):
            if keep:
                flattened.append(value)
    return flattened


def masked_mean(values: Matrix, mask: Sequence[Sequence[int]]) -> float:
    if _is_tensor_like(values):
        return (values * mask).sum() / mask.sum().clamp_min(1.0)
    flattened = _masked_values(values, mask)
    return sum(flattened) / max(len(flattened), 1)


def aggregate_loss(
    loss_matrix: Matrix,
    response_mask: Sequence[Sequence[int]],
    loss_agg_mode: str = "token-mean",
    loss_scale_factor: Optional[int] = None,
) -> float:
    if loss_agg_mode == "token-mean":
        # 每个 token 的地位是平等的。如果批次中有一条序列特别长，另一条特别短，长序列会对总 Loss 产生更大的主导影响。
        return masked_mean(loss_matrix, response_mask)

    if _is_tensor_like(loss_matrix):
        import torch

        per_sequence_losses = []
        per_sequence_lengths: List[float] = []
        for row, mask_row in zip(loss_matrix, response_mask):
            valid = row[mask_row > 0]
            if valid.numel() == 0:
                continue
            per_sequence_losses.append(valid.sum())
            per_sequence_lengths.append(float(valid.numel()))
        if not per_sequence_losses:
            return loss_matrix.new_tensor(0.0)
        stacked_losses = torch.stack(per_sequence_losses) # shape [num_sequences]
        if loss_agg_mode == "seq-mean-token-sum":
            # 首先对单条序列内部的所有有效 token 损失求和 (Sum)，得到该序列的总损失；然后将所有序列的总损失相加，除以序列数量 $N$ 做平均 (Mean)。
            # 把“一条句子”当做优化的基本单位。因为是单句求和，生成的回复越长，该样本的损失绝对值通常越大，对应的梯度也越大。这可能会促使模型倾向于调整长句子的生成策略。
            return stacked_losses.mean()
        if loss_agg_mode == "seq-mean-token-mean":
            # 首先对单条序列内部的所有有效 token 损失做内部平均 (Mean) $\big( \frac{1}{L_i} \sum l_{i,j} \big)$，得到该句子的平均 token 损失；然后再将这 $N$ 个平均值相加，除以序列数量 $N$ 做外部平均 (Mean)。
            # 也是把“一条句子”当做优化的基本单位，但在求平均之前先对每条句子内部的 token loss 求平均。这样可以在一定程度上缓解长句子对总 Loss 的主导影响，因为每条句子的损失都是基于其平均 token loss 来计算的，而不是总和。
            means = torch.stack([loss / length for loss, length in zip(per_sequence_losses, per_sequence_lengths)])
            return means.mean()
        if loss_agg_mode == "seq-mean-token-sum-norm":
            scale = float(loss_scale_factor or max(per_sequence_lengths))
            return stacked_losses.mean() / scale
        raise ValueError("Unsupported loss_agg_mode: %s" % loss_agg_mode)

    per_sequence_losses: List[float] = []
    per_sequence_lengths: List[int] = []
    for row, mask_row in zip(loss_matrix, response_mask):
        valid = [value for value, keep in zip(row, mask_row) if keep]
        if not valid:
            continue
        per_sequence_losses.append(sum(valid))
        per_sequence_lengths.append(len(valid))

    if not per_sequence_losses:
        return 0.0

    if loss_agg_mode == "seq-mean-token-sum":
        return sum(per_sequence_losses) / len(per_sequence_losses)
    if loss_agg_mode == "seq-mean-token-mean":
        means = [loss / length for loss, length in zip(per_sequence_losses, per_sequence_lengths)]
        return sum(means) / len(means)
    if loss_agg_mode == "seq-mean-token-sum-norm":
        scale = float(loss_scale_factor or max(per_sequence_lengths))
        return (sum(per_sequence_losses) / len(per_sequence_losses)) / scale
    raise ValueError("Unsupported loss_agg_mode: %s" % loss_agg_mode)


def compute_policy_loss(
    old_log_probs: Matrix,
    log_probs: Matrix,
    advantages: Matrix,
    response_mask: Sequence[Sequence[int]],
    cliprange: float,
    cliprange_low: Optional[float] = None,
    cliprange_high: Optional[float] = None,
    clip_ratio_c: float = 3.0,
    loss_agg_mode: str = "token-mean",
    loss_scale_factor: Optional[int] = None,
) -> Tuple[float, Dict[str, float]]:
    clip_low = cliprange if cliprange_low is None else cliprange_low
    clip_high = cliprange if cliprange_high is None else cliprange_high

    if _is_tensor_like(old_log_probs):
        ratio = (log_probs - old_log_probs).exp()
        unclipped = -advantages * ratio
        clipped_ratio = ratio.clamp(min=1.0 - clip_low, max=1.0 + clip_high)
        clipped = -advantages * clipped_ratio
        candidate = unclipped.maximum(clipped)
        lower_clipped = (-advantages * clip_ratio_c).minimum(candidate)
        final_loss = candidate.where(advantages >= 0.0, lower_clipped)
        valid_token_count = response_mask.sum().clamp_min(1.0)
        clip_fraction = ((clipped > unclipped).float() * response_mask).sum() / valid_token_count
        lower_clip_fraction = (
            (((advantages < 0.0) & (lower_clipped < candidate)).float() * response_mask).sum()
            / valid_token_count
        )
        approx_kl = masked_mean(-(log_probs - old_log_probs), response_mask)
        loss = aggregate_loss(final_loss, response_mask, loss_agg_mode, loss_scale_factor)
        return loss, {
            "policy_clipfrac": float(clip_fraction.detach().cpu()), # 发生 PPO 标准“截断 (Clip)”的 token 占整体有效 token 的比例。
            "policy_clipfrac_lower": float(lower_clip_fraction.detach().cpu()), # 发生“下限截断”的 token 占整体有效 token 的比例。
            "policy_approx_kl": float(approx_kl.detach().cpu()), # 更新后的新策略（log_probs）相对于之前采样时的旧策略（old_log_probs）发生了多大变化的近似度量。
        }

    loss_matrix: List[List[float]] = []
    approx_kl_matrix: List[List[float]] = []
    clip_hits = 0
    lower_clip_hits = 0
    total_tokens = 0

    for old_row, new_row, adv_row, mask_row in zip(old_log_probs, log_probs, advantages, response_mask):
        loss_row: List[float] = []
        kl_row: List[float] = []
        for old_log_prob, log_prob, advantage, keep in zip(old_row, new_row, adv_row, mask_row):
            if not keep:
                loss_row.append(0.0)
                kl_row.append(0.0)
                continue
            total_tokens += 1
            ratio = math.exp(max(min(log_prob - old_log_prob, 20.0), -20.0))
            unclipped = -advantage * ratio
            clipped_ratio = min(max(ratio, 1.0 - clip_low), 1.0 + clip_high)
            clipped = -advantage * clipped_ratio
            candidate = max(unclipped, clipped)
            if clipped > unclipped:
                clip_hits += 1
            lower_clipped = min(-advantage * clip_ratio_c, candidate)
            if advantage < 0.0 and lower_clipped < candidate:
                lower_clip_hits += 1
            final_loss = lower_clipped if advantage < 0.0 else candidate
            loss_row.append(final_loss)
            kl_row.append(-(log_prob - old_log_prob))
        loss_matrix.append(loss_row)
        approx_kl_matrix.append(kl_row)

    loss = aggregate_loss(loss_matrix, response_mask, loss_agg_mode, loss_scale_factor)
    metrics = {
        "policy_clipfrac": clip_hits / max(total_tokens, 1),
        "policy_clipfrac_lower": lower_clip_hits / max(total_tokens, 1),
        "policy_approx_kl": masked_mean(approx_kl_matrix, response_mask),
    }
    return loss, metrics


def compute_value_loss(
    values: Matrix,
    returns: Matrix,
    response_mask: Sequence[Sequence[int]],
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
) -> Tuple[float, Dict[str, float]]:
    if _is_tensor_like(values):
        clipped_values = values.clamp(min=returns - cliprange_value, max=returns + cliprange_value)
        squared_error = (values - returns) ** 2
        clipped_squared_error = (clipped_values - returns) ** 2
        loss_matrix = squared_error.maximum(clipped_squared_error)
        loss = aggregate_loss(loss_matrix, response_mask, loss_agg_mode)
        abs_error = masked_mean((returns - values).abs(), response_mask)
        return loss, {"value_abs_error": float(abs_error.detach().cpu())}

    loss_matrix: List[List[float]] = []
    abs_errors: List[float] = []
    for value_row, return_row, mask_row in zip(values, returns, response_mask):
        row_loss: List[float] = []
        for value, target, keep in zip(value_row, return_row, mask_row):
            if not keep:
                row_loss.append(0.0)
                continue
            clipped_value = min(max(value, target - cliprange_value), target + cliprange_value)
            unclipped = (value - target) ** 2
            clipped = (clipped_value - target) ** 2
            row_loss.append(max(unclipped, clipped))
            abs_errors.append(abs(target - value))
        loss_matrix.append(row_loss)
    loss = aggregate_loss(loss_matrix, response_mask, loss_agg_mode)
    mean_abs_error = sum(abs_errors) / max(len(abs_errors), 1)
    return loss, {"value_abs_error": mean_abs_error}


__all__ = ["aggregate_loss", "compute_policy_loss", "compute_value_loss", "masked_mean"]
