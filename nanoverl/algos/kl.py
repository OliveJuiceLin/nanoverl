"""KL helpers for reward shaping and actor regularization."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


Matrix = Sequence[Sequence[float]]


def _is_tensor_like(value) -> bool:
    return hasattr(value, "detach") and hasattr(value, "shape")


def compute_kl_penalty(log_probs: Matrix, ref_log_probs: Matrix, mode: str = "kl") -> List[List[float]]:
    if _is_tensor_like(log_probs):
        diff = log_probs - ref_log_probs
        if mode in {"kl", "k1"}:
            return diff
        if mode == "abs":
            return diff.abs()
        if mode in {"mse", "k2"}:
            return 0.5 * (diff ** 2)
        if mode in {"low_var_kl", "k3"}:
            return (-diff).exp() + diff - 1.0
        raise ValueError("Unsupported KL penalty mode: %s" % mode)

    penalties: List[List[float]] = []
    for row, ref_row in zip(log_probs, ref_log_probs):
        penalty_row: List[float] = []
        for log_prob, ref_log_prob in zip(row, ref_row):
            diff: float = log_prob - ref_log_prob
            if mode in {"kl", "k1"}:
                penalty = diff
            elif mode == "abs":
                penalty = abs(diff)
            elif mode in {"mse", "k2"}:
                penalty = 0.5 * diff * diff
            elif mode in {"low_var_kl", "k3"}:
                penalty = math.exp(-diff) + diff - 1.0
            else:
                raise ValueError("Unsupported KL penalty mode: %s" % mode)
            penalty_row.append(penalty)
        penalties.append(penalty_row)
    return penalties


def apply_kl_penalty(
    token_level_scores: Matrix,
    old_log_probs: Matrix,
    ref_log_probs: Matrix,
    response_mask: Sequence[Sequence[int]],
    beta: float,
    mode: str = "kl",
) -> Tuple[List[List[float]], float]:
    """
    Function:
    """
    if _is_tensor_like(token_level_scores):
        penalties = compute_kl_penalty(old_log_probs, ref_log_probs, mode=mode)
        rewards = token_level_scores.where(response_mask <= 0, token_level_scores - beta * penalties)
        valid_penalties = penalties[response_mask > 0]
        mean_penalty = 0.0
        if valid_penalties.numel() > 0:
            mean_penalty = float(valid_penalties.mean().detach().cpu())
        return rewards, mean_penalty

    penalties: List[List[float]] = compute_kl_penalty(old_log_probs, ref_log_probs, mode=mode)
    rewards: List[List[float]] = []
    masked_penalties: List[float] = []
    for score_row, penalty_row, mask_row in zip(token_level_scores, penalties, response_mask):
        reward_row: List[float] = []
        for score, penalty, mask in zip(score_row, penalty_row, mask_row):
            if mask:
                reward_row.append(score - beta * penalty) # 原始的得分减去 KL 惩罚项，得到新的奖励值。KL 惩罚项是根据当前策略的 log_probs 和参考模型的 log_probs 计算得到的，表示当前策略与参考模型之间的差异程度。通过减去 KL 惩罚项，可以鼓励当前策略在奖励 shaping 中考虑与参考模型的一致性，从而实现更稳定和有效的训练。
                masked_penalties.append(penalty)
            else:
                reward_row.append(score)
        rewards.append(reward_row)
    mean_penalty: float = sum(masked_penalties) / max(len(masked_penalties), 1)
    return rewards, mean_penalty


__all__ = ["apply_kl_penalty", "compute_kl_penalty"]
