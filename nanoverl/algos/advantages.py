"""Advantage estimators used by PPO- and GRPO-like trainers."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple


Matrix = Sequence[Sequence[float]]


def _zeros_like(matrix: Matrix) -> List[List[float]]:
    return [[0.0 for _ in row] for row in matrix]


def compute_gae_advantages(
    token_level_rewards: Matrix,
    values: Matrix,
    response_mask: Sequence[Sequence[int]],
    gamma: float,
    lam: float,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Standard GAE over token-level rewards."""

    advantages = _zeros_like(token_level_rewards)
    returns = _zeros_like(token_level_rewards)
    for row_index, (rewards_row, values_row, mask_row) in enumerate(zip(token_level_rewards, values, response_mask)):
        next_advantage = 0.0
        next_value = 0.0
        for token_index in reversed(range(len(rewards_row))):
            if not mask_row[token_index]:
                continue
            delta = rewards_row[token_index] + gamma * next_value - values_row[token_index]
            next_advantage = delta + gamma * lam * next_advantage
            advantages[row_index][token_index] = next_advantage
            returns[row_index][token_index] = next_advantage + values_row[token_index]
            next_value = values_row[token_index]
    return advantages, returns


def compute_grpo_advantages(
    token_level_rewards: Matrix,
    response_mask: Sequence[Sequence[int]],
    group_ids: Sequence[str],
    normalize_by_std: bool = True,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Group-relative outcome advantages used by GRPO."""

    sequence_scores = [] # 每个响应文本的总奖励分数，计算方式是将 token_level_rewards 中对应响应文本的 token 的奖励分数加总起来（只加总 response_mask 中标记为 1 的 token），得到一个单一的分数，代表整个响应文本的奖励水平。
    for rewards_row, mask_row in zip(token_level_rewards, response_mask):
        sequence_scores.append(sum(value for value, mask in zip(rewards_row, mask_row) if mask))

    grouped: Dict[str, List[float]] = {} # 根据 group_ids 将 sequence_scores 分组，得到一个字典 grouped，其中键是 group_id，值是一个列表，包含了属于该 group_id 的所有 sequence_scores。这一步的目的是为了后续计算每个 group 内的平均分数和标准差，以便进行归一化处理。
    for group_id, score in zip(group_ids, sequence_scores):
        grouped.setdefault(group_id, []).append(score)

    grouped_stats: Dict[str, Tuple[float, float]] = {}
    for group_id, scores in grouped.items():
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / max(len(scores), 1)
        std = math.sqrt(variance)
        grouped_stats[group_id] = (mean, std)

    advantages = _zeros_like(token_level_rewards)
    returns = _zeros_like(token_level_rewards)
    for row_index, (group_id, score, mask_row) in enumerate(zip(group_ids, sequence_scores, response_mask)):
        mean, std = grouped_stats[group_id]
        normalized = score - mean
        if normalize_by_std and std > 1e-8:
            normalized /= std
        for token_index, mask in enumerate(mask_row):
            if mask:
                advantages[row_index][token_index] = normalized
                returns[row_index][token_index] = normalized
    return advantages, returns


__all__ = ["compute_gae_advantages", "compute_grpo_advantages"]
