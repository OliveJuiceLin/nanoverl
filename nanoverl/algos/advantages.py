"""Advantage estimators used by built-in RL algorithms."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Sequence, Tuple


Matrix = Sequence[Sequence[float]]
AdvantageEstimatorFn = Callable[[Any, Any], Tuple[List[List[float]], List[List[float]]]]
_ADVANTAGE_ESTIMATOR_REGISTRY: Dict[str, AdvantageEstimatorFn] = {}


def register_advantage_estimator(name: str) -> Callable[[AdvantageEstimatorFn], AdvantageEstimatorFn]:
    def decorator(fn: AdvantageEstimatorFn) -> AdvantageEstimatorFn:
        if name in _ADVANTAGE_ESTIMATOR_REGISTRY and _ADVANTAGE_ESTIMATOR_REGISTRY[name] is not fn:
            raise ValueError("Advantage estimator already registered: %s" % name)
        _ADVANTAGE_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_advantage_estimator(name: str) -> AdvantageEstimatorFn:
    try:
        return _ADVANTAGE_ESTIMATOR_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            "Unsupported advantage estimator: %s. Available estimators: %s"
            % (name, sorted(_ADVANTAGE_ESTIMATOR_REGISTRY))
        ) from exc


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


def compute_rloo_advantages(
    token_level_rewards: Matrix,
    response_mask: Sequence[Sequence[int]],
    group_ids: Sequence[str],
) -> Tuple[List[List[float]], List[List[float]]]:
    """Leave-one-out outcome advantages for grouped rollouts."""

    sequence_scores = []
    for rewards_row, mask_row in zip(token_level_rewards, response_mask):
        sequence_scores.append(sum(value for value, mask in zip(rewards_row, mask_row) if mask))

    grouped_indices: Dict[str, List[int]] = {}
    for row_index, group_id in enumerate(group_ids):
        grouped_indices.setdefault(group_id, []).append(row_index)

    advantages = _zeros_like(token_level_rewards)
    returns = _zeros_like(token_level_rewards)
    for group_id, indices in grouped_indices.items():
        if len(indices) < 2:
            raise ValueError("RLOO requires at least two responses per group: %s" % group_id)
        group_total = sum(sequence_scores[index] for index in indices)
        for row_index in indices:
            score = sequence_scores[row_index]
            baseline = (group_total - score) / (len(indices) - 1)
            advantage = score - baseline
            for token_index, mask in enumerate(response_mask[row_index]):
                if mask:
                    advantages[row_index][token_index] = advantage
                    returns[row_index][token_index] = advantage
    return advantages, returns


@register_advantage_estimator("gae")
def estimate_gae_advantages(batch: Any, algorithm_config: Any) -> Tuple[List[List[float]], List[List[float]]]:
    return compute_gae_advantages(
        token_level_rewards=batch.batch["token_level_rewards"],
        values=batch.batch["values"],
        response_mask=batch.batch["response_mask"],
        gamma=algorithm_config.gamma,
        lam=algorithm_config.lam,
    )


@register_advantage_estimator("grpo")
def estimate_grpo_advantages(batch: Any, algorithm_config: Any) -> Tuple[List[List[float]], List[List[float]]]:
    return compute_grpo_advantages(
        token_level_rewards=batch.batch["token_level_rewards"],
        response_mask=batch.batch["response_mask"],
        group_ids=batch.non_tensor["uid"],
        normalize_by_std=algorithm_config.norm_adv_by_std_in_grpo,
    )


@register_advantage_estimator("rloo")
def estimate_rloo_advantages(batch: Any, algorithm_config: Any) -> Tuple[List[List[float]], List[List[float]]]:
    return compute_rloo_advantages(
        token_level_rewards=batch.batch["token_level_rewards"],
        response_mask=batch.batch["response_mask"],
        group_ids=batch.non_tensor["uid"],
    )


__all__ = [
    "compute_gae_advantages",
    "compute_grpo_advantages",
    "compute_rloo_advantages",
    "estimate_gae_advantages",
    "estimate_grpo_advantages",
    "estimate_rloo_advantages",
    "get_advantage_estimator",
    "register_advantage_estimator",
]
