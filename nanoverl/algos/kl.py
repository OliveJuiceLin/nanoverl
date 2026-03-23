"""KL helpers for reward shaping and actor regularization."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


Matrix = Sequence[Sequence[float]]


def compute_kl_penalty(log_probs: Matrix, ref_log_probs: Matrix, mode: str = "kl") -> List[List[float]]:
    penalties: List[List[float]] = []
    for row, ref_row in zip(log_probs, ref_log_probs):
        penalty_row: List[float] = []
        for log_prob, ref_log_prob in zip(row, ref_row):
            diff = log_prob - ref_log_prob
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
    penalties = compute_kl_penalty(old_log_probs, ref_log_probs, mode=mode)
    rewards: List[List[float]] = []
    masked_penalties: List[float] = []
    for score_row, penalty_row, mask_row in zip(token_level_scores, penalties, response_mask):
        reward_row: List[float] = []
        for score, penalty, mask in zip(score_row, penalty_row, mask_row):
            if mask:
                reward_row.append(score - beta * penalty)
                masked_penalties.append(penalty)
            else:
                reward_row.append(score)
        rewards.append(reward_row)
    mean_penalty = sum(masked_penalties) / max(len(masked_penalties), 1)
    return rewards, mean_penalty


__all__ = ["apply_kl_penalty", "compute_kl_penalty"]
