"""High-signal training metrics for RL debugging."""

from __future__ import annotations

from math import isfinite
from typing import Dict, List

from nanoverl.core.batch import RLBatch


def _row_sums(matrix: List[List[float]]) -> List[float]:
    return [sum(row) for row in matrix]


def _stats(prefix: str, values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "%s/mean" % prefix: sum(values) / len(values),
        "%s/max" % prefix: max(values),
        "%s/min" % prefix: min(values),
    }


def compute_data_metrics(
    batch: RLBatch,
    use_critic: bool = True,
    response_length_limit: int | None = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    response_lengths = [sum(mask_row) for mask_row in batch.batch.get("response_mask", [])]
    prompt_lengths = [len(prompt_row) for prompt_row in batch.batch.get("prompts", [])]
    if "token_level_scores" in batch.batch:
        metrics.update(_stats("reward/score", _row_sums(batch.batch["token_level_scores"])))
    if "token_level_rewards" in batch.batch:
        metrics.update(_stats("reward/shaped", _row_sums(batch.batch["token_level_rewards"])))
    if "advantages" in batch.batch:
        valid_advantages = []
        for adv_row, mask_row in zip(batch.batch["advantages"], batch.batch["response_mask"]):
            valid_advantages.extend(value for value, keep in zip(adv_row, mask_row) if keep)
        metrics.update(_stats("advantage", valid_advantages))
    if "returns" in batch.batch:
        valid_returns = []
        for return_row, mask_row in zip(batch.batch["returns"], batch.batch["response_mask"]):
            valid_returns.extend(value for value, keep in zip(return_row, mask_row) if keep)
        metrics.update(_stats("return", valid_returns))
    if use_critic and "values" in batch.batch:
        valid_values = []
        for value_row, mask_row in zip(batch.batch["values"], batch.batch["response_mask"]):
            valid_values.extend(value for value, keep in zip(value_row, mask_row) if keep)
        metrics.update(_stats("value", valid_values))
        if "returns" in batch.batch and valid_values:
            valid_returns = []
            for return_row, mask_row in zip(batch.batch["returns"], batch.batch["response_mask"]):
                valid_returns.extend(value for value, keep in zip(return_row, mask_row) if keep)
            if valid_returns:
                return_mean = sum(valid_returns) / len(valid_returns)
                return_variance = sum((return_value - return_mean) ** 2 for return_value in valid_returns) / len(valid_returns)
                if return_variance > 0.0:
                    explained_variance = 1.0 - (
                        sum((return_value - value) ** 2 for return_value, value in zip(valid_returns, valid_values))
                        / len(valid_returns)
                    ) / return_variance
                    if isfinite(explained_variance):
                        metrics["value/explained_variance"] = explained_variance
    metrics.update(_stats("response_length", [float(value) for value in response_lengths]))
    metrics.update(_stats("prompt_length", [float(value) for value in prompt_lengths]))
    if response_lengths:
        aborted = sum(1 for value in response_lengths if value == 0)
        metrics["response/aborted_ratio"] = aborted / len(response_lengths)
        non_aborted_lengths = [float(value) for value in response_lengths if value > 0]
        if non_aborted_lengths:
            metrics["response_length_non_aborted/mean"] = sum(non_aborted_lengths) / len(non_aborted_lengths)
        if response_length_limit is not None and response_length_limit > 0:
            clipped = sum(1 for value in response_lengths if value >= response_length_limit)
            metrics["response_length/clip_ratio"] = clipped / len(response_lengths)
    return metrics


def compute_timing_metrics(timing: Dict[str, float]) -> Dict[str, float]:
    return {"timing/%s" % key: value for key, value in timing.items()}


def compute_throughput_metrics(batch: RLBatch, step_time: float, world_size: int = 1) -> Dict[str, float]:
    total_tokens = sum(len(prompt) + len(response) for prompt, response in zip(batch.batch["prompts"], batch.batch["responses"]))
    return {
        "perf/total_tokens": float(total_tokens),
        "perf/step_time": float(step_time),
        "perf/tokens_per_second_per_worker": total_tokens / max(step_time * world_size, 1e-8),
    }


__all__ = ["compute_data_metrics", "compute_timing_metrics", "compute_throughput_metrics"]
