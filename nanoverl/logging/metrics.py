"""High-signal training metrics for RL debugging."""

from __future__ import annotations

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


def compute_data_metrics(batch: RLBatch, use_critic: bool = True) -> Dict[str, float]:
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
    metrics.update(_stats("response_length", [float(value) for value in response_lengths]))
    metrics.update(_stats("prompt_length", [float(value) for value in prompt_lengths]))
    if response_lengths:
        aborted = sum(1 for value in response_lengths if value == 0)
        metrics["response/aborted_ratio"] = aborted / len(response_lengths)
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
