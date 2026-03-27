"""Validation helpers."""

from __future__ import annotations

from numbers import Number
from typing import Dict, List, Mapping, Sequence


def summarize_validation(
    scores: Sequence[float],
    data_sources: Sequence[str],
    reward_extras: Mapping[str, Sequence[object]] | None = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not scores:
        return metrics
    metrics["val/reward_mean"] = sum(scores) / len(scores)
    grouped: Dict[str, List[float]] = {}
    for data_source, score in zip(data_sources, scores):
        grouped.setdefault(data_source, []).append(score)
    for data_source, grouped_scores in grouped.items():
        metrics["val/%s/reward_mean" % data_source] = sum(grouped_scores) / len(grouped_scores)
    if reward_extras:
        for key, values in reward_extras.items():
            numeric_values = [float(value) for value in values if isinstance(value, Number)]
            if numeric_values:
                metrics["val/extra/%s_mean" % key] = sum(numeric_values) / len(numeric_values)
    return metrics


__all__ = ["summarize_validation"]
