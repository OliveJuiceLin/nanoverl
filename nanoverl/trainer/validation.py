"""Validation helpers."""

from __future__ import annotations

from typing import Dict, List, Sequence


def summarize_validation(scores: Sequence[float], data_sources: Sequence[str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not scores:
        return metrics
    metrics["val/reward_mean"] = sum(scores) / len(scores)
    grouped: Dict[str, List[float]] = {}
    for data_source, score in zip(data_sources, scores):
        grouped.setdefault(data_source, []).append(score)
    for data_source, grouped_scores in grouped.items():
        metrics["val/%s/reward_mean" % data_source] = sum(grouped_scores) / len(grouped_scores)
    return metrics


__all__ = ["summarize_validation"]
