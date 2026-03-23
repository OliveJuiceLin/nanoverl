"""Reward interfaces."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from nanoverl.core.batch import RLBatch

RewardFn = Callable[[str, str, Mapping[str, Any]], Any]


@dataclass
class RewardResult:
    token_level_scores: List[List[float]]
    extra: Dict[str, List[Any]] = field(default_factory=dict)


def exact_match_reward(prompt: str, response: str, sample: Mapping[str, Any]) -> float:
    expected = sample.get("expected_response")
    if expected is None:
        reward_model = sample.get("reward_model") or {}
        expected = reward_model.get("ground_truth")
    if expected is None:
        return 0.0
    return 1.0 if str(response).strip() == str(expected).strip() else 0.0


def load_reward_function(path: Optional[str], function_name: str) -> RewardFn:
    if not path:
        return exact_match_reward
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load reward function module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reward_fn = getattr(module, function_name)
    if not callable(reward_fn):
        raise TypeError("Reward function must be callable.")
    return reward_fn


class RewardManager:
    """Computes per-sample rewards and expands them to token-level terminal rewards."""

    def __init__(self, reward_fn: RewardFn):
        self.reward_fn = reward_fn

    def compute(self, batch: RLBatch) -> RewardResult:
        prompt_texts = batch.non_tensor.get("prompt_text") or batch.non_tensor.get("prompt")
        response_texts = batch.non_tensor.get("response_text")
        response_mask = batch.batch.get("response_mask")
        if prompt_texts is None or response_texts is None or response_mask is None:
            raise ValueError("RewardManager requires prompt_text, response_text, and response_mask.")

        token_level_scores: List[List[float]] = []
        extras: Dict[str, List[Any]] = {}
        for index in range(len(batch)):
            row = batch.row(index)
            result = self.reward_fn(str(prompt_texts[index]), str(response_texts[index]), row)
            extra_payload: Dict[str, Any] = {}
            score = result
            if isinstance(result, dict):
                score = result.get("score", 0.0)
                extra_payload = {key: value for key, value in result.items() if key != "score"}
            mask_row = response_mask[index]
            rewards = [0.0 for _ in mask_row]
            valid_length = sum(1 for keep in mask_row if keep)
            if valid_length > 0:
                rewards[valid_length - 1] = float(score)
            token_level_scores.append(rewards)
            for key, value in extra_payload.items():
                extras.setdefault(key, []).append(value)
        return RewardResult(token_level_scores=token_level_scores, extra=extras)


__all__ = ["RewardManager", "RewardResult", "RewardFn", "exact_match_reward", "load_reward_function"]
