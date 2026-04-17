"""A small batch container tailored to RL training."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _is_sequence(value: Any) -> bool:
    return hasattr(value, "__len__") and hasattr(value, "__getitem__") and not isinstance(value, (str, bytes))


def _ensure_sequence(value: Any) -> Sequence[Any]:
    """
    Logic:
    - If the value is already a sequence (like list, tuple), return it as is
    - If not, raise an error since batch fields must be indexable sequences
    """
    if not _is_sequence(value):
        raise TypeError("Batch fields must be indexable sequences.")
    return value


def _copy_item(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, tuple):
        return tuple(value)
    return value


def _select_field(value: Any, indices: Sequence[int]) -> Any:
    sequence = _ensure_sequence(value)
    selected = [_copy_item(sequence[index]) for index in indices]
    if isinstance(value, tuple):
        return tuple(selected)
    return selected


def _field_length(value: Any) -> int:
    sequence = _ensure_sequence(value)
    return len(sequence)


def _range_for_repeat(length: int, repeat_times: int, interleave: bool) -> List[int]:
    if interleave:
        indices: List[int] = []
        for index in range(length):
            indices.extend([index] * repeat_times)
        return indices
    indices = list(range(length))
    return indices * repeat_times


@dataclass
class RLBatch:
    """
    A deliberately small batch protocol for RL training.
    所有的操作均不修改原始数据，而是返回新的 RLBatch 实例，保证数据不可变性和安全性。
    """

    batch: Dict[str, Any] = field(default_factory=dict) # 张量数据，需要梯度/设备转移
    non_tensor: Dict[str, Any] = field(default_factory=dict) # 非张量数据，不需要梯度/设备转移
    meta: Dict[str, Any] = field(default_factory=dict) # 其他元信息，不参与批处理操作

    def __len__(self) -> int:
        for value in self.batch.values():
            return _field_length(value)
        for value in self.non_tensor.values():
            return _field_length(value)
        return 0

    def clone(self) -> "RLBatch":
        return RLBatch(
            batch=copy.deepcopy(self.batch),
            non_tensor=copy.deepcopy(self.non_tensor),
            meta=copy.deepcopy(self.meta),
        )

    def select(self, indices: Sequence[int]) -> "RLBatch":
        """
        按行选择数据
        Example:
            batch = RLBatch(
                batch={"scores": [10, 20, 30, 40]},
                non_tensor={"ids": ["a", "b", "c", "d"]}
            )
            subset = batch.select([0, 2, 3])
            # subset.batch["scores"] = [10, 30, 40]
            # subset.non_tensor["ids"] = ["a", "c", "d"]
        """
        return RLBatch(
            batch={key: _select_field(value, indices) for key, value in self.batch.items()},
            non_tensor={key: _select_field(value, indices) for key, value in self.non_tensor.items()},
            meta=copy.deepcopy(self.meta),
        )

    def repeat(self, repeat_times: int, interleave: bool = True) -> "RLBatch":
        """
        Args:
            repeat_times: Number of times to repeat each row.
            interleave: If True, repeats each row consecutively (e.g., [0, 0, 1, 1, 2, 2]). If False, repeats the entire batch in sequence (e.g., [0, 1, 2, 0, 1, 2]).
        Example:
            每个 prompt 生成 4 个响应
            prompts_batch = RLBatch(non_tensor={"prompt": ["Q1", "Q2"]})
            expanded = prompts_batch.repeat(4, interleave=True)
            expanded.non_tensor["prompt"] = ["Q1", "Q1", "Q1", "Q1", "Q2", "Q2", "Q2", "Q2"]
            这样可以批量送入 rollout 引擎
        """
        if repeat_times <= 0:
            raise ValueError("repeat_times must be positive.")
        indices = _range_for_repeat(len(self), repeat_times, interleave)
        return self.select(indices)

    def union(self, other: "RLBatch") -> "RLBatch":
        """
        Function:
            - Merges two batches of the same length by combining their fields.
        Example:
            # Rollout 阶段生成的数据
            rollout_data = RLBatch(
                non_tensor={"prompt": ["Q1", "Q2"], "response": ["A1", "A2"]}
            )

            # Reward 阶段计算的数据
            reward_data = RLBatch(
                batch={"reward": [0.8, 0.3]}
            )

            # 合并为完整训练数据
            full_batch = rollout_data.union(reward_data)
            # full_batch.non_tensor = {"prompt": [...], "response": [...]}
            # full_batch.batch = {"reward": [...]}
        """
        if len(self) != len(other):
            raise ValueError("Both batches must have the same length for union().")
        merged_batch = {key: value for key, value in self.batch.items()}
        merged_non_tensor = {key: value for key, value in self.non_tensor.items()}
        for target, source in ((merged_batch, other.batch), (merged_non_tensor, other.non_tensor)):
            for key, value in source.items():
                if key in target:
                    if target[key] != value:
                        raise ValueError("Cannot union batches with conflicting field values.")
                    continue
                target[key] = value
        merged_meta = copy.deepcopy(self.meta)
        merged_meta.update(copy.deepcopy(other.meta))
        return RLBatch(batch=merged_batch, non_tensor=merged_non_tensor, meta=merged_meta)

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]], batch_keys: Iterable[str] = ()) -> "RLBatch":
        """
        Function:
            作用：从行式数据（字典列表）构建批次
        Example:
            rows = [{"x": 1, "uid": "a"}, {"x": 2, "uid": "b"}]
            batch = RLBatch.from_rows(rows, batch_keys=("x",))
            # batch.batch["x"] = [1, 2]
            # batch.non_tensor["uid"] = ["a", "b"]
        """
        batch_keys = set(batch_keys)
        batch: Dict[str, List[Any]] = {}
        non_tensor: Dict[str, List[Any]] = {}
        for row in rows:
            for key, value in row.items():
                target = batch if key in batch_keys else non_tensor
                target.setdefault(key, []).append(copy.deepcopy(value))
        return cls(batch=batch, non_tensor=non_tensor)


__all__ = ["RLBatch"]
