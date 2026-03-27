"""A small batch container tailored to RL training."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


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
    return copy.deepcopy(value)


def _select_field(value: Any, indices: Sequence[int]) -> Any:
    sequence = _ensure_sequence(value)
    selected = [_copy_item(sequence[index]) for index in indices]
    if isinstance(value, tuple):
        return tuple(selected)
    return selected


def _concat_field(chunks: Sequence[Any]) -> Any:
    merged: List[Any] = []
    for chunk in chunks:
        sequence = _ensure_sequence(chunk)
        merged.extend(_copy_item(item) for item in sequence)
    return merged


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

    def row(self, index: int) -> Dict[str, Any]:
        """
        Example:
            batch = RLBatch(
            batch={"logits": [0.1, 0.2, 0.3]},
            non_tensor={"text": ["A", "B", "C"]}
            )
            row = batch.row(1)
            # row = {"logits": 0.2, "text": "B"}
        """
        result: Dict[str, Any] = {}
        for key, value in self.batch.items():
            result[key] = _copy_item(_ensure_sequence(value)[index])
        for key, value in self.non_tensor.items():
            result[key] = _copy_item(_ensure_sequence(value)[index])
        return result

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
        merged = self.clone()
        for target, source in ((merged.batch, other.batch), (merged.non_tensor, other.non_tensor)):
            for key, value in source.items():
                if key in target:
                    if target[key] != value:
                        raise ValueError("Cannot union batches with conflicting field values.")
                    continue
                target[key] = copy.deepcopy(value)
        merged.meta.update(copy.deepcopy(other.meta))
        return merged

    def chunk(self, chunks: int) -> List["RLBatch"]:
        """
        Function:
            把 batch 切分成指定数量的 chunk，尽量平均分配, 但如果不能整除，前面几个 chunk 会多一个样本。
        Example:
            batch = RLBatch(batch={"x": [1, 2, 3, 4, 5]})
            chunks = batch.chunk(3)
            # chunks[0].batch["x"] = [1, 2]
            # chunks[1].batch["x"] = [3, 4]
            # chunks[2].batch["x"] = [5]
        """
        if chunks <= 0:
            raise ValueError("chunks must be positive.")
        length = len(self)
        if length == 0:
            return []
        chunks = min(chunks, length)
        base, extra = divmod(length, chunks)
        result: List[RLBatch] = []
        start = 0
        for chunk_index in range(chunks):
            width = base + (1 if chunk_index < extra else 0)
            result.append(self.select(list(range(start, start + width))))
            start += width
        return result

    @classmethod
    def concat(cls, batches: Sequence["RLBatch"]) -> "RLBatch":
        """
        Function:
            将多个批次合并成一个大批次（chunk 的逆操作）
        Example:
            batch1 = RLBatch(batch={"x": [1, 2]})
            batch2 = RLBatch(batch={"x": [3, 4]})
            batch3 = RLBatch(batch={"x": [5]})

            merged = RLBatch.concat([batch1, batch2, batch3])
            # merged.batch["x"] = [1, 2, 3, 4, 5]
        """
        if not batches:
            return RLBatch()
        keys_batch = set().union(*(batch.batch.keys() for batch in batches))
        keys_non_tensor = set().union(*(batch.non_tensor.keys() for batch in batches))
        concatenated_batch: Dict[str, Any] = {}
        concatenated_non_tensor: Dict[str, Any] = {}
        for key in keys_batch:
            concatenated_batch[key] = _concat_field([batch.batch[key] for batch in batches if key in batch.batch])
        for key in keys_non_tensor:
            concatenated_non_tensor[key] = _concat_field(
                [batch.non_tensor[key] for batch in batches if key in batch.non_tensor]
            )
        meta = copy.deepcopy(batches[0].meta)
        return cls(batch=concatenated_batch, non_tensor=concatenated_non_tensor, meta=meta)

    def reorder(self, indices: Sequence[int]) -> "RLBatch":
        return self.select(indices)

    def pad_to_divisor(self, divisor: int) -> Tuple["RLBatch", int]:
        """
        作用：
            将批次填充到 divisor 的整数倍，返回填充后的批次和填充数量, 
        填充策略：
            循环重复原始样本，而不是用零填充。
        Example:
            batch = RLBatch(batch={"x": [1, 2, 3, 4, 5]})
            padded, pad_size = batch.pad_to_divisor(3)
            # padded.batch["x"] = [1, 2, 3, 4, 5, 1]  (长度 6 是 3 的倍数)
            # pad_size = 1
        """
        if divisor <= 0:
            raise ValueError("divisor must be positive.")
        length = len(self)
        if length == 0 or length % divisor == 0:
            return self.clone(), 0
        pad_size = divisor - (length % divisor)
        indices = list(range(length))
        while len(indices) < length + pad_size:
            indices.append(indices[len(indices) % length])
        return self.select(indices), pad_size

    def unpad(self, pad_size: int) -> "RLBatch":
        """
        Function:
            移除前面 pad_to_divisor 添加的填充样本，恢复到原始长度。
        """
        if pad_size <= 0:
            return self.clone()
        return self.select(list(range(len(self) - pad_size)))

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]], batch_keys: Iterable[str] = ()) -> "RLBatch":
        """
        Function:
            作用：从行式数据（字典列表）构建批次
        Example:
            batch = RLBatch(batch={"x": [1, 2, 3, 4, 5]})
            padded, pad_size = batch.pad_to_divisor(3)
            # padded.batch["x"] = [1, 2, 3, 4, 5, 1]  (长度 6 是 3 的倍数)
            # pad_size = 1
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
