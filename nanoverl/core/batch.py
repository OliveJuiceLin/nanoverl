"""A small batch container tailored to RL training."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _is_sequence(value: Any) -> bool:
    return hasattr(value, "__len__") and hasattr(value, "__getitem__") and not isinstance(value, (str, bytes))


def _ensure_sequence(value: Any) -> Sequence[Any]:
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
    """A deliberately small batch protocol for RL training."""

    batch: Dict[str, Any] = field(default_factory=dict)
    non_tensor: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

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
        result: Dict[str, Any] = {}
        for key, value in self.batch.items():
            result[key] = _copy_item(_ensure_sequence(value)[index])
        for key, value in self.non_tensor.items():
            result[key] = _copy_item(_ensure_sequence(value)[index])
        return result

    def select(self, indices: Sequence[int]) -> "RLBatch":
        return RLBatch(
            batch={key: _select_field(value, indices) for key, value in self.batch.items()},
            non_tensor={key: _select_field(value, indices) for key, value in self.non_tensor.items()},
            meta=copy.deepcopy(self.meta),
        )

    def repeat(self, repeat_times: int, interleave: bool = True) -> "RLBatch":
        if repeat_times <= 0:
            raise ValueError("repeat_times must be positive.")
        indices = _range_for_repeat(len(self), repeat_times, interleave)
        return self.select(indices)

    def union(self, other: "RLBatch") -> "RLBatch":
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
        if pad_size <= 0:
            return self.clone()
        return self.select(list(range(len(self) - pad_size)))

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]], batch_keys: Iterable[str] = ()) -> "RLBatch":
        batch_keys = set(batch_keys)
        batch: Dict[str, List[Any]] = {}
        non_tensor: Dict[str, List[Any]] = {}
        for row in rows:
            for key, value in row.items():
                target = batch if key in batch_keys else non_tensor
                target.setdefault(key, []).append(copy.deepcopy(value))
        return cls(batch=batch, non_tensor=non_tensor)


__all__ = ["RLBatch"]
