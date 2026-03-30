"""Simple dataset and stateful dataloader primitives."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from nanoverl.core.batch import RLBatch


class JsonDataset:
    """Reads prompt data from JSON or JSONL."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = self._load_rows(self.path)

    @staticmethod
    def _load_rows(path: Path) -> List[Dict[str, Any]]:
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("JSON dataset must contain a top-level list.")
            rows = payload
        else:
            raise ValueError("Only .json and .jsonl datasets are supported by the built-in loader.")
        if not rows:
            raise ValueError("Dataset is empty.")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


def collate_rows(rows: Sequence[Mapping[str, Any]], prompt_key: str = "prompt") -> RLBatch:
    batch = RLBatch.from_rows(rows)
    if prompt_key in batch.non_tensor and "prompt_text" not in batch.non_tensor:
        batch.non_tensor["prompt_text"] = list(batch.non_tensor[prompt_key])
    return batch


@dataclass
class StatefulIndexSampler:
    size: int
    shuffle: bool = False
    seed: Optional[int] = None
    rank: int = 0
    world_size: int = 1
    epoch: int = 0
    position: int = 0

    def __post_init__(self) -> None:
        self._order = list(range(self.size))
        self._shuffle_for_epoch()

    def _shuffle_for_epoch(self) -> None:
        self._order = list(range(self.size))
        if self.shuffle:
            rng = random.Random((self.seed or 0) + self.epoch)
            rng.shuffle(self._order)

    def next_indices(self, batch_size: int, drop_last: bool = True) -> Optional[List[int]]:
        if self.position >= self.size:
            return None
        global_batch_size = batch_size * max(self.world_size, 1)
        end = self.position + global_batch_size
        if end > self.size and drop_last:
            return None
        batch_end = min(end, self.size)
        shard_start = self.position + (self.rank * batch_size)
        shard_end = min(shard_start + batch_size, batch_end)
        indices = self._order[shard_start:shard_end]
        self.position += global_batch_size
        return indices

    def reset_for_new_epoch(self) -> None:
        self.epoch += 1
        self.position = 0
        self._shuffle_for_epoch()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "position": self.position,
            "order": list(self._order),
            "size": self.size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self._order = list(state["order"])
        self.rank = int(state.get("rank", self.rank))
        self.world_size = int(state.get("world_size", self.world_size))


class StatefulDataLoader:
    """A tiny, checkpointable alternative to a data loader."""

    def __init__(
        self,
        dataset: JsonDataset,
        batch_size: int,
        prompt_key: str = "prompt",
        shuffle: bool = False,
        seed: Optional[int] = None,
        drop_last: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.prompt_key = prompt_key
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.sampler = StatefulIndexSampler(
            len(dataset),
            shuffle=shuffle,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )

    def __len__(self) -> int:
        global_batch_size = self.batch_size * max(self.world_size, 1)
        if self.drop_last:
            return len(self.dataset) // global_batch_size
        return (len(self.dataset) + global_batch_size - 1) // global_batch_size

    @property
    def epoch(self) -> int:
        return self.sampler.epoch

    def next_batch(self) -> Optional[RLBatch]:
        indices = self.sampler.next_indices(self.batch_size, drop_last=self.drop_last)
        if indices is None:
            return None
        rows = [self.dataset[index] for index in indices]
        return collate_rows(rows, prompt_key=self.prompt_key)

    def reset_for_new_epoch(self) -> None:
        self.sampler.reset_for_new_epoch()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "prompt_key": self.prompt_key,
            "drop_last": self.drop_last,
            "rank": self.rank,
            "world_size": self.world_size,
            "sampler": self.sampler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.sampler.load_state_dict(state["sampler"])


__all__ = ["JsonDataset", "StatefulDataLoader", "collate_rows"]
