"""Tests for checkpointable data loading primitives."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nanoverl.data.dataset import JsonDataset, StatefulDataLoader


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class StatefulDataLoaderTest(unittest.TestCase):
    def test_rank_sharded_batches_are_disjoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            _write_jsonl(
                dataset_path,
                [
                    {"prompt": "a"},
                    {"prompt": "b"},
                    {"prompt": "c"},
                    {"prompt": "d"},
                ],
            )
            dataset = JsonDataset(dataset_path)
            rank0_loader = StatefulDataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=True,
                rank=0,
                world_size=2,
            )
            rank1_loader = StatefulDataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=True,
                rank=1,
                world_size=2,
            )

            rank0_batch = rank0_loader.next_batch()
            rank1_batch = rank1_loader.next_batch()

            self.assertEqual(rank0_batch.non_tensor["prompt"], ["a"])
            self.assertEqual(rank1_batch.non_tensor["prompt"], ["b"])
            self.assertEqual(len(rank0_loader), 2)
            self.assertEqual(len(rank1_loader), 2)


if __name__ == "__main__":
    unittest.main()
