"""Tests for RLBatch operations."""

import unittest

from nanoverl.core.batch import RLBatch


class RLBatchTest(unittest.TestCase):
    def test_repeat_select_and_clone(self):
        batch = RLBatch(
            batch={"responses": [[1], [2], [3]]},
            non_tensor={"uid": ["a", "b", "c"]},
        )
        repeated = batch.repeat(2, interleave=True)
        self.assertEqual(repeated.non_tensor["uid"], ["a", "a", "b", "b", "c", "c"])

        selected = batch.select([2, 0, 1])
        self.assertEqual(selected.non_tensor["uid"], ["c", "a", "b"])

        cloned = batch.clone()
        cloned.batch["responses"][0][0] = 99
        self.assertEqual(batch.batch["responses"][0][0], 1)

    def test_union_and_from_rows(self):
        left = RLBatch(batch={"responses": [[1], [2]]}, non_tensor={"uid": ["x", "y"]})
        right = RLBatch(batch={"response_mask": [[1], [1]]}, non_tensor={"prompt": ["p1", "p2"]})
        merged = left.union(right)
        self.assertIn("responses", merged.batch)
        self.assertIn("response_mask", merged.batch)
        self.assertEqual(merged.non_tensor["prompt"], ["p1", "p2"])

        rows = [
            {"responses": [1], "uid": "x", "meta": {"ok": True}},
            {"responses": [2], "uid": "y", "meta": {"ok": False}},
        ]
        from_rows = RLBatch.from_rows(rows, batch_keys=("responses",))
        self.assertEqual(from_rows.batch["responses"], [[1], [2]])
        self.assertEqual(from_rows.non_tensor["uid"], ["x", "y"])

        rows[0]["meta"]["ok"] = False
        self.assertEqual(from_rows.non_tensor["meta"], [{"ok": True}, {"ok": False}])


if __name__ == "__main__":
    unittest.main()
