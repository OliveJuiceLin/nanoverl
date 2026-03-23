"""Tests for RLBatch operations."""

import unittest

from nanoverl.core.batch import RLBatch


class RLBatchTest(unittest.TestCase):
    def test_repeat_chunk_concat_and_reorder(self):
        batch = RLBatch(
            batch={"responses": [[1], [2], [3]]},
            non_tensor={"uid": ["a", "b", "c"]},
        )
        repeated = batch.repeat(2, interleave=True)
        self.assertEqual(repeated.non_tensor["uid"], ["a", "a", "b", "b", "c", "c"])

        chunks = repeated.chunk(3)
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 2])
        merged = RLBatch.concat(chunks)
        self.assertEqual(merged.non_tensor["uid"], repeated.non_tensor["uid"])

        reordered = batch.reorder([2, 0, 1])
        self.assertEqual(reordered.non_tensor["uid"], ["c", "a", "b"])

    def test_union_and_padding(self):
        left = RLBatch(batch={"responses": [[1], [2]]}, non_tensor={"uid": ["x", "y"]})
        right = RLBatch(batch={"response_mask": [[1], [1]]}, non_tensor={"prompt": ["p1", "p2"]})
        merged = left.union(right)
        self.assertIn("responses", merged.batch)
        self.assertIn("response_mask", merged.batch)
        self.assertEqual(merged.non_tensor["prompt"], ["p1", "p2"])

        padded, pad_size = merged.pad_to_divisor(3)
        self.assertEqual(len(padded), 3)
        self.assertEqual(pad_size, 1)
        unpadded = padded.unpad(pad_size)
        self.assertEqual(unpadded.non_tensor["uid"], ["x", "y"])


if __name__ == "__main__":
    unittest.main()
