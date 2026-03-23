"""Tests for local checkpoint save and load."""

import tempfile
import unittest

from nanoverl.checkpoint import CheckpointManager, find_latest_checkpoint


class CheckpointManagerTest(unittest.TestCase):
    def test_save_and_load_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_to_keep=1)
            manager.save(1, {"value": 1})
            manager.save(2, {"value": 2})
            latest = find_latest_checkpoint(tmpdir)
            self.assertIsNotNone(latest)
            payload = manager.load_latest()
            self.assertEqual(payload["value"], 2)


if __name__ == "__main__":
    unittest.main()
