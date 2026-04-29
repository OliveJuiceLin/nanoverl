"""Tests for actor -> rollout policy synchronization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nanoverl.config import TrainerConfig
from nanoverl.rollout.sync import PolicySyncer
from nanoverl.trainer import build_trainer
from nanoverl.workers.debug import DebugPolicyWorker


class DummyPolicyWorker:
    def __init__(self):
        self.state_dict_calls = 0
        self.policy_state_dict_calls = 0

    def state_dict(self):
        self.state_dict_calls += 1
        raise AssertionError("PolicySyncer must not read checkpoint state.")

    def policy_state_dict(self):
        self.policy_state_dict_calls += 1
        return {"version": 7}


class DummyRolloutEngine:
    def __init__(self):
        self.sync_calls = 0
        self.synced_state = None

    def sync_policy(self, policy_state):
        self.sync_calls += 1
        self.synced_state = dict(policy_state)


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _make_debug_config(tmpdir: str, dataset_path: Path, total_training_steps: int) -> TrainerConfig:
    return TrainerConfig.from_dict(
        {
            "data": {
                "train_path": str(dataset_path),
                "val_path": None,
                "train_batch_size": 2,
                "shuffle": False,
            },
            "actor": {"backend": "debug", "mini_batch_size": 2},
            "critic": {"backend": "debug", "enable": False},
            "reference": {"backend": "debug", "enable": False},
            "rollout": {"backend": "debug", "response_length": 8},
            "trainer": {
                "total_training_steps": total_training_steps,
                "validate_before_train": False,
                "save_freq": 0,
                "test_freq": 0,
                "loggers": [],
                "default_local_dir": tmpdir,
            },
        }
    )


class PolicySyncTest(unittest.TestCase):
    def test_policy_syncer_calls_policy_state_dict_and_sync_once(self):
        policy_worker = DummyPolicyWorker()
        rollout_engine = DummyRolloutEngine()
        syncer = PolicySyncer()

        result = syncer.sync(policy_worker, rollout_engine, "unit")

        self.assertEqual(policy_worker.state_dict_calls, 0)
        self.assertEqual(policy_worker.policy_state_dict_calls, 1)
        self.assertEqual(rollout_engine.sync_calls, 1)
        self.assertEqual(rollout_engine.synced_state, {"version": 7})
        self.assertEqual(result.reason, "unit")
        self.assertEqual(result.count, 1)
        self.assertEqual(result.metrics["rollout/policy_sync_count"], 1.0)
        self.assertGreaterEqual(result.metrics["rollout/policy_sync_seconds"], 0.0)

    def test_debug_policy_export_state_matches_sync_needs(self):
        policy_worker = DebugPolicyWorker(type("ActorConfig", (), {"policy_loss": "ppo_clip"})())
        checkpoint_state = policy_worker.state_dict()
        export_state = policy_worker.policy_state_dict()

        self.assertEqual(export_state, {"version": checkpoint_state["version"]})
        self.assertNotIn("optimizer_state", export_state)

    def test_fresh_fit_performs_startup_sync_once_before_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            _write_jsonl(dataset_path, [{"prompt": "a"}, {"prompt": "b"}])
            config = _make_debug_config(tmpdir, dataset_path, total_training_steps=0)
            trainer = build_trainer(config)
            try:
                trainer.fit()
                self.assertEqual(trainer.policy_syncer.policy_sync_count, 1)
                self.assertEqual(trainer.rollout_engine.policy_version, trainer.policy_worker.version)
            finally:
                trainer.close()

    def test_train_step_syncs_after_actor_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            _write_jsonl(dataset_path, [{"prompt": "a"}, {"prompt": "b"}])
            config = _make_debug_config(tmpdir, dataset_path, total_training_steps=1)
            trainer = build_trainer(config)
            try:
                batch = trainer.train_loader.next_batch()
                metrics = trainer.train_step(batch)
                self.assertEqual(trainer.policy_syncer.policy_sync_count, 1)
                self.assertEqual(metrics["rollout/policy_sync_count"], 1.0)
                self.assertEqual(trainer.rollout_engine.policy_version, trainer.policy_worker.version)
            finally:
                trainer.close()

    def test_resume_fit_syncs_once_when_no_more_training_is_needed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            _write_jsonl(dataset_path, [{"prompt": "a"}, {"prompt": "b"}])
            config = _make_debug_config(tmpdir, dataset_path, total_training_steps=1)

            trainer = build_trainer(config)
            try:
                trainer.fit()
            finally:
                trainer.close()

            resumed = build_trainer(config)
            try:
                resumed.fit()
                self.assertEqual(resumed.global_step, 1)
                self.assertEqual(resumed.policy_syncer.policy_sync_count, 1)
                self.assertEqual(resumed.rollout_engine.policy_version, resumed.policy_worker.version)
            finally:
                resumed.close()


if __name__ == "__main__":
    unittest.main()
