"""Tests for worker and rollout backend registries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nanoverl.config import TrainerConfig
from nanoverl.rollout import (
    create_rollout_engine,
    get_rollout_engine,
    register_rollout_engine,
    registered_rollout_backends,
)
from nanoverl.rollout.base import RolloutEngine
from nanoverl.trainer import build_trainer
from nanoverl.workers import (
    create_policy_worker,
    create_reference_worker,
    create_value_worker,
    get_policy_worker,
    get_reference_worker,
    get_value_worker,
    register_policy_worker,
    register_reference_worker,
    register_value_worker,
    registered_worker_backends,
)
from nanoverl.workers.base import PolicyWorker, ReferenceWorker, ValueWorker


class DummyPolicyWorker(PolicyWorker):
    pass


class DummyReferenceWorker(ReferenceWorker):
    pass


class DummyValueWorker(ValueWorker):
    pass


class DummyRolloutEngine(RolloutEngine):
    pass


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class BackendRegistryTest(unittest.TestCase):
    def test_register_and_create_worker_backends(self):
        register_policy_worker("unit-policy", lambda model_config, actor_config: DummyPolicyWorker())
        register_reference_worker("unit-reference", lambda model_config, ref_config: DummyReferenceWorker())
        register_value_worker("unit-value", lambda model_config, critic_config: DummyValueWorker())

        self.assertIsInstance(create_policy_worker("unit-policy", None, None), DummyPolicyWorker)
        self.assertIsInstance(create_reference_worker("unit-reference", None, None), DummyReferenceWorker)
        self.assertIsInstance(create_value_worker("unit-value", None, None), DummyValueWorker)
        self.assertIn("unit-policy", registered_worker_backends()["policy"])
        self.assertIn("unit-reference", registered_worker_backends()["reference"])
        self.assertIn("unit-value", registered_worker_backends()["value"])

    def test_unknown_worker_backend_reports_role_and_registered_names(self):
        with self.assertRaisesRegex(ValueError, "Unknown policy backend: missing"):
            get_policy_worker("missing")
        with self.assertRaisesRegex(ValueError, "Unknown reference backend: missing"):
            get_reference_worker("missing")
        with self.assertRaisesRegex(ValueError, "Unknown value backend: missing"):
            get_value_worker("missing")

    def test_register_and_create_rollout_backend(self):
        register_rollout_engine("unit-rollout", lambda model_config, data_config, rollout_config: DummyRolloutEngine())

        self.assertIsInstance(create_rollout_engine("unit-rollout", None, None, None), DummyRolloutEngine)
        self.assertIn("unit-rollout", registered_rollout_backends())

    def test_unknown_rollout_backend_reports_registered_names(self):
        with self.assertRaisesRegex(ValueError, "Unknown rollout backend: missing"):
            get_rollout_engine("missing")

    def test_build_trainer_uses_registered_debug_backends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            _write_jsonl(dataset_path, [{"prompt": "a"}, {"prompt": "b"}])
            config = TrainerConfig.from_dict(
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
                        "total_training_steps": 0,
                        "validate_before_train": False,
                        "save_freq": 0,
                        "loggers": [],
                        "default_local_dir": tmpdir,
                    },
                }
            )
            trainer = build_trainer(config)
            try:
                self.assertEqual(trainer.policy_worker.__class__.__name__, "DebugPolicyWorker")
                self.assertEqual(trainer.rollout_engine.__class__.__name__, "DebugRolloutEngine")
            finally:
                trainer.close()


if __name__ == "__main__":
    unittest.main()
