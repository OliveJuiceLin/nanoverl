"""Tests for algorithm plugin plumbing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nanoverl.algos import RLAlgorithm, get_advantage_estimator, register_algorithm
from nanoverl.algos.ppo import get_policy_loss_fn
from nanoverl.config import TrainerConfig
from nanoverl.trainer import build_trainer


@register_algorithm("dummy_test_algo")
class DummyAlgorithm(RLAlgorithm):
    def uses_critic(self, config):
        return False

    def run_step(self, batch, context):
        return {
            "dummy/rows": float(len(batch)),
            "training/global_step": float(context.global_step + 1),
            "training/epoch": float(context.train_epoch),
        }


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class AlgorithmPluginTest(unittest.TestCase):
    def test_legacy_configs_infer_algorithm_name(self):
        ppo_config = TrainerConfig.from_dict({"algorithm": {"advantage_estimator": "gae"}})
        self.assertEqual(ppo_config.algorithm.name, "ppo")

        grpo_config = TrainerConfig.from_dict(
            {
                "algorithm": {"advantage_estimator": "grpo"},
                "actor": {"ppo_mini_batch_size": 2},
                "rollout": {"train": {"n": 2}},
            }
        )
        self.assertEqual(grpo_config.algorithm.name, "grpo")

    def test_explicit_grpo_disables_critic_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            _write_jsonl(
                dataset_path,
                [
                    {"prompt": "say yes", "expected_response": "yes"},
                    {"prompt": "say no", "expected_response": "no"},
                ],
            )
            config = TrainerConfig.from_dict(
                {
                    "data": {
                        "train_path": str(dataset_path),
                        "val_path": str(dataset_path),
                        "train_batch_size": 2,
                        "val_batch_size": 2,
                        "shuffle": False,
                    },
                    "algorithm": {"name": "grpo"},
                    "actor": {"backend": "debug", "ppo_mini_batch_size": 2},
                    "critic": {"backend": "debug", "enable": True, "ppo_mini_batch_size": 2},
                    "reference": {"backend": "debug", "enable": True},
                    "rollout": {"backend": "debug", "train": {"n": 2}},
                    "trainer": {"loggers": [], "default_local_dir": tmpdir},
                }
            )
            trainer = build_trainer(config)
            try:
                self.assertEqual(trainer.algorithm.name, "grpo")
                self.assertEqual(trainer.config.algorithm.advantage_estimator, "grpo")
                self.assertIsNone(trainer.value_worker)
            finally:
                trainer.close()

    def test_explicit_rloo_uses_reinforce_actor_only_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            _write_jsonl(
                dataset_path,
                [
                    {"prompt": "say yes", "expected_response": "yes"},
                    {"prompt": "say no", "expected_response": "no"},
                ],
            )
            config = TrainerConfig.from_dict(
                {
                    "data": {
                        "train_path": str(dataset_path),
                        "val_path": None,
                        "train_batch_size": 2,
                        "shuffle": False,
                    },
                    "algorithm": {"name": "rloo"},
                    "actor": {"backend": "debug", "ppo_mini_batch_size": 2},
                    "critic": {"backend": "debug", "enable": True, "ppo_mini_batch_size": 2},
                    "reference": {"backend": "debug", "enable": False},
                    "rollout": {"backend": "debug", "train": {"n": 2}},
                    "trainer": {
                        "validate_before_train": False,
                        "loggers": [],
                        "default_local_dir": tmpdir,
                    },
                }
            )
            self.assertEqual(config.algorithm.advantage_estimator, "rloo")
            self.assertEqual(config.actor.policy_loss, "reinforce")

            trainer = build_trainer(config)
            try:
                self.assertEqual(trainer.algorithm.name, "rloo")
                self.assertIsNone(trainer.value_worker)
                batch = trainer.train_loader.next_batch()
                metrics = trainer.train_step(batch)
            finally:
                trainer.close()

        self.assertIn("actor/actor_loss", metrics)
        self.assertNotIn("critic/critic_loss", metrics)

    def test_registries_report_unknown_names(self):
        with self.assertRaisesRegex(ValueError, "Unsupported advantage estimator"):
            get_advantage_estimator("missing_estimator")
        with self.assertRaisesRegex(ValueError, "Unsupported policy loss"):
            get_policy_loss_fn("missing_loss")

    def test_trainer_delegates_train_step_to_algorithm_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            _write_jsonl(
                dataset_path,
                [
                    {"prompt": "alpha"},
                    {"prompt": "beta"},
                ],
            )
            config = TrainerConfig.from_dict(
                {
                    "data": {
                        "train_path": str(dataset_path),
                        "val_path": None,
                        "train_batch_size": 2,
                        "shuffle": False,
                    },
                    "algorithm": {"name": "dummy_test_algo"},
                    "critic": {"backend": "debug", "enable": True},
                    "reference": {"backend": "debug", "enable": False},
                    "rollout": {"backend": "debug"},
                    "trainer": {
                        "validate_before_train": False,
                        "loggers": [],
                        "default_local_dir": tmpdir,
                    },
                }
            )
            trainer = build_trainer(config)
            try:
                self.assertIsNone(trainer.value_worker)
                batch = trainer.train_loader.next_batch()
                metrics = trainer.train_step(batch)
            finally:
                trainer.close()

        self.assertEqual(metrics["dummy/rows"], 2.0)
        self.assertEqual(metrics["training/global_step"], 1.0)


if __name__ == "__main__":
    unittest.main()
