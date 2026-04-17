"""End-to-end trainer smoke tests."""

import json
import tempfile
import unittest
from pathlib import Path

from nanoverl.config import TrainerConfig
from nanoverl.trainer import build_trainer


class TrainerSmokeTest(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows):
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def _make_config(self, checkpoint_dir, dataset_path):
        return TrainerConfig.from_dict(
            {
                "data": {
                    "train_path": str(dataset_path),
                    "val_path": str(dataset_path),
                    "train_batch_size": 2,
                    "val_batch_size": 2,
                    "shuffle": False,
                    "seed": 5,
                },
                "algorithm": {
                    "advantage_estimator": "gae",
                    "use_kl_in_reward": True,
                    "kl_penalty": "low_var_kl",
                    "kl_coef": 0.01,
                },
                "actor": {
                    "backend": "debug",
                    "ppo_mini_batch_size": 2,
                    "ppo_epochs": 1,
                    "clip_ratio": 0.2,
                },
                "critic": {
                    "backend": "debug",
                    "enable": True,
                    "ppo_mini_batch_size": 2,
                    "ppo_epochs": 1,
                },
                "reference": {
                    "backend": "debug",
                    "enable": True,
                    "fixed_kl_offset": -0.1,
                },
                "rollout": {
                    "backend": "debug",
                    "response_length": 32,
                    "train": {"do_sample": True, "n": 1},
                    "validation": {"do_sample": False, "n": 1, "temperature": 0.0},
                },
                "trainer": {
                    "total_epochs": 1,
                    "total_training_steps": 2,
                    "validate_before_train": True,
                    "test_freq": 1,
                    "save_freq": 1,
                    "default_local_dir": checkpoint_dir,
                    "loggers": [],
                    "project_name": "nanoverl-test",
                    "experiment_name": "smoke",
                },
            }
        )

    def test_fit_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            self._write_jsonl(
                dataset_path,
                [
                    {"prompt": "say yes", "expected_response": "yes", "data_source": "qa"},
                    {"prompt": "say no", "expected_response": "no", "data_source": "qa"},
                    {"prompt": "what is two plus two ?", "expected_response": "four", "data_source": "math"},
                ],
            )
            config = self._make_config(tmpdir, dataset_path)
            trainer = build_trainer(config)
            try:
                val_metrics = trainer.fit()
                self.assertIn("val/reward_mean", val_metrics)
                self.assertEqual(trainer.global_step, 2)
                saved_loader_state = trainer.train_loader.state_dict()
            finally:
                trainer.close()

            resumed_trainer = build_trainer(config)
            try:
                self.assertTrue(resumed_trainer.load_checkpoint())
                self.assertEqual(resumed_trainer.global_step, 2)
                self.assertEqual(
                    resumed_trainer.train_loader.state_dict()["sampler"]["position"],
                    saved_loader_state["sampler"]["position"],
                )
            finally:
                resumed_trainer.close()


if __name__ == "__main__":
    unittest.main()
