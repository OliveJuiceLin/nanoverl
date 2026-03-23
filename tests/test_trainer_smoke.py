"""End-to-end trainer smoke tests."""

import tempfile
import unittest

from nanoverl.config import TrainerConfig
from nanoverl.trainer import build_trainer


class TrainerSmokeTest(unittest.TestCase):
    def _make_config(self, checkpoint_dir):
        return TrainerConfig.from_dict(
            {
                "data": {
                    "train_path": "examples/data/debug_prompts.jsonl",
                    "val_path": "examples/data/debug_prompts.jsonl",
                    "train_batch_size": 2,
                    "val_batch_size": 2,
                    "shuffle": False,
                    "seed": 5,
                },
                "algorithm": {
                    "name": "ppo",
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
            config = self._make_config(tmpdir)
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
