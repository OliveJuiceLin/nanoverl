"""Phase 2 regression tests for trainer usability features."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nanoverl.config import ConfigError, TrainerConfig
from nanoverl.core.batch import RLBatch
from nanoverl.trainer import build_trainer


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n", encoding="utf-8")


class ConfigValidationTest(unittest.TestCase):
    def test_actor_micro_batch_must_divide_ppo_mini_batch(self):
        with self.assertRaises(ConfigError):
            TrainerConfig.from_dict(
                {
                    "actor": {
                        "backend": "debug",
                        "ppo_mini_batch_size": 3,
                        "micro_batch_size": 2,
                    }
                }
            )

    def test_balance_batch_requires_matching_actor_and_critic_minibatches(self):
        with self.assertRaises(ConfigError):
            TrainerConfig.from_dict(
                {
                    "actor": {"backend": "debug", "ppo_mini_batch_size": 4},
                    "critic": {"backend": "debug", "enable": True, "ppo_mini_batch_size": 2},
                    "trainer": {"balance_batch": True},
                }
            )

    def test_hf_actor_requires_hf_reference(self):
        with self.assertRaises(ConfigError):
            TrainerConfig.from_dict(
                {
                    "actor": {"backend": "hf"},
                    "reference": {"backend": "debug", "enable": True},
                    "rollout": {"backend": "hf"},
                }
            )

    def test_grpo_requires_grouped_rollouts(self):
        with self.assertRaises(ConfigError):
            TrainerConfig.from_dict(
                {
                    "algorithm": {"name": "grpo"},
                    "rollout": {"train": {"n": 1}},
                }
            )


class TrainerPhase2Test(unittest.TestCase):
    def _make_debug_config(self, checkpoint_dir: str, dataset_path: str) -> TrainerConfig:
        return TrainerConfig.from_dict(
            {
                "data": {
                    "train_path": dataset_path,
                    "val_path": dataset_path,
                    "train_batch_size": 4,
                    "val_batch_size": 4,
                    "shuffle": False,
                },
                "actor": {
                    "backend": "debug",
                    "ppo_mini_batch_size": 4,
                    "ppo_epochs": 1,
                },
                "critic": {
                    "backend": "debug",
                    "enable": True,
                    "ppo_mini_batch_size": 4,
                    "ppo_epochs": 1,
                },
                "reference": {
                    "backend": "debug",
                    "enable": True,
                },
                "rollout": {
                    "backend": "debug",
                    "response_length": 16,
                    "train": {"do_sample": True, "n": 1},
                    "validation": {"do_sample": False, "n": 1, "temperature": 0.0},
                },
                "trainer": {
                    "total_training_steps": 1,
                    "save_freq": 0,
                    "test_freq": 0,
                    "validate_before_train": False,
                    "loggers": [],
                    "default_local_dir": checkpoint_dir,
                    "balance_batch": True,
                },
            }
        )

    def test_balance_batch_reorders_groups_without_splitting_them(self):
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
            trainer = build_trainer(self._make_debug_config(tmpdir, str(dataset_path)))
            try:
                batch = RLBatch(
                    batch={
                        "prompts": [[1], [1], [1], [1], [1], [1], [1], [1]],
                        "response_mask": [[1] * 9, [1] * 9, [1] * 8, [1] * 8, [1], [1], [1], []],
                    },
                    non_tensor={"uid": ["a", "a", "b", "b", "c", "c", "d", "d"]},
                )
                balanced = trainer._balance_rollout_batch(batch)
            finally:
                trainer.close()

        before_chunk_sums = [20 + 18, 2 + 1]
        after_uids = balanced.non_tensor["uid"]
        self.assertEqual(sorted(after_uids), sorted(batch.non_tensor["uid"]))
        for uid in {"a", "b", "c", "d"}:
            indices = [index for index, value in enumerate(after_uids) if value == uid]
            self.assertEqual(indices, list(range(indices[0], indices[0] + len(indices))))

        after_chunk_sums = []
        for start in range(0, len(balanced), 4):
            workload = 0
            for row_index in range(start, min(start + 4, len(balanced))):
                workload += len(balanced.batch["prompts"][row_index])
                workload += sum(balanced.batch["response_mask"][row_index])
            after_chunk_sums.append(workload)
        self.assertLessEqual(max(after_chunk_sums) - min(after_chunk_sums), max(before_chunk_sums) - min(before_chunk_sums))

    def test_grpo_uses_actor_only_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "grpo.jsonl"
            _write_jsonl(
                dataset_path,
                [
                    {
                        "prompt": "say yes",
                        "expected_response": "yes",
                        "scripted_responses": ["yes", "no"],
                        "data_source": "qa",
                    },
                    {
                        "prompt": "say yes again",
                        "expected_response": "yes",
                        "scripted_responses": ["yes", "no"],
                        "data_source": "qa",
                    },
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
                    "algorithm": {
                        "name": "grpo",
                    },
                    "actor": {
                        "backend": "debug",
                        "ppo_mini_batch_size": 2,
                        "ppo_epochs": 1,
                    },
                    "critic": {
                        "backend": "debug",
                        "enable": True,
                        "ppo_mini_batch_size": 2,
                    },
                    "reference": {
                        "backend": "debug",
                        "enable": True,
                    },
                    "rollout": {
                        "backend": "debug",
                        "response_length": 8,
                        "train": {"do_sample": True, "n": 2},
                        "validation": {"do_sample": False, "n": 1, "temperature": 0.0},
                    },
                    "trainer": {
                        "total_training_steps": 1,
                        "validate_before_train": False,
                        "save_freq": 0,
                        "test_freq": 0,
                        "loggers": [],
                        "default_local_dir": tmpdir,
                    },
                }
            )
            trainer = build_trainer(config)
            try:
                self.assertIsNone(trainer.value_worker)
                train_batch = trainer.train_loader.next_batch()
                metrics = trainer.train_step(train_batch)
                self.assertIn("actor/actor_loss", metrics)
                self.assertNotIn("critic/critic_loss", metrics)
            finally:
                trainer.close()

    def test_validation_summarizes_numeric_reward_extras(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reward_path = Path(tmpdir) / "reward_fn.py"
            reward_path.write_text(
                "\n".join(
                    [
                        "def compute_reward(prompt, response, sample):",
                        "    return {",
                        "        'score': 1.0 if response.strip() == sample['expected_response'] else 0.0,",
                        "        'confidence': 0.75 if sample['data_source'] == 'math' else 0.25,",
                        "        'label': sample['data_source'],",
                        "    }",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            dataset_path = Path(tmpdir) / "train.jsonl"
            _write_jsonl(
                dataset_path,
                [
                    {"prompt": "Respond with yes", "expected_response": "yes", "data_source": "qa"},
                    {"prompt": "What is two plus two ?", "expected_response": "four", "data_source": "math"},
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
                    "reward": {
                        "function_path": str(reward_path),
                        "function_name": "compute_reward",
                    },
                    "actor": {"backend": "debug", "ppo_mini_batch_size": 2},
                    "critic": {"backend": "debug", "enable": False},
                    "reference": {"backend": "debug", "enable": False},
                    "rollout": {
                        "backend": "debug",
                        "response_length": 8,
                        "validation": {"do_sample": False, "n": 1, "temperature": 0.0},
                    },
                    "trainer": {
                        "validate_before_train": False,
                        "loggers": [],
                        "default_local_dir": tmpdir,
                    },
                }
            )
            trainer = build_trainer(config)
            try:
                metrics = trainer.validate()
            finally:
                trainer.close()
            self.assertIn("val/extra/confidence_mean", metrics)
            self.assertNotIn("val/extra/label_mean", metrics)


if __name__ == "__main__":
    unittest.main()
