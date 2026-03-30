"""Dependency-gated tests for the first FSDP training path."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nanoverl.config import TrainerConfig
from nanoverl.core.batch import RLBatch
from nanoverl.rollout.base import SamplingParams
from nanoverl.trainer import build_trainer

try:  # pragma: no cover - exercised only when optional deps are installed
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from nanoverl.backends.hf import encode_text, load_tokenizer, pack_prompt_response_tokens
    from nanoverl.backends.train.fsdp import FSDPPolicyWorker, FSDPReferenceWorker, FSDPValueWorker
    from nanoverl.cli.train_rl import main as train_main
    from nanoverl.rollout.hf import HFRolloutEngine

    FSDP_TEST_DEPS = True
except ImportError:  # pragma: no cover - default when optional deps are missing
    FSDP_TEST_DEPS = False


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


@unittest.skipUnless(FSDP_TEST_DEPS, "FSDP backend tests require torch, transformers, and tokenizers.")
class FSDPBackendTest(unittest.TestCase):
    def _make_model_dir(self, root: Path) -> Path:
        model_dir = root / "tiny-fsdp-hf"
        model_dir.mkdir(parents=True, exist_ok=True)

        vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "say": 4,
            "yes": 5,
            "no": 6,
            "math": 7,
            "four": 8,
            "two": 9,
            "plus": 10,
            "what": 11,
            "?": 12,
        }
        tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer_obj.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            bos_token="<bos>",
            eos_token="<eos>",
            unk_token="<unk>",
            pad_token="<pad>",
        )
        tokenizer.save_pretrained(model_dir)

        config = GPT2Config(
            vocab_size=len(vocab),
            n_positions=64,
            n_ctx=64,
            n_embd=32,
            n_layer=2,
            n_head=2,
            bos_token_id=vocab["<bos>"],
            eos_token_id=vocab["<eos>"],
            pad_token_id=vocab["<pad>"],
        )
        GPT2LMHeadModel(config).save_pretrained(model_dir)
        return model_dir

    def _make_batch(self, tokenizer, prompts, responses) -> RLBatch:
        packed_rows = [
            pack_prompt_response_tokens(
                tokenizer=tokenizer,
                prompt_token_ids=encode_text(tokenizer, prompt),
                response_token_ids=encode_text(tokenizer, response),
                max_prompt_length=8,
                max_response_length=4,
            )
            for prompt, response in zip(prompts, responses)
        ]
        return RLBatch(
            batch={
                "prompts": [row["prompts"] for row in packed_rows],
                "responses": [row["responses"] for row in packed_rows],
                "input_ids": [row["input_ids"] for row in packed_rows],
                "attention_mask": [row["attention_mask"] for row in packed_rows],
                "response_mask": [row["response_mask"] for row in packed_rows],
            },
            non_tensor={"prompt_text": list(prompts), "response_text": list(responses)},
        )

    def _make_config(self, model_dir: Path, checkpoint_dir: Path, dataset_path: Path) -> TrainerConfig:
        return TrainerConfig.from_dict(
            {
                "data": {
                    "train_path": str(dataset_path),
                    "val_path": str(dataset_path),
                    "train_batch_size": 2,
                    "val_batch_size": 2,
                    "shuffle": False,
                    "seed": 9,
                    "max_prompt_length": 8,
                    "max_response_length": 4,
                },
                "model": {
                    "path": str(model_dir),
                    "tokenizer_path": str(model_dir),
                    "dtype": "float32",
                },
                "algorithm": {
                    "name": "ppo",
                    "advantage_estimator": "gae",
                    "use_kl_in_reward": True,
                    "kl_penalty": "low_var_kl",
                    "kl_coef": 0.01,
                },
                "actor": {
                    "backend": "fsdp",
                    "ppo_mini_batch_size": 2,
                    "ppo_epochs": 1,
                    "micro_batch_size": 1,
                    "clip_ratio": 0.2,
                    "lr": 1e-4,
                },
                "critic": {
                    "backend": "fsdp",
                    "enable": True,
                    "ppo_mini_batch_size": 2,
                    "ppo_epochs": 1,
                    "micro_batch_size": 1,
                    "cliprange_value": 0.5,
                    "lr": 1e-4,
                },
                "reference": {
                    "backend": "fsdp",
                    "enable": True,
                },
                "rollout": {
                    "backend": "hf",
                    "response_length": 4,
                    "train": {"do_sample": False, "n": 1, "temperature": 0.0},
                    "validation": {"do_sample": False, "n": 1, "temperature": 0.0},
                },
                "trainer": {
                    "total_training_steps": 2,
                    "validate_before_train": True,
                    "test_freq": 1,
                    "save_freq": 1,
                    "default_local_dir": str(checkpoint_dir),
                    "loggers": [],
                    "experiment_name": "fsdp-smoke",
                },
            }
        )

    def test_worker_shapes_and_rollout_sync(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_model_dir(Path(tmpdir))
            config = TrainerConfig.from_dict(
                {
                    "model": {"path": str(model_dir), "tokenizer_path": str(model_dir), "dtype": "float32"},
                    "actor": {"backend": "fsdp", "ppo_mini_batch_size": 2, "micro_batch_size": 1},
                    "critic": {"backend": "fsdp", "ppo_mini_batch_size": 2, "micro_batch_size": 1},
                    "reference": {"backend": "fsdp", "enable": True},
                    "rollout": {"backend": "hf", "response_length": 4},
                }
            )
            tokenizer = load_tokenizer(config.model)
            batch = self._make_batch(tokenizer, ["say yes", "what is two ?"], ["yes", "four"])

            policy_worker = FSDPPolicyWorker(config.model, config.actor)
            reference_worker = FSDPReferenceWorker(config.model, config.reference)
            value_worker = FSDPValueWorker(config.model, config.critic)
            log_probs = policy_worker.compute_log_probs(batch)
            ref_log_probs = reference_worker.compute_log_probs(batch)
            values = value_worker.compute_values(batch)

            self.assertEqual(len(log_probs.log_probs), 2)
            self.assertEqual(len(ref_log_probs.log_probs[0]), len(batch.batch["responses"][0]))
            self.assertEqual(len(values.values[1]), len(batch.batch["responses"][1]))

            rollout = HFRolloutEngine(config.model, config.data, config.rollout)
            before = rollout.generate(RLBatch(non_tensor={"prompt_text": ["say yes"]}), SamplingParams(do_sample=False, temperature=0.0, n=1))
            rollout.sync_policy(policy_worker.state_dict())
            after = rollout.generate(RLBatch(non_tensor={"prompt_text": ["say yes"]}), SamplingParams(do_sample=False, temperature=0.0, n=1))
            self.assertEqual(len(before.batch["rollout_log_probs"][0]), len(after.batch["rollout_log_probs"][0]))

    def test_fsdp_fit_resume_and_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = self._make_model_dir(root)
            dataset_path = root / "train.jsonl"
            checkpoint_dir = root / "checkpoints"
            _write_jsonl(
                dataset_path,
                [
                    {"prompt": "say yes", "expected_response": "yes", "data_source": "qa"},
                    {"prompt": "what is two plus two ?", "expected_response": "four", "data_source": "math"},
                    {"prompt": "say no", "expected_response": "no", "data_source": "qa"},
                ],
            )
            config = self._make_config(model_dir, checkpoint_dir, dataset_path)
            trainer = build_trainer(config)
            try:
                metrics = trainer.fit()
                self.assertIn("val/reward_mean", metrics)
                self.assertEqual(trainer.global_step, 2)
            finally:
                trainer.close()

            resumed = build_trainer(config)
            try:
                self.assertTrue(resumed.load_checkpoint())
                self.assertEqual(resumed.global_step, 2)
            finally:
                resumed.close()

            config_path = root / "fsdp_config.json"
            config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            train_main(["--config", str(config_path)])


if __name__ == "__main__":
    unittest.main()
