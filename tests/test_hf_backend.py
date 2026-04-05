"""Dependency-gated tests for the local Hugging Face backend."""

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
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from nanoverl.backends.hf import encode_text, load_tokenizer, pack_prompt_response_tokens
    from nanoverl.rollout.hf import HFRolloutEngine
    from nanoverl.workers.hf import HFPolicyWorker, HFValueWorker

    HF_TEST_DEPS = True
except ImportError:  # pragma: no cover - default in this workspace
    HF_TEST_DEPS = False


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


@unittest.skipUnless(HF_TEST_DEPS, "HF backend tests require torch, transformers, and tokenizers.")
class HFBackendTest(unittest.TestCase):
    def _make_model_dir(self, root: Path, pad_token_id: int = 0) -> Path:
        model_dir = root / "tiny-hf"
        model_dir.mkdir(parents=True, exist_ok=True)

        token_order = [
            "<pad>",
            "<bos>",
            "<eos>",
            "<unk>",
            "say",
            "yes",
            "no",
            "math",
            "four",
            "two",
            "plus",
            "what",
            "?",
            "answer",
        ]
        if pad_token_id != 0:
            token_order[0], token_order[pad_token_id] = token_order[pad_token_id], token_order[0]
        vocab = {token: index for index, token in enumerate(token_order)}
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
        model = GPT2LMHeadModel(config)
        model.save_pretrained(model_dir)
        return model_dir

    def _make_config(self, model_dir: Path, checkpoint_dir: Path, dataset_path: Path) -> TrainerConfig:
        return TrainerConfig.from_dict(
            {
                "data": {
                    "train_path": str(dataset_path),
                    "val_path": str(dataset_path),
                    "train_batch_size": 2,
                    "val_batch_size": 2,
                    "shuffle": False,
                    "seed": 3,
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
                    "backend": "hf",
                    "ppo_mini_batch_size": 2,
                    "ppo_epochs": 1,
                    "micro_batch_size": 1,
                    "clip_ratio": 0.2,
                    "lr": 1e-4,
                    "max_grad_norm": 1.0,
                },
                "critic": {
                    "backend": "hf",
                    "enable": True,
                    "ppo_mini_batch_size": 2,
                    "ppo_epochs": 1,
                    "micro_batch_size": 1,
                    "cliprange_value": 0.5,
                    "lr": 1e-4,
                    "max_grad_norm": 1.0,
                },
                "reference": {
                    "backend": "hf",
                    "enable": True,
                },
                "rollout": {
                    "backend": "hf",
                    "response_length": 4,
                    "train": {"do_sample": False, "n": 1, "temperature": 0.0},
                    "validation": {"do_sample": False, "n": 1, "temperature": 0.0},
                },
                "trainer": {
                    "total_epochs": 1,
                    "total_training_steps": 2,
                    "validate_before_train": True,
                    "test_freq": 1,
                    "save_freq": 1,
                    "default_local_dir": str(checkpoint_dir),
                    "loggers": [],
                    "project_name": "nanoverl-test",
                    "experiment_name": "hf-smoke",
                },
            }
        )

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
            non_tensor={
                "prompt_text": list(prompts),
                "response_text": list(responses),
            },
        )

    def test_pack_truncation_and_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_model_dir(Path(tmpdir))
            tokenizer = load_tokenizer(
                TrainerConfig.from_dict({"model": {"path": str(model_dir), "tokenizer_path": str(model_dir)}}).model
            )
            packed = pack_prompt_response_tokens(
                tokenizer=tokenizer,
                prompt_token_ids=encode_text(tokenizer, "what is two plus two ? say yes"),
                response_token_ids=encode_text(tokenizer, "answer four yes no"),
                max_prompt_length=4,
                max_response_length=2,
            )
            self.assertEqual(len(packed["prompts"]), 4)
            self.assertEqual(len(packed["responses"]), 2)
            self.assertEqual(len(packed["input_ids"]), 6)
            self.assertEqual(packed["input_ids"], packed["prompts"] + packed["responses"])
            self.assertEqual(packed["response_mask"], [1, 1])

    def test_worker_input_padding_uses_tokenizer_pad_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_model_dir(Path(tmpdir), pad_token_id=7)
            config = TrainerConfig.from_dict(
                {
                    "model": {"path": str(model_dir), "tokenizer_path": str(model_dir), "dtype": "float32"},
                    "actor": {"backend": "hf", "device": "cpu", "ppo_mini_batch_size": 2, "micro_batch_size": 1},
                    "critic": {"enable": False},
                    "reference": {"enable": False},
                    "rollout": {"backend": "hf", "response_length": 4},
                }
            )
            tokenizer = load_tokenizer(config.model)
            batch = self._make_batch(tokenizer, ["say yes", "what is two plus two ?"], ["yes", "four"])
            policy_worker = HFPolicyWorker(config.model, config.actor)
            input_ids, _ = policy_worker._build_model_inputs(batch)
            self.assertEqual(int(input_ids[0, len(batch.batch["input_ids"][0]) - 1].item()), batch.batch["input_ids"][0][-1])
            self.assertEqual(int(input_ids[0, len(batch.batch["input_ids"][0])].item()), tokenizer.pad_token_id)

    def test_rollout_supports_chat_template_prompts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = self._make_model_dir(root)
            chat_template_path = root / "chat_template.jinja"
            chat_template_path.write_text(
                "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
                "{% if add_generation_prompt %}assistant: {% endif %}",
                encoding="utf-8",
            )
            config = TrainerConfig.from_dict(
                {
                    "model": {
                        "path": str(model_dir),
                        "tokenizer_path": str(model_dir),
                        "dtype": "float32",
                        "chat_template_path": str(chat_template_path),
                    },
                    "data": {"max_prompt_length": 16},
                    "actor": {"backend": "hf", "ppo_mini_batch_size": 1},
                    "critic": {"enable": False},
                    "reference": {"enable": False},
                    "rollout": {"backend": "hf", "response_length": 4, "device": "cpu"},
                }
            )
            rollout = HFRolloutEngine(config.model, config.data, config.rollout)
            rollout_batch = rollout.generate(
                RLBatch(
                    non_tensor={
                        "prompt": [[{"role": "user", "content": "say yes"}]],
                    }
                ),
                SamplingParams(do_sample=False, temperature=0.0, n=1),
            )
            self.assertIn("response_text", rollout_batch.non_tensor)
            self.assertEqual(len(rollout_batch.batch["responses"]), 1)
            self.assertLessEqual(len(rollout_batch.batch["responses"][0]), 4)

    def test_policy_and_value_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_model_dir(Path(tmpdir))
            config = TrainerConfig.from_dict(
                {
                    "model": {"path": str(model_dir), "tokenizer_path": str(model_dir), "dtype": "float32"},
                    "rollout": {"backend": "hf", "response_length": 4},
                    "actor": {"backend": "hf", "ppo_mini_batch_size": 2, "micro_batch_size": 1},
                    "critic": {"backend": "hf", "ppo_mini_batch_size": 2, "micro_batch_size": 1},
                    "reference": {"enable": False},
                }
            )
            tokenizer = load_tokenizer(config.model)
            batch = self._make_batch(tokenizer, ["say yes", "what is two ?"], ["yes", "four"])

            policy_worker = HFPolicyWorker(config.model, config.actor)
            log_probs = policy_worker.compute_log_probs(batch)
            self.assertEqual(len(log_probs.log_probs), 2)
            self.assertEqual(len(log_probs.log_probs[0]), len(batch.batch["responses"][0]))
            self.assertEqual(log_probs.entropy, [])

            value_worker = HFValueWorker(config.model, config.critic)
            values = value_worker.compute_values(batch)
            self.assertEqual(len(values.values), 2)
            self.assertEqual(len(values.values[0]), len(batch.batch["responses"][0]))

    def test_rollout_sync_changes_log_probs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = self._make_model_dir(root)
            config = TrainerConfig.from_dict(
                {
                    "model": {"path": str(model_dir), "tokenizer_path": str(model_dir), "dtype": "float32"},
                    "data": {"max_prompt_length": 8},
                    "actor": {"backend": "hf", "ppo_mini_batch_size": 1},
                    "critic": {"enable": False},
                    "reference": {"enable": False},
                    "rollout": {"backend": "hf", "response_length": 4},
                }
            )
            rollout = HFRolloutEngine(config.model, config.data, config.rollout)
            policy_worker = HFPolicyWorker(config.model, config.actor)
            first_rollout_parameter = next(rollout.model.parameters()).detach().clone()
            with torch.no_grad():
                for parameter in policy_worker.model.parameters():
                    parameter.add_(0.25)
            rollout.sync_policy(policy_worker.state_dict())
            synced_rollout_parameter = next(rollout.model.parameters()).detach().clone()
            self.assertFalse(torch.equal(first_rollout_parameter, synced_rollout_parameter))

    def test_local_hf_components_can_use_explicit_devices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_model_dir(Path(tmpdir))
            config = TrainerConfig.from_dict(
                {
                    "model": {"path": str(model_dir), "tokenizer_path": str(model_dir), "dtype": "float32"},
                    "actor": {"backend": "hf", "device": "cpu", "ppo_mini_batch_size": 1},
                    "critic": {"backend": "hf", "device": "cpu", "enable": True, "ppo_mini_batch_size": 1},
                    "reference": {"backend": "hf", "device": "cpu", "enable": True},
                    "rollout": {"backend": "hf", "device": "cpu", "response_length": 4},
                }
            )
            policy_worker = HFPolicyWorker(config.model, config.actor)
            value_worker = HFValueWorker(config.model, config.critic)
            rollout_engine = HFRolloutEngine(config.model, config.data, config.rollout)
            self.assertEqual(str(policy_worker.device), "cpu")
            self.assertEqual(str(value_worker.device), "cpu")
            self.assertEqual(str(rollout_engine.device), "cpu")

    def test_hf_fit_and_resume(self):
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
