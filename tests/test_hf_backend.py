"""Dependency-gated tests for the local Hugging Face backend."""

from __future__ import annotations

import json
import math
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

    from nanoverl.backends.hf import encode_text, load_tokenizer
    from nanoverl.rollout.hf import HFRolloutEngine
    from nanoverl.workers.hf import HFPolicyWorker, HFValueWorker

    HF_TEST_DEPS = True
except ImportError:  # pragma: no cover - default in this workspace
    HF_TEST_DEPS = False


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class _RecordingTracker:
    def __init__(self):
        self.events = []

    def log(self, data, step):
        self.events.append((step, dict(data)))

    def close(self):
        return


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
        packed_rows = []
        for prompt, response in zip(prompts, responses):
            prompt_token_ids = encode_text(tokenizer, prompt)
            response_token_ids = encode_text(tokenizer, response)
            packed_rows.append(
                {
                    "prompts": prompt_token_ids,
                    "responses": response_token_ids,
                    "input_ids": prompt_token_ids + response_token_ids,
                    "attention_mask": [1] * (len(prompt_token_ids) + len(response_token_ids)),
                    "response_mask": [1] * len(response_token_ids),
                }
            )
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

    def _attach_policy_training_fields(self, batch: RLBatch, old_log_probs, advantages, ref_log_probs=None) -> RLBatch:
        batch.batch["old_log_probs"] = old_log_probs
        batch.batch["advantages"] = advantages
        if ref_log_probs is not None:
            batch.batch["ref_log_probs"] = ref_log_probs
        return batch

    def _attach_value_training_fields(self, batch: RLBatch, returns) -> RLBatch:
        batch.batch["returns"] = returns
        return batch

    def test_manual_batch_contract_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_model_dir(Path(tmpdir))
            tokenizer = load_tokenizer(
                TrainerConfig.from_dict({"model": {"path": str(model_dir), "tokenizer_path": str(model_dir)}}).model
            )
            prompt_token_ids = encode_text(tokenizer, "what is two plus two ? say yes")
            response_token_ids = encode_text(tokenizer, "answer four yes no")
            input_ids = prompt_token_ids + response_token_ids
            attention_mask = [1] * len(input_ids)
            response_mask = [1] * len(response_token_ids)
            self.assertEqual(input_ids, prompt_token_ids + response_token_ids)
            self.assertEqual(len(attention_mask), len(input_ids))
            self.assertEqual(response_mask, [1] * len(response_token_ids))

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

    def test_policy_entropy_recording_is_decoupled_from_entropy_loss(self):
        cases = [
            {"entropy_coeff": 0.0, "record_entropy": False, "expected_compute_entropy": False, "expect_metric": False},
            {"entropy_coeff": 0.0, "record_entropy": True, "expected_compute_entropy": True, "expect_metric": True},
            {"entropy_coeff": 0.1, "record_entropy": False, "expected_compute_entropy": True, "expect_metric": False},
            {"entropy_coeff": 0.1, "record_entropy": True, "expected_compute_entropy": True, "expect_metric": True},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_model_dir(Path(tmpdir))
            tokenizer = load_tokenizer(
                TrainerConfig.from_dict({"model": {"path": str(model_dir), "tokenizer_path": str(model_dir)}}).model
            )
            batch = self._attach_policy_training_fields(
                self._make_batch(tokenizer, ["say yes", "say no"], ["yes", "no"]),
                old_log_probs=[[0.0], [0.0]],
                advantages=[[1.0], [1.0]],
            )

            for case in cases:
                with self.subTest(case=case):
                    config = TrainerConfig.from_dict(
                        {
                            "model": {"path": str(model_dir), "tokenizer_path": str(model_dir), "dtype": "float32"},
                            "rollout": {"backend": "hf", "response_length": 4},
                            "actor": {
                                "backend": "hf",
                                "device": "cpu",
                                "ppo_mini_batch_size": 2,
                                "micro_batch_size": 1,
                                "ppo_epochs": 1,
                                "entropy_coeff": case["entropy_coeff"],
                                "record_entropy": case["record_entropy"],
                            },
                            "critic": {"enable": False},
                            "reference": {"enable": False},
                        }
                    )
                    policy_worker = HFPolicyWorker(config.model, config.actor)
                    compute_entropy_flags = []

                    def fake_compute(_model, microbatch, compute_entropy=False):
                        del microbatch
                        compute_entropy_flags.append(bool(compute_entropy))
                        log_probs = torch.zeros((1, 1), dtype=torch.float32, device=policy_worker.device, requires_grad=True)
                        entropy = None
                        if compute_entropy:
                            entropy = torch.full(
                                (1, 1),
                                0.5,
                                dtype=torch.float32,
                                device=policy_worker.device,
                                requires_grad=True,
                            )
                        return log_probs, entropy

                    policy_worker._compute_response_log_probs_and_entropy = fake_compute
                    result = policy_worker.update(batch)
                    self.assertEqual(compute_entropy_flags, [case["expected_compute_entropy"], case["expected_compute_entropy"]])
                    self.assertEqual("policy_entropy" in result.metrics, case["expect_metric"])

    def test_policy_update_metrics_use_weighted_microbatch_aggregation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_model_dir(Path(tmpdir))
            config = TrainerConfig.from_dict(
                {
                    "model": {"path": str(model_dir), "tokenizer_path": str(model_dir), "dtype": "float32"},
                    "rollout": {"backend": "hf", "response_length": 4},
                    "actor": {
                        "backend": "hf",
                        "device": "cpu",
                        "ppo_mini_batch_size": 2,
                        "micro_batch_size": 1,
                        "ppo_epochs": 1,
                        "clip_ratio": 0.2,
                        "entropy_coeff": 0.0,
                        "record_entropy": True,
                    },
                    "critic": {"enable": False},
                    "reference": {"enable": False},
                }
            )
            tokenizer = load_tokenizer(config.model)
            batch = self._attach_policy_training_fields(
                self._make_batch(tokenizer, ["say yes", "what is two plus two ?"], ["yes", "answer four yes"]),
                old_log_probs=[[0.0], [0.0, 0.0, 0.0]],
                advantages=[[1.0], [1.0, 1.0, 1.0]],
            )
            policy_worker = HFPolicyWorker(config.model, config.actor)

            def fake_compute(_model, microbatch, compute_entropy=False):
                self.assertTrue(compute_entropy)
                response_length = len(microbatch.batch["responses"][0])
                if response_length == 1:
                    log_probs = torch.tensor(
                        [[math.log(2.0)]],
                        dtype=torch.float32,
                        device=policy_worker.device,
                        requires_grad=True,
                    )
                    entropy = torch.tensor(
                        [[10.0]],
                        dtype=torch.float32,
                        device=policy_worker.device,
                        requires_grad=True,
                    )
                else:
                    log_probs = torch.tensor(
                        [[0.0, 0.0, 0.0]],
                        dtype=torch.float32,
                        device=policy_worker.device,
                        requires_grad=True,
                    )
                    entropy = torch.tensor(
                        [[1.0, 1.0, 1.0]],
                        dtype=torch.float32,
                        device=policy_worker.device,
                        requires_grad=True,
                    )
                return log_probs, entropy

            policy_worker._compute_response_log_probs_and_entropy = fake_compute
            result = policy_worker.update(batch)
            self.assertAlmostEqual(result.metrics["policy_entropy"], 3.25, places=6)
            self.assertAlmostEqual(result.metrics["policy_clipfrac"], 0.25, places=6)
            self.assertAlmostEqual(result.metrics["policy_clipfrac_lower"], 0.0, places=6)
            self.assertAlmostEqual(result.metrics["policy_approx_kl"], -math.log(2.0) / 4.0, places=6)
            self.assertAlmostEqual(result.metrics["actor_loss"], -1.05, places=6)
            self.assertEqual(len(result.step_metrics), 1)
            self.assertAlmostEqual(result.step_metrics[0]["policy_entropy"], 3.25, places=6)

    def test_value_update_metrics_use_weighted_microbatch_aggregation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_model_dir(Path(tmpdir))
            config = TrainerConfig.from_dict(
                {
                    "model": {"path": str(model_dir), "tokenizer_path": str(model_dir), "dtype": "float32"},
                    "rollout": {"backend": "hf", "response_length": 4},
                    "actor": {"backend": "hf", "ppo_mini_batch_size": 2, "micro_batch_size": 1},
                    "critic": {
                        "backend": "hf",
                        "device": "cpu",
                        "enable": True,
                        "ppo_mini_batch_size": 2,
                        "micro_batch_size": 1,
                        "ppo_epochs": 1,
                        "cliprange_value": 0.5,
                    },
                    "reference": {"enable": False},
                }
            )
            tokenizer = load_tokenizer(config.model)
            batch = self._attach_value_training_fields(
                self._make_batch(tokenizer, ["say yes", "what is two plus two ?"], ["yes", "answer four yes"]),
                returns=[[10.0], [1.0, 1.0, 1.0]],
            )
            value_worker = HFValueWorker(config.model, config.critic)

            def fake_values(microbatch):
                response_length = len(microbatch.batch["responses"][0])
                if response_length == 1:
                    return torch.tensor(
                        [[0.0]],
                        dtype=torch.float32,
                        device=value_worker.device,
                        requires_grad=True,
                    )
                return torch.tensor(
                    [[0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                    device=value_worker.device,
                    requires_grad=True,
                )

            value_worker._compute_response_values = fake_values
            result = value_worker.update(batch)
            self.assertAlmostEqual(result.metrics["value_abs_error"], 3.25, places=6)
            self.assertAlmostEqual(result.metrics["critic_loss"], 25.75, places=6)
            self.assertEqual(len(result.step_metrics), 1)
            self.assertAlmostEqual(result.step_metrics[0]["value_abs_error"], 3.25, places=6)

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

    def test_trainer_can_log_optimizer_steps_and_resume_log_counters(self):
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
                    "model": {"path": str(model_dir), "tokenizer_path": str(model_dir), "dtype": "float32"},
                    "actor": {
                        "backend": "hf",
                        "device": "cpu",
                        "ppo_mini_batch_size": 1,
                        "micro_batch_size": 1,
                        "ppo_epochs": 1,
                    },
                    "critic": {
                        "backend": "hf",
                        "device": "cpu",
                        "enable": True,
                        "ppo_mini_batch_size": 1,
                        "micro_batch_size": 1,
                        "ppo_epochs": 1,
                    },
                    "reference": {"backend": "hf", "device": "cpu", "enable": True},
                    "rollout": {
                        "backend": "hf",
                        "device": "cpu",
                        "response_length": 4,
                        "train": {"do_sample": False, "n": 1, "temperature": 0.0},
                        "validation": {"do_sample": False, "n": 1, "temperature": 0.0},
                    },
                    "trainer": {
                        "total_training_steps": 1,
                        "validate_before_train": False,
                        "test_freq": 0,
                        "save_freq": 1,
                        "default_local_dir": str(checkpoint_dir),
                        "loggers": [],
                        "log_optimizer_steps": True,
                    },
                }
            )
            trainer = build_trainer(config)
            recorder = _RecordingTracker()
            trainer.tracker = recorder
            try:
                trainer.fit()
                self.assertEqual([step for step, _ in recorder.events], [0, 1, 2, 3, 4])
                self.assertEqual(recorder.events[0][1]["critic/optimizer_step"], 1.0)
                self.assertEqual(recorder.events[1][1]["critic/optimizer_step"], 2.0)
                self.assertEqual(recorder.events[2][1]["actor/optimizer_step"], 1.0)
                self.assertEqual(recorder.events[3][1]["actor/optimizer_step"], 2.0)
                self.assertIn("actor/actor_loss", recorder.events[4][1])
                saved_log_step = trainer.log_step
                saved_actor_optimizer_step = trainer.actor_optimizer_step
                saved_critic_optimizer_step = trainer.critic_optimizer_step
            finally:
                trainer.close()

            resumed_trainer = build_trainer(config)
            try:
                self.assertTrue(resumed_trainer.load_checkpoint())
                self.assertEqual(resumed_trainer.log_step, saved_log_step)
                self.assertEqual(resumed_trainer.actor_optimizer_step, saved_actor_optimizer_step)
                self.assertEqual(resumed_trainer.critic_optimizer_step, saved_critic_optimizer_step)
            finally:
                resumed_trainer.close()


if __name__ == "__main__":
    unittest.main()
