"""Dependency-gated tests for the local vLLM rollout backend."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nanoverl.config import ConfigError, TrainerConfig
from nanoverl.core.batch import RLBatch
from nanoverl.rollout.base import SamplingParams
from nanoverl.rollout.sync import PolicySyncer
from nanoverl.trainer import build_trainer

try:  # pragma: no cover - exercised only when optional deps are installed
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from nanoverl.rollout.vllm import VLLMRolloutEngine
    from nanoverl.workers.hf import HFPolicyWorker

    VLLM_TEST_DEPS = True
    VLLM_RUNTIME_READY = torch.cuda.is_available()
except ImportError:  # pragma: no cover - default when optional deps are absent
    VLLM_TEST_DEPS = False
    VLLM_RUNTIME_READY = False


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class VLLMConfigValidationTest(unittest.TestCase):
    def test_debug_actor_cannot_use_vllm_rollout(self):
        with self.assertRaises(ConfigError):
            TrainerConfig.from_dict(
                {
                    "actor": {"backend": "debug"},
                    "critic": {"enable": False},
                    "reference": {"enable": False},
                    "rollout": {"backend": "vllm"},
                }
            )

    def test_hf_actor_can_use_vllm_rollout(self):
        config = TrainerConfig.from_dict(
            {
                "model": {"path": "debug-model"},
                "actor": {"backend": "hf"},
                "critic": {"backend": "hf", "enable": False},
                "reference": {"backend": "hf", "enable": False},
                "rollout": {"backend": "vllm"},
            }
        )
        self.assertEqual(config.rollout.backend, "vllm")

    def test_vllm_rollout_rejects_tensor_parallel_above_one(self):
        with self.assertRaises(ConfigError):
            TrainerConfig.from_dict(
                {
                    "model": {"path": "debug-model"},
                    "actor": {"backend": "hf"},
                    "critic": {"backend": "hf", "enable": False},
                    "reference": {"backend": "hf", "enable": False},
                    "rollout": {"backend": "vllm", "tensor_model_parallel_size": 2},
                }
            )


@unittest.skipUnless(VLLM_TEST_DEPS and VLLM_RUNTIME_READY, "vLLM backend tests require vllm and CUDA.")
class VLLMBackendTest(unittest.TestCase):
    def _make_model_dir(self, root: Path) -> Path:
        model_dir = root / "tiny-vllm"
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
            "answer": 13,
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
        model = GPT2LMHeadModel(config)
        model.save_pretrained(model_dir)
        return model_dir

    def _make_config(self, model_dir: Path, checkpoint_dir: Path, dataset_path: Path) -> TrainerConfig:
        return TrainerConfig.from_dict(
            {
                "data": {
                    "train_path": str(dataset_path),
                    "val_path": str(dataset_path),
                    "train_batch_size": 1,
                    "val_batch_size": 1,
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
                },
                "actor": {
                    "backend": "hf",
                    "mini_batch_size": 1,
                    "update_epochs": 1,
                    "micro_batch_size": 1,
                    "clip_ratio": 0.2,
                    "lr": 1e-4,
                },
                "critic": {
                    "backend": "hf",
                    "enable": False,
                },
                "reference": {
                    "backend": "hf",
                    "enable": False,
                },
                "rollout": {
                    "backend": "vllm",
                    "response_length": 4,
                    "enforce_eager": True,
                    "gpu_memory_utilization": 0.2,
                    "tensor_model_parallel_size": 1,
                    "engine_kwargs": {"max_model_len": 64},
                    "train": {"do_sample": False, "n": 1, "temperature": 0.0},
                    "validation": {"do_sample": False, "n": 1, "temperature": 0.0},
                },
                "trainer": {
                    "total_epochs": 1,
                    "total_training_steps": 1,
                    "validate_before_train": True,
                    "test_freq": 1,
                    "save_freq": 1,
                    "default_local_dir": str(checkpoint_dir),
                    "loggers": [],
                    "project_name": "nanoverl-test",
                    "experiment_name": "vllm-smoke",
                },
            }
        )

    def test_rollout_output_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_model_dir(Path(tmpdir))
            config = TrainerConfig.from_dict(
                {
                    "model": {"path": str(model_dir), "tokenizer_path": str(model_dir), "dtype": "float32"},
                    "data": {"max_prompt_length": 8},
                    "actor": {"backend": "hf", "mini_batch_size": 1},
                    "critic": {"backend": "hf", "enable": False},
                    "reference": {"backend": "hf", "enable": False},
                    "rollout": {
                        "backend": "vllm",
                        "response_length": 4,
                        "enforce_eager": True,
                        "gpu_memory_utilization": 0.2,
                        "tensor_model_parallel_size": 1,
                        "engine_kwargs": {"max_model_len": 64},
                    },
                }
            )
            rollout_engine = VLLMRolloutEngine(config.model, config.data, config.rollout)
            rollout_batch = rollout_engine.generate(
                RLBatch(non_tensor={"prompt": ["say yes"]}),
                SamplingParams(do_sample=False, temperature=0.0, n=1),
            )

            for field_name in (
                "prompts",
                "responses",
                "input_ids",
                "attention_mask",
                "response_mask",
            ):
                self.assertIn(field_name, rollout_batch.batch)
            self.assertIn("response_text", rollout_batch.non_tensor)
            self.assertEqual(len(rollout_batch.batch["responses"]), 1)

    def test_sync_policy_changes_rollout_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_model_dir(Path(tmpdir))
            config = TrainerConfig.from_dict(
                {
                    "model": {"path": str(model_dir), "tokenizer_path": str(model_dir), "dtype": "float32"},
                    "data": {"max_prompt_length": 8},
                    "actor": {"backend": "hf", "mini_batch_size": 1, "lr": 1e-4},
                    "critic": {"backend": "hf", "enable": False},
                    "reference": {"backend": "hf", "enable": False},
                    "rollout": {
                        "backend": "vllm",
                        "response_length": 4,
                        "enforce_eager": True,
                        "gpu_memory_utilization": 0.2,
                        "tensor_model_parallel_size": 1,
                        "engine_kwargs": {"max_model_len": 64},
                    },
                }
            )
            rollout_engine = VLLMRolloutEngine(config.model, config.data, config.rollout)
            policy_worker = HFPolicyWorker(config.model, config.actor)
            before_sync_steps = rollout_engine.state_dict()["policy_sync_steps"]
            with torch.no_grad():
                for parameter in policy_worker.model.parameters():
                    parameter.add_(0.1)
            PolicySyncer().sync(policy_worker, rollout_engine, "test")
            self.assertGreater(rollout_engine.state_dict()["policy_sync_steps"], before_sync_steps)

    def test_hf_actor_vllm_rollout_fit_and_resume(self):
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

            config = self._make_config(model_dir, checkpoint_dir, dataset_path)
            trainer = build_trainer(config)
            try:
                validation_metrics = trainer.fit()
                self.assertIn("val/reward_mean", validation_metrics)
                self.assertEqual(trainer.global_step, 1)
            finally:
                trainer.close()

            resumed_trainer = build_trainer(config)
            try:
                self.assertTrue(resumed_trainer.load_checkpoint())
                self.assertEqual(resumed_trainer.global_step, 1)
            finally:
                resumed_trainer.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
