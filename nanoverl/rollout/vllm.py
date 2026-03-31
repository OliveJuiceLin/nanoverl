"""Local vLLM rollout engine."""

from __future__ import annotations

from typing import Any, Dict, List

from nanoverl.backends.hf import encode_text, load_tokenizer, pack_prompt_response_tokens
from nanoverl.backends.vllm import (
    build_vllm_ipc_weight_update_request,
    build_vllm_sampling_params,
    extract_vllm_chosen_token_log_probs,
    require_vllm_dependencies,
)
from nanoverl.core.batch import RLBatch
from nanoverl.rollout.base import RolloutEngine


class VLLMRolloutEngine(RolloutEngine):
    """In-process decoder-only rollout using vLLM.generate()."""

    def __init__(self, model_config, data_config, rollout_config):
        self.model_config = model_config
        self.data_config = data_config
        self.rollout_config = rollout_config
        self.tokenizer = load_tokenizer(model_config)
        self.policy_sync_steps = 0

        torch, llm_cls, _ = require_vllm_dependencies()
        if not torch.cuda.is_available():
            raise RuntimeError("The thin vLLM rollout backend currently requires CUDA.")

        engine_kwargs = dict(rollout_config.engine_kwargs)
        self.sync_device = torch.device("cuda", torch.cuda.current_device())
        self.llm = llm_cls(
            model=model_config.path,
            tokenizer=model_config.tokenizer_path or model_config.path,
            trust_remote_code=model_config.trust_remote_code,
            dtype=model_config.dtype,
            tensor_parallel_size=rollout_config.tensor_model_parallel_size,
            gpu_memory_utilization=rollout_config.gpu_memory_utilization,
            enforce_eager=rollout_config.enforce_eager,
            weight_transfer_config={"backend": "ipc"},
            **engine_kwargs,
        )
        self.llm.init_weight_transfer_engine({"init_info": {}})

    def generate(self, batch: RLBatch, sampling) -> RLBatch:
        # This method is new in Phase 3 because rollout can now come from vLLM
        # while the trainer, reward path, and worker contracts stay unchanged.
        prompt_texts: List[str] = batch.non_tensor.get("prompt_text") or batch.non_tensor.get("prompt")
        if prompt_texts is None:
            raise ValueError("Rollout requires prompt_text or prompt in batch.non_tensor.")

        prompt_token_rows: List[List[int]] = []
        prompt_inputs: List[Dict[str, List[int]]] = []
        for prompt_text in prompt_texts:
            packed_prompt = pack_prompt_response_tokens(
                tokenizer=self.tokenizer,
                prompt_token_ids=encode_text(self.tokenizer, str(prompt_text)),
                response_token_ids=[],
                max_prompt_length=self.data_config.max_prompt_length,
                max_response_length=self.rollout_config.response_length,
            )
            prompt_token_rows.append(packed_prompt["prompts"])
            prompt_inputs.append({"prompt_token_ids": packed_prompt["prompts"]})

        request_outputs = self.llm.generate(
            prompt_inputs,
            sampling_params=build_vllm_sampling_params(sampling, self.rollout_config.response_length),
        )

        packed_rows = []
        response_texts = []
        rollout_log_prob_rows = []
        for row_index, request_output in enumerate(request_outputs):
            completion_output = request_output.outputs[0]
            packed_row = pack_prompt_response_tokens(
                tokenizer=self.tokenizer,
                prompt_token_ids=prompt_token_rows[row_index],
                response_token_ids=list(completion_output.token_ids),
                max_prompt_length=self.data_config.max_prompt_length,
                max_response_length=self.rollout_config.response_length,
            )
            packed_rows.append(packed_row)
            response_texts.append(
                self.tokenizer.decode(packed_row["responses"], skip_special_tokens=True).strip()
            )
            response_length = len(packed_row["responses"])
            rollout_log_prob_rows.append(
                extract_vllm_chosen_token_log_probs(completion_output)[:response_length]
            )

        rollout_batch = RLBatch(
            batch={
                "prompts": [row["prompts"] for row in packed_rows],
                "responses": [row["responses"] for row in packed_rows],
                "input_ids": [row["input_ids"] for row in packed_rows],
                "attention_mask": [row["attention_mask"] for row in packed_rows],
                "response_mask": [row["response_mask"] for row in packed_rows],
                "rollout_log_probs": rollout_log_prob_rows,
            },
            non_tensor={"response_text": response_texts},
            meta={"policy_sync_steps": self.policy_sync_steps},
        )
        return batch.union(rollout_batch)

    def sync_policy(self, policy_state: Dict[str, Any]) -> None:
        # This method is new in Phase 3 because vLLM keeps its own inference-side
        # model and must be refreshed through `LLM.update_weights(...)` instead of
        # a direct PyTorch `load_state_dict`.
        model_state = policy_state.get("model_state")
        if model_state is None:
            return

        torch, _, _ = require_vllm_dependencies()
        update_request, resident_tensors = build_vllm_ipc_weight_update_request(model_state, self.sync_device)
        self.llm.update_weights(update_request)
        torch.cuda.synchronize(self.sync_device)
        del resident_tensors
        self.policy_sync_steps += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"policy_sync_steps": self.policy_sync_steps}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.policy_sync_steps = int(state.get("policy_sync_steps", 0))


__all__ = ["VLLMRolloutEngine"]
