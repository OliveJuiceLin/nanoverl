"""Local vLLM rollout engine."""

from __future__ import annotations

from typing import Any, Dict, List

from nanoverl.backends.hf import (
    encode_text,
    ensure_prompt_tokens,
    load_tokenizer,
    render_prompt_text,
    resolve_device,
    trim_generated_response,
)
from nanoverl.backends.vllm import (
    build_vllm_ipc_weight_update_request,
    build_vllm_sampling_params,
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

        # This device hook is new because the earlier local rollout path assumed
        # the process-wide current CUDA device. For single-process experiments we
        # still keep vLLM thin, but now we can pin it to one explicit local GPU.
        self.sync_device = resolve_device(rollout_config.device or "cuda")
        if self.sync_device.type != "cuda":
            raise RuntimeError("The thin vLLM rollout backend currently requires a CUDA rollout.device.")
        torch.cuda.set_device(self.sync_device)

        engine_kwargs = dict(rollout_config.engine_kwargs)
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
        prompt_values = batch.non_tensor.get("prompt")
        if prompt_values is None:
            raise ValueError("Batch is missing 'prompt' field in non-tensor data, which is required for VLLMRolloutEngine.")

        prompt_token_rows: List[List[int]] = []
        prompt_inputs: List[Dict[str, List[int]]] = []
        for prompt_value in prompt_values:
            prompt_text = render_prompt_text(self.tokenizer, prompt_value)
            prompt_token_ids = ensure_prompt_tokens(encode_text(self.tokenizer, prompt_text), self.tokenizer)
            prompt_token_rows.append(prompt_token_ids)
            prompt_inputs.append({"prompt_token_ids": prompt_token_ids})

        request_outputs = self.llm.generate(
            prompt_inputs,
            sampling_params=build_vllm_sampling_params(sampling, self.rollout_config.response_length),
        )

        prompts = []
        responses = []
        input_ids = []
        attention_masks = []
        response_masks = []
        response_texts = []
        for row_index, request_output in enumerate(request_outputs):
            completion_output = request_output.outputs[0]
            prompt_ids = prompt_token_rows[row_index]
            response_token_ids = trim_generated_response(self.tokenizer, completion_output.token_ids)
            prompts.append(prompt_ids)
            responses.append(response_token_ids)
            input_ids.append(prompt_ids + response_token_ids)
            attention_masks.append([1] * (len(prompt_ids) + len(response_token_ids)))
            response_masks.append([1] * len(response_token_ids))
            response_texts.append(self.tokenizer.decode(response_token_ids, skip_special_tokens=True).strip())

        rollout_batch = RLBatch(
            batch={
                "prompts": prompts,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "response_mask": response_masks,
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
