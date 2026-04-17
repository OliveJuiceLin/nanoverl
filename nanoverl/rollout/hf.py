"""Local Hugging Face rollout engine."""

from __future__ import annotations

from typing import Any, Dict, List

from nanoverl.backends.hf import (
    batch_lists_to_tensor,
    clone_model_state,
    encode_text,
    ensure_prompt_tokens,
    load_causal_lm,
    load_tokenizer,
    render_prompt_text,
    resolve_device,
    trim_generated_response,
)
from nanoverl.core.batch import RLBatch
from nanoverl.rollout.base import RolloutEngine


class HFRolloutEngine(RolloutEngine):
    """In-process decoder-only rollout using transformers.generate()."""

    def __init__(self, model_config, data_config, rollout_config):
        self.model_config = model_config
        self.data_config = data_config
        self.rollout_config = rollout_config
        self.device = resolve_device(rollout_config.device)
        self.tokenizer = load_tokenizer(model_config)
        self.model = load_causal_lm(model_config).to(self.device)
        self.model.eval()
        self.policy_sync_steps = 0

    def generate(self, batch: RLBatch, sampling) -> RLBatch:
        # This method changed after the first HF rollout pass because real chat
        # templates and post-generation padding need to be handled before the
        # rollout batch is turned back into the shared training fields.
        torch, _, _, _ = self._dependencies()
        prompt_values = batch.non_tensor.get("prompt")
        if prompt_values is None:
            raise ValueError("Batch is missing 'prompt' field in non-tensor data, which is required for HFRolloutEngine.")

        prompt_token_ids: List[List[int]] = []
        for prompt_value in prompt_values:
            prompt_text = render_prompt_text(self.tokenizer, prompt_value)
            prompt_token_ids.append(ensure_prompt_tokens(encode_text(self.tokenizer, prompt_text), self.tokenizer))

        prompt_input_ids = batch_lists_to_tensor(
            prompt_token_ids,
            self.tokenizer.pad_token_id,
            device=self.device,
            dtype=torch.long,
            padding_side="left",
        )
        prompt_attention_mask = (prompt_input_ids != self.tokenizer.pad_token_id).long()

        generation_kwargs: Dict[str, Any] = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,

            "max_new_tokens": self.rollout_config.response_length,
            "do_sample": sampling.do_sample,
            "top_p": sampling.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if sampling.do_sample:
            generation_kwargs["temperature"] = sampling.temperature
        if sampling.top_k is not None and sampling.top_k > 0:
            generation_kwargs["top_k"] = sampling.top_k

        with torch.no_grad():
            generated_token_ids = self.model.generate(**generation_kwargs)

        padded_prompt_len = prompt_input_ids.shape[1]
        prompts = []
        responses = []
        input_ids = []
        attention_masks = []
        response_masks = []
        response_texts = []
        for row_index, generated in enumerate(generated_token_ids):
            full_token_ids = generated.detach().cpu().tolist()
            prompt_ids = prompt_token_ids[row_index]
            response_token_ids = trim_generated_response(self.tokenizer, full_token_ids[padded_prompt_len:])
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
        model_state = policy_state.get("model_state")
        if model_state is not None:
            self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()
        self.policy_sync_steps += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model_state": clone_model_state(self.model.state_dict()),
            "policy_sync_steps": self.policy_sync_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("model_state") is not None:
            self.model.load_state_dict(state["model_state"])
        self.policy_sync_steps = int(state.get("policy_sync_steps", 0))
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _dependencies():
        from nanoverl.backends.hf import require_hf_dependencies

        return require_hf_dependencies()


__all__ = ["HFRolloutEngine"]
