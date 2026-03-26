"""Local Hugging Face rollout engine."""

from __future__ import annotations

from typing import Any, Dict, List

from nanoverl.backends.hf import (
    batch_lists_to_tensor,
    clone_model_state,
    default_device,
    encode_text,
    extract_response_stats,
    load_causal_lm,
    load_tokenizer,
    pack_prompt_response_tokens,
    prompt_lengths,
    require_hf_dependencies,
    response_lengths,
    tensor_to_list_rows,
)
from nanoverl.core.batch import RLBatch
from nanoverl.rollout.base import RolloutEngine


class HFRolloutEngine(RolloutEngine):
    """In-process decoder-only rollout using transformers.generate()."""

    def __init__(self, model_config, data_config, rollout_config):
        self.model_config = model_config
        self.data_config = data_config
        self.rollout_config = rollout_config
        
        self.device = default_device()
        self.tokenizer = load_tokenizer(model_config)
        self.model = load_causal_lm(model_config).to(self.device)
        self.model.eval()

        self.sync_count = 0 # Number of times the policy has been synced, for tracking staleness of rollouts.

    def generate(self, batch: RLBatch, sampling) -> RLBatch:
        torch, _, _, _ = require_hf_dependencies()
        prompt_texts = batch.non_tensor.get("prompt_text") or batch.non_tensor.get("prompt") # List[str]
        if prompt_texts is None:
            raise ValueError("Rollout requires prompt_text or prompt in batch.non_tensor.")

        prompt_token_rows: List[List[int]] = []
        for prompt_text in prompt_texts:
            prompt_tokens = encode_text(self.tokenizer, str(prompt_text))
            prompt_token_rows.append(
                pack_prompt_response_tokens(
                    tokenizer=self.tokenizer,
                    prompt_tokens=prompt_tokens,
                    response_tokens=[],
                    max_prompt_length=self.data_config.max_prompt_length,
                    max_response_length=self.rollout_config.response_length,
                )["prompts"] # 一个被截断的prompt
            )

        prompt_input_ids: torch.Tensor = batch_lists_to_tensor(
            prompt_token_rows,
            self.tokenizer.pad_token_id,
            device=self.device,
            dtype=torch.long,
        )
        # 下面两行的功能一样
        # prompt_attention_mask = batch_lists_to_tensor(prompt_token_rows, 0, device=self.device, dtype=torch.long)
        prompt_attention_mask = (prompt_input_ids != self.tokenizer.pad_token_id).long()

        generation_kwargs: Dict[str, Any] = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": self.rollout_config.response_length,
            "do_sample": sampling.do_sample,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if sampling.top_k is not None and sampling.top_k > 0:
            generation_kwargs["top_k"] = sampling.top_k
        if not sampling.do_sample and "temperature" in generation_kwargs:
            generation_kwargs.pop("temperature", None)

        with torch.no_grad():
            generated = self.model.generate(**generation_kwargs)

        prompt_lengths_list = [len(row) for row in prompt_token_rows]
        prompt_rows: List[List[int]] = []
        response_rows: List[List[int]] = []
        response_texts: List[str] = []
        input_ids_rows: List[List[int]] = []
        attention_mask_rows: List[List[int]] = []
        response_mask_rows: List[List[int]] = []

        for row_index, prompt_len in enumerate(prompt_lengths_list):
            generated_row = generated[row_index].detach().cpu().tolist()
            response_tokens = generated_row[prompt_len:]
            packed = pack_prompt_response_tokens(
                tokenizer=self.tokenizer,
                prompt_tokens=prompt_token_rows[row_index],
                response_tokens=response_tokens,
                max_prompt_length=self.data_config.max_prompt_length,
                max_response_length=self.rollout_config.response_length,
            )
            prompt_rows.append(packed["prompts"])
            response_rows.append(packed["responses"])
            input_ids_rows.append(packed["input_ids"])
            attention_mask_rows.append(packed["attention_mask"])
            response_mask_rows.append(packed["response_mask"])
            response_texts.append(self.tokenizer.decode(packed["responses"], skip_special_tokens=True).strip())

        input_ids = batch_lists_to_tensor(
            input_ids_rows,
            self.tokenizer.pad_token_id,
            device=self.device,
            dtype=torch.long,
        )
        attention_mask = batch_lists_to_tensor(attention_mask_rows, 0, device=self.device, dtype=torch.long)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        response_log_probs, _ = extract_response_stats(
            logits=logits,
            input_ids=input_ids,
            prompt_lens=[len(row) for row in prompt_rows],
            response_lens=[len(row) for row in response_rows],
        )

        update = RLBatch(
            batch={
                "prompts": prompt_rows,
                "responses": response_rows,
                "input_ids": input_ids_rows,
                "attention_mask": attention_mask_rows,
                "response_mask": response_mask_rows,
                "rollout_log_probs": tensor_to_list_rows(response_log_probs, [len(row) for row in response_rows]),
            },
            non_tensor={"response_text": response_texts},
            meta={"sync_count": self.sync_count},
        )
        return batch.union(update)

    def sync_policy(self, policy_state: Dict[str, Any]) -> None:
        model_state = policy_state.get("model_state")
        if model_state is not None:
            self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()
        self.sync_count += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model_state": clone_model_state(self.model.state_dict()),
            "sync_count": self.sync_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("model_state") is not None:
            self.model.load_state_dict(state["model_state"])
        self.sync_count = int(state.get("sync_count", 0))
        self.model.to(self.device)
        self.model.eval()


__all__ = ["HFRolloutEngine"]
