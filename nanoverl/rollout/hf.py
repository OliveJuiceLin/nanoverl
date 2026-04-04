"""Local Hugging Face rollout engine."""

from __future__ import annotations

from typing import Any, Dict, List

from nanoverl.backends.hf import (
    batch_lists_to_tensor,
    clone_model_state,
    encode_text,
    extract_response_stats,
    get_default_device,
    load_causal_lm,
    load_tokenizer,
    pack_prompt_response_tokens,
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
        self.device = rollout_config.get("device", get_default_device())
        self.tokenizer = load_tokenizer(model_config)
        self.model = load_causal_lm(model_config).to(self.device)
        self.model.eval()
        self.policy_sync_steps = 0

    def generate(self, batch: RLBatch, sampling) -> RLBatch:
        """加入字段 prompts、responses、input_ids(、attention_mask、response_mask 和 rollout_log_probs) 到 batch 中："""
        torch, _, _, _ = self._dependencies()
        prompts = batch.non_tensor.get("prompt") # 可能是 List[str]，也可能是 List[dict], 取决于 tokenizer 有没有模板
        if prompts is None:
            raise ValueError("Rollout requires prompt_text or prompt in batch.non_tensor.")

        prompt_token_ids: List[List[int]] = []
        for prompt in prompts:
            if self.tokenizer.chat_template is not None:
                prompt_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = prompt
            packed_prompt = pack_prompt_response_tokens(
                tokenizer=self.tokenizer,
                prompt_token_ids=encode_text(self.tokenizer, str(prompt_text)),
                response_token_ids=[],
                max_prompt_length=self.data_config.max_prompt_length,
                max_response_length=self.rollout_config.response_length,
            )
            prompt_token_ids.append(packed_prompt["prompts"])

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
        packed_rows = []
        response_texts = []
        for row_index, generated in enumerate(generated_token_ids):
            full_token_ids = generated.detach().cpu().tolist()
            response_token_ids = full_token_ids[padded_prompt_len :] # 会包含生成的 response 和后续的 padding，需要截断掉 padding 部分

            # 找到第一个 eos_token_id 并截断，避免将生成的 padding 数据当作有效 response
            if self.tokenizer.eos_token_id in response_token_ids:
                eos_idx = response_token_ids.index(self.tokenizer.eos_token_id)
                response_token_ids = response_token_ids[: eos_idx + 1]
            packed_row = pack_prompt_response_tokens(
                tokenizer=self.tokenizer,
                prompt_token_ids=prompt_token_ids[row_index],
                response_token_ids=response_token_ids,
                max_prompt_length=self.data_config.max_prompt_length,
                max_response_length=self.rollout_config.response_length,
            )
            packed_rows.append(packed_row)
            response_texts.append(self.tokenizer.decode(packed_row["responses"], skip_special_tokens=True).strip())

        # 注：在 Trainer 中 PolicyWorker 会重新进行一次前向传播计算 old_log_probs，那么在 Rollout 阶段计算的 logits 和 rollout_log_probs 是多余的
        # 标准的强化学习训练（如使用大型并行框架时），通常 Rollout 引擎（比如 vLLM）在生成时顺便把对数概率（Log Probs）算出来返回，作为 PPO 的 old_log_probs，以节省一次前向传播。
        # logits的计算和rollout_log_probs的记录实际没有必要，后续可以考虑优化掉
        # full_input_ids = batch_lists_to_tensor(
        #     [row["input_ids"] for row in packed_rows],
        #     self.tokenizer.pad_token_id,
        #     device=self.device,
        #     dtype=torch.long,
        #     padding_side="right",
        # )
        # full_attention_mask = batch_lists_to_tensor(
        #     [row["attention_mask"] for row in packed_rows],
        #     0,
        #     device=self.device,
        #     dtype=torch.long,
        #     padding_side="right",
        # )
        # with torch.no_grad():
        #     logits = self.model(input_ids=full_input_ids, attention_mask=full_attention_mask).logits
        # response_log_probs, _ = extract_response_stats(
        #     logits=logits,
        #     input_ids=full_input_ids,
        #     prompt_token_counts=[len(row["prompts"]) for row in packed_rows],
        #     response_token_counts=[len(row["responses"]) for row in packed_rows],
        # )

        rollout_batch = RLBatch(
            batch={
                "prompts": [row["prompts"] for row in packed_rows],
                "responses": [row["responses"] for row in packed_rows],
                "input_ids": [row["input_ids"] for row in packed_rows],
                "attention_mask": [row["attention_mask"] for row in packed_rows],
                "response_mask": [row["response_mask"] for row in packed_rows],
                # "rollout_log_probs": tensor_to_list_rows(
                #     response_log_probs,
                #     [len(row["responses"]) for row in packed_rows],
                # ),
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
