"""A small deterministic rollout engine for smoke tests and algorithm debugging."""

from __future__ import annotations

from typing import Any, Dict, List

from nanoverl.core.batch import RLBatch
from nanoverl.rollout.base import RolloutEngine, SamplingParams


class DebugRolloutEngine(RolloutEngine):
    """Produces deterministic text and logprobs from prompt metadata."""

    def __init__(self, max_response_length: int = 64):
        self.max_response_length = max_response_length
        self.policy_version = 0
        self.policy_sync_steps = 0

    @staticmethod
    def _tokenize_text(text: str) -> List[int]:
        # 这里模拟了真实的文本 tokenization 过程，将输入文本拆分成单词（或 token），并为每个 token 分配一个唯一的整数 ID。
        pieces = [piece for piece in text.strip().split() if piece]
        if not pieces:
            return []
        return list(range(1, len(pieces) + 1))

    def generate(self, batch: RLBatch, sampling: SamplingParams) -> RLBatch:
        """
        Function:
            - 从批次中的每一行数据提取提示文本和相关元数据，根据这些信息生成一个响应文本。
            - 将提示文本和响应文本转换为整数 token 列表，模拟文本的 tokenization 过程。
            - 生成一个与响应长度相同的 logprobs 列表，模拟模型在生成响应时的 logprobs 输出。
            - 将生成的 prompts、responses、response_masks、attention_masks 和 rollout_log_probs 组织成一个新的 RLBatch，并将其与原始批次合并返回。
        """
        del sampling
        prompts: List[List[int]] = []
        responses: List[List[int]] = []
        input_ids: List[List[int]] = []
        response_masks: List[List[int]] = []
        attention_masks: List[List[int]] = []
        rollout_log_probs: List[List[float]] = []
        response_texts: List[str] = []

        for index in range(len(batch)):
            prompt_text = str(
                (batch.non_tensor.get("prompt_text", [None] * len(batch))[index])
                or (batch.non_tensor.get("prompt", [None] * len(batch))[index])
                or ""
            )
            rollout_index = int(batch.non_tensor.get("rollout_index", [0] * len(batch))[index])
            scripted = batch.non_tensor.get("scripted_responses", [None] * len(batch))[index] # scipted 意思是预先定义好的响应列表，可能是为了测试不同版本的响应或者模拟模型在不同采样版本下的输出。
            # 这里的逻辑是：如果 scripted_responses 存在且是一个非空列表，就根据 rollout_index 从中选择一个响应文本；
            # 否则，如果 expected_response 存在，就使用它作为响应文本；
            # 如果两者都不存在，则使用一个默认的调试响应文本 "debug-response"。
            if isinstance(scripted, list) and scripted:
                response_text = str(scripted[min(rollout_index, len(scripted) - 1)])
            elif batch.non_tensor.get("expected_response", [None] * len(batch))[index] is not None:
                response_text = str(batch.non_tensor["expected_response"][index])
            else:
                response_text = "debug-response"
            prompt_tokens = self._tokenize_text(prompt_text) # List[int]
            response_tokens = self._tokenize_text(response_text)[: self.max_response_length]
            prompts.append(prompt_tokens)
            responses.append(response_tokens)
            input_ids.append(prompt_tokens + response_tokens)
            response_masks.append([1 for _ in response_tokens])
            attention_masks.append([1 for _ in range(len(prompt_tokens) + len(response_tokens))])
            base_logprob = -0.25 - (0.05 * rollout_index) - (0.01 * self.policy_version)
            rollout_log_probs.append([base_logprob for _ in response_tokens])
            response_texts.append(response_text)

        update = RLBatch(
            batch={
                "prompts": prompts, # List[List[int]]
                "responses": responses, # List[List[int]]
                "input_ids": input_ids,
                "response_mask": response_masks,
                "attention_mask": attention_masks,
                "rollout_log_probs": rollout_log_probs,
            },
            non_tensor={"response_text": response_texts},
            meta={"policy_version": self.policy_version},
        )
        return batch.union(update)

    def sync_policy(self, policy_state: Dict[str, Any]) -> None:
        self.policy_version = int(policy_state.get("version", self.policy_version))
        self.policy_sync_steps += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"policy_sync_steps": self.policy_sync_steps}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.policy_sync_steps = int(state.get("policy_sync_steps", 0))


__all__ = ["DebugRolloutEngine"]
