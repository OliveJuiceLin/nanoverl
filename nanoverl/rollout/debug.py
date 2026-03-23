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

    @staticmethod
    def _tokenize_text(text: str) -> List[int]:
        pieces = [piece for piece in text.strip().split() if piece]
        if not pieces:
            return []
        return list(range(1, len(pieces) + 1))

    def generate(self, batch: RLBatch, sampling: SamplingParams) -> RLBatch:
        del sampling
        prompts: List[List[int]] = []
        responses: List[List[int]] = []
        response_masks: List[List[int]] = []
        attention_masks: List[List[int]] = []
        rollout_log_probs: List[List[float]] = []
        response_texts: List[str] = []

        for index in range(len(batch)):
            row = batch.row(index)
            prompt_text = str(row.get("prompt_text") or row.get("prompt") or "")
            rollout_index = int(row.get("rollout_index", 0))
            scripted = row.get("scripted_responses")
            if isinstance(scripted, list) and scripted:
                response_text = str(scripted[min(rollout_index, len(scripted) - 1)])
            elif row.get("expected_response") is not None:
                response_text = str(row["expected_response"])
            else:
                response_text = "debug-response"
            prompt_tokens = self._tokenize_text(prompt_text)
            response_tokens = self._tokenize_text(response_text)[: self.max_response_length]
            prompts.append(prompt_tokens)
            responses.append(response_tokens)
            response_masks.append([1 for _ in response_tokens])
            attention_masks.append([1 for _ in range(len(prompt_tokens) + len(response_tokens))])
            base_logprob = -0.25 - (0.05 * rollout_index) - (0.01 * self.policy_version)
            rollout_log_probs.append([base_logprob for _ in response_tokens])
            response_texts.append(response_text)

        update = RLBatch(
            batch={
                "prompts": prompts,
                "responses": responses,
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

    def state_dict(self) -> Dict[str, Any]:
        return {"policy_version": self.policy_version}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.policy_version = int(state.get("policy_version", 0))


__all__ = ["DebugRolloutEngine"]
