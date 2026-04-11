"""Shared helpers for the optional local Hugging Face backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


class MissingDependencyError(RuntimeError):
    """Raised when an optional local HF backend dependency is unavailable."""


def require_hf_dependencies():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised when torch is installed
        raise MissingDependencyError(
            "The local HF backend requires torch. Install nanoverl with the 'train' extra first."
        ) from exc
    try:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised when transformers is installed
        raise MissingDependencyError(
            "The local HF backend requires transformers. Install nanoverl with the 'train' extra first."
        ) from exc
    return torch, AutoModel, AutoModelForCausalLM, AutoTokenizer


def resolve_torch_dtype(dtype_name: Optional[str]):
    torch, _, _, _ = require_hf_dependencies()
    if dtype_name is None:
        return torch.float32
    dtype_lookup = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    normalized_name = dtype_name.lower()
    if normalized_name not in dtype_lookup:
        raise ValueError("Unsupported model.dtype: %s" % dtype_name)
    return dtype_lookup[normalized_name]


def get_default_device():
    torch, _, _, _ = require_hf_dependencies()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_device(device_name: Optional[str] = None):
    # This helper is new because the earlier local backend assumed one default
    # device for every HF component. Once actor, reference, critic, and rollout
    # may live on different local devices, device resolution needs one shared path.
    torch, _, _, _ = require_hf_dependencies()
    if not device_name:
        return get_default_device()
    return torch.device(device_name)


def load_tokenizer(model_config):
    _, _, _, auto_tokenizer = require_hf_dependencies()
    tokenizer_path = model_config.tokenizer_path or model_config.path
    tokenizer = auto_tokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    if model_config.chat_template_path:
        tokenizer.chat_template = Path(model_config.chat_template_path).read_text(encoding="utf-8")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer must expose a pad_token, eos_token, or unk_token for local HF rollout.")
    tokenizer.padding_side = "left"
    return tokenizer


def load_causal_lm(model_config):
    _, _, auto_model_for_causal_lm, _ = require_hf_dependencies()
    return auto_model_for_causal_lm.from_pretrained(
        model_config.path,
        dtype=resolve_torch_dtype(model_config.dtype),
        trust_remote_code=model_config.trust_remote_code,
    )


def load_backbone_model(model_config, path: Optional[str] = None):
    _, auto_model, _, _ = require_hf_dependencies()
    model_path = path or model_config.path
    return auto_model.from_pretrained(
        model_path,
        dtype=resolve_torch_dtype(model_config.dtype),
        trust_remote_code=model_config.trust_remote_code,
    )


def ensure_prompt_tokens(prompt_token_ids: Sequence[int], tokenizer) -> List[int]:
    token_ids = list(prompt_token_ids)
    if token_ids:
        return token_ids
    for fallback_token_id in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id):
        if fallback_token_id is not None:
            return [int(fallback_token_id)]
    raise ValueError("Prompt text is empty and tokenizer has no usable fallback token.")


def truncate_prompt(prompt_token_ids: Sequence[int], max_prompt_length: int) -> List[int]:
    if max_prompt_length <= 0:
        return []
    return list(prompt_token_ids)[-max_prompt_length:]


def truncate_response(response_token_ids: Sequence[int], max_response_length: int) -> List[int]:
    if max_response_length <= 0:
        return []
    return list(response_token_ids[:max_response_length])


def pack_prompt_response_tokens(
    tokenizer,
    prompt_token_ids: Sequence[int],
    response_token_ids: Sequence[int],
    max_prompt_length: int,
    max_response_length: int,
) -> Dict[str, List[int]]:
    prompt_ids = truncate_prompt(ensure_prompt_tokens(prompt_token_ids, tokenizer), max_prompt_length)
    response_ids = truncate_response(response_token_ids, max_response_length)
    return {
        "prompts": prompt_ids,
        "responses": response_ids,
        "input_ids": prompt_ids + response_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(response_ids)),
        "response_mask": [1] * len(response_ids),
    }


def encode_text(tokenizer, text: str) -> List[int]:
    # encoded = tokenizer(text, add_special_tokens=False)
    # return list(encoded["input_ids"])
    # 更简洁的实现：
    return tokenizer.encode(text, add_special_tokens=False)


def render_prompt_text(tokenizer, prompt_value: Any) -> str:
    # 究竟是使用 template 还是纯字符串取决于数据本身是 str 还是一个符合模板的结构
    # 换句话说，在数据处理阶段就决定好了模型的输入格式。
    if isinstance(prompt_value, str):
        return prompt_value
    # if tokenizer.chat_template is None:
    #     return str(prompt_value)
    if isinstance(prompt_value, Mapping):
        prompt_value = [dict(prompt_value)]
    # TODO: 这里有一个问题，apply_chat_template 会假如一个 system_prompt，有可能和我们的要求冲突
    if isinstance(prompt_value, (list, tuple)):
        return str(tokenizer.apply_chat_template(prompt_value, tokenize=False, add_generation_prompt=True))
    return str(prompt_value)


def trim_generated_response(tokenizer, response_token_ids: Sequence[int]) -> List[int]:
    # This helper is new because the earlier rollout path treated everything
    # after the prompt as valid response tokens. Real generation needs one place
    # to stop at EOS and ignore post-generation padding.
    trimmed = list(response_token_ids)
    eos_token_id = tokenizer.eos_token_id # '<|im_end|>'
    pad_token_id = tokenizer.pad_token_id # '<|endoftext|>'
    if eos_token_id is not None and eos_token_id in trimmed:
        trimmed = trimmed[: trimmed.index(eos_token_id) + 1]
    if pad_token_id is not None and pad_token_id != eos_token_id:
        while trimmed and trimmed[-1] == pad_token_id:
            trimmed.pop()
    return trimmed


def pad_rows(rows: Sequence[Sequence[Any]], pad_value: Any, padding_side: str = "right") -> List[List[Any]]:
    if not rows:
        return []
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'.")
    max_row_length = max(len(row) for row in rows)
    padded_rows: List[List[Any]] = []
    for row in rows:
        row_values = list(row)
        padding = [pad_value] * (max_row_length - len(row_values))
        padded_rows.append(padding + row_values if padding_side == "left" else row_values + padding)
    return padded_rows


def batch_lists_to_tensor(
    rows: Sequence[Sequence[Any]],
    pad_value: Any,
    device,
    dtype=None,
    padding_side: str = "right",
):
    torch, _, _, _ = require_hf_dependencies()
    padded_rows = pad_rows(rows, pad_value, padding_side=padding_side)
    tensor_kwargs: Dict[str, Any] = {"device": device}
    if dtype is not None:
        tensor_kwargs["dtype"] = dtype
    return torch.tensor(padded_rows, **tensor_kwargs)


def get_prompt_lengths(batch) -> List[int]:
    return [len(prompt_ids) for prompt_ids in batch.batch["prompts"]]


def get_response_lengths(batch) -> List[int]:
    # 语义等价于直接写成 [sum(response_mask) for response_mask in batch.batch["response_mask"]] 
    # 但是采用了防御性编程的方式，确保即使response_mask中的元素不是int类型（例如bool类型），也能正确计算响应长度。
    return [sum(int(token_mask) for token_mask in response_mask) for response_mask in batch.batch["response_mask"]]


def build_training_tensors(batch, device):
    """
    Function:
        - 从batch中提取训练所需的各种张量，包括response_mask、old_log_probs、advantages、returns和ref_log_probs，并将它们转换为适当的PyTorch张量，准备好在训练过程中使用。
    """
    torch, _, _, _ = require_hf_dependencies()
    return {
        "response_mask": batch_lists_to_tensor(batch.batch["response_mask"], 0, device=device, dtype=torch.float32),
        "old_log_probs": batch_lists_to_tensor(batch.batch.get("old_log_probs", []), 0.0, device=device, dtype=torch.float32),
        "advantages": batch_lists_to_tensor(batch.batch.get("advantages", []), 0.0, device=device, dtype=torch.float32),
        "returns": batch_lists_to_tensor(batch.batch.get("returns", []), 0.0, device=device, dtype=torch.float32),
        "ref_log_probs": batch_lists_to_tensor(batch.batch.get("ref_log_probs", []), 0.0, device=device, dtype=torch.float32),
    }


def extract_response_stats(
    logits,
    input_ids,
    prompt_token_counts: Sequence[int],
    response_token_counts: Sequence[int],
    compute_entropy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Args:
        - logits: 完整一句话的logits，shape为 (batch_size, sequence_length, vocab_size)
        - input_ids: 完整一句话的token ids，shape为 (batch_size, sequence_length)
        - prompt_token_counts: 每个样本中prompt部分的token数量，shape为 (batch_size,)
        - response_token_counts: 每个样本中response部分的token数量，shape为 (batch_size,)
     Returns:
        - response_log_probs: 每个样本中response部分的log_prob，shape为 (batch_size, max_response_length)，其中max_response_length是response_token_counts中的最大值
        - response_entropy: 每个样本中response部分的token熵值，shape为 (batch_size, max_response_length)
    """
    torch, _, _, _ = require_hf_dependencies()
    shifted_logits = logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    token_log_normalizers = torch.logsumexp(shifted_logits, dim=-1) # shape为 (batch_size, sequence_length - 1)
    selected_target_logits = torch.gather(shifted_logits, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    target_log_probs = selected_target_logits - token_log_normalizers
    # 取出对应response部分的log_prob，shape为 (batch_size, max_response_length)，其中max_response_length是response_token_counts中的最大值
    batch_size = input_ids.shape[0]
    max_response_length = max(response_token_counts, default=0)
    response_log_probs = target_log_probs.new_zeros((batch_size, max_response_length)) # 继承数据类型和设备
    response_entropy = None
    token_entropy = None
    if compute_entropy:
        shifted_probs = torch.softmax(shifted_logits, dim=-1)
        token_entropy = token_log_normalizers - (shifted_probs * shifted_logits).sum(dim=-1)
        response_entropy = token_log_normalizers.new_zeros((batch_size, max_response_length))

    for row_index, (prompt_token_count, response_token_count) in enumerate(
        zip(prompt_token_counts, response_token_counts)
    ):
        if response_token_count <= 0:
            continue
        response_start = max(prompt_token_count - 1, 0) # 指向 prompt 最后
        response_stop = response_start + response_token_count
        response_log_probs[row_index, :response_token_count] = target_log_probs[row_index, response_start:response_stop]
        if response_entropy is not None and token_entropy is not None:
            response_entropy[row_index, :response_token_count] = token_entropy[row_index, response_start:response_stop]
    return response_log_probs, response_entropy


def tensor_to_list_rows(matrix, lengths: Sequence[int]) -> List[List[float]]:
    """
    Function:
        将一个二维tensor转换为一个列表的列表，其中每个子列表的长度由对应的lengths指定。原本的 tensor每一行的长度都一样，但转换后的列表中每行的长度可能不同，取决于lengths中的值。
    """
    rows: List[List[float]] = []
    for row_values, valid_length in zip(matrix.detach().cpu().tolist(), lengths):
        rows.append([float(value) for value in row_values[:valid_length]])
    return rows


def count_valid_tokens(response_masks: Sequence[Sequence[int]]) -> int:
    return sum(sum(int(token_mask) for token_mask in response_mask) for response_mask in response_masks)


def get_loss_weight(batch, loss_agg_mode: str) -> float:
    """
    Function:
        - 根据loss_agg_mode的不同，计算并返回一个用于加权损失
    Ruturn:
        - 如果loss_agg_mode是"token-mean"，则返回batch中所有response_mask中有效token的总数（即所有response_mask中值为1的元素的数量）。这个值可以用来对损失进行加权，使得损失的平均值是每个有效token的平均损失。
        - 如果loss_agg_mode不是"token-mean"，则返回batch的长度（即样本数量）。
    """
    if loss_agg_mode == "token-mean":
        return float(count_valid_tokens(batch.batch["response_mask"]))
    return float(len(batch))


def average_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def clone_model_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    torch, _, _, _ = require_hf_dependencies()
    cloned_state: Dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned_state[key] = value.detach().cpu().clone()
        else:
            cloned_state[key] = value
    return cloned_state
