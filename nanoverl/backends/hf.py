"""Shared helpers for the optional local Hugging Face backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


def torch_dtype_from_string(dtype_name: str | None):
    torch, _, _, _ = require_hf_dependencies()
    if dtype_name is None:
        return torch.float32
    normalized = dtype_name.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError("Unsupported model.dtype: %s" % dtype_name)
    return mapping[normalized]


def default_device():
    torch, _, _, _ = require_hf_dependencies()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    return tokenizer


def load_causal_lm(model_config):
    _, _, auto_model, _ = require_hf_dependencies()
    return auto_model.from_pretrained(
        model_config.path,
        torch_dtype=torch_dtype_from_string(model_config.dtype),
        trust_remote_code=model_config.trust_remote_code,
    )


def load_backbone_model(model_config, path: Optional[str] = None):
    _, auto_model, _, _ = require_hf_dependencies()
    model_path = path or model_config.path
    return auto_model.from_pretrained(
        model_path,
        torch_dtype=torch_dtype_from_string(model_config.dtype),
        trust_remote_code=model_config.trust_remote_code,
    )


def ensure_prompt_tokens(prompt_tokens: Sequence[int], tokenizer) -> List[int]:
    """
    作用：确保提示令牌的有效性。如果提供了非空的提示令牌，则直接返回它们。
    否则，尝试使用tokenizer的bos_token_id、eos_token_id或pad_token_id作为回退令牌。如果这些都不可用，则引发错误。
    """
    tokens = list(prompt_tokens)
    if tokens:
        return tokens
    for token_id in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id):
        if token_id is not None:
            return [int(token_id)]
    raise ValueError("Prompt text is empty and tokenizer has no usable fallback token.")


def truncate_prompt(prompt_tokens: Sequence[int], max_prompt_length: int) -> List[int]:
    """
    Function:
        Truncate the prompt tokens to the specified maximum length. 
        If the max_prompt_length is less than or equal to 0, return an empty list.
    """
    
    tokens = list(prompt_tokens)
    if max_prompt_length <= 0:
        return []
    # 提取列表中的最后 max_prompt_length 个元素。
    # 这在处理 LLM 输入时很常见，用于确保 prompt 不会超过模型的最大上下文长度，通常保留最近（末尾）的 token。
    # 如果 max_prompt_length 大于列表的总长度，Python 会自动将其视为从列表的起始位置（索引 0）开始截取。在这种情况下，会得到整个列表的副本。
    return tokens[-max_prompt_length:]


def truncate_response(response_tokens: Sequence[int], max_response_length: int) -> List[int]:
    if max_response_length <= 0:
        return []
    return list(response_tokens[:max_response_length])


def pack_prompt_response_tokens(
    tokenizer,
    prompt_tokens: Sequence[int],
    response_tokens: Sequence[int],
    max_prompt_length: int,
    max_response_length: int,
) -> Dict[str, List[int]]:
    """
    Function:
        - 将提示和响应令牌打包成一个结构化的字典，包含截断后的提示、响应、输入 ID、注意力掩码和响应掩码
    """
    prompt = truncate_prompt(ensure_prompt_tokens(prompt_tokens, tokenizer), max_prompt_length)
    response = truncate_response(response_tokens, max_response_length)
    return {
        "prompts": prompt,
        "responses": response,
        "input_ids": prompt + response,
        "attention_mask": [1] * (len(prompt) + len(response)),
        "response_mask": [1] * len(response),
    }


def encode_text(tokenizer, text: str) -> List[int]:
    """
    Function:
        str -> List[int]
    """
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def pad_rows(rows: Sequence[Sequence[Any]], pad_value: Any) -> List[List[Any]]:
    """
    Function:
        使用最大长度度对行进行填充，使它们具有相同的长度。每行将被填充 pad_value 直到达到最大长度。
    """
    if not rows:
        return []
    width = max(len(row) for row in rows)
    # 如果 row 已经是 list，list(row) 的唯一作用是做了一次显式的浅拷贝，
    # 但在包含 + 运算的推导式中，这个拷贝通常是多余的。
    # 如果 row 可能是元组或其他不可变序列，list(row) 可以确保我们得到一个可变的列表来进行连接操作。
    return [list(row) + [pad_value] * (width - len(row)) for row in rows]


def batch_lists_to_tensor(
    rows: Sequence[Sequence[Any]],
    pad_value: Any,
    device,
    dtype=None,
) -> torch.Tensor:
    """
    Function:
        - 将一批列表转换为 PyTorch 张量。首先使用 pad_rows 函数对列表进行(右)填充，使它们具有相同的长度，然后将填充后的列表转换为 PyTorch 张量。
        - 可以指定填充值、设备和数据类型。
    """
    torch, _, _, _ = require_hf_dependencies()
    padded = pad_rows(rows, pad_value)
    kwargs: Dict[str, Any] = {"device": device}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return torch.tensor(padded, **kwargs)


def response_lengths(batch) -> List[int]:
    return [sum(int(mask) for mask in row) for row in batch.batch["response_mask"]]


def prompt_lengths(batch) -> List[int]:
    return [len(row) for row in batch.batch["prompts"]]


def build_response_tensors(batch, device):
    torch, _, _, _ = require_hf_dependencies()
    mask = batch_lists_to_tensor(batch.batch["response_mask"], 0, device=device, dtype=torch.float32)
    old_log_probs = batch_lists_to_tensor(batch.batch.get("old_log_probs", []), 0.0, device=device, dtype=torch.float32)
    advantages = batch_lists_to_tensor(batch.batch.get("advantages", []), 0.0, device=device, dtype=torch.float32)
    returns = batch_lists_to_tensor(batch.batch.get("returns", []), 0.0, device=device, dtype=torch.float32)
    ref_log_probs = batch_lists_to_tensor(batch.batch.get("ref_log_probs", []), 0.0, device=device, dtype=torch.float32)
    return {
        "response_mask": mask,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
        "returns": returns,
        "ref_log_probs": ref_log_probs,
    }


def extract_response_stats(
    logits,
    input_ids,
    prompt_lens: Sequence[int],
    response_lens: Sequence[int],
):
    """
    Function:
    - 从模型的输出 logits 和输入 input_ids 中提取与响应相关的统计信息，包括响应的对数概率和熵。
    - 计算每个时间步的对数概率和熵，然后根据提示长度和响应长度将这些统计信息提取到单独的张量中，以便后续处理。
    Args:
    - logits: 模型输出的原始 logits，形状为 (batch_size, sequence_length, vocab_size)。
    - input_ids: 模型输入的 token ID，形状为 (batch_size, sequence_length)。
    - prompt_lens: 每个样本的提示长度列表，表示每个样本中提示部分的 token 数量。
    - response_lens: 每个样本的响应长度列表，表示每个样本中响应部分的 token 数量。
    Returns:
    - response_log_probs: 包含响应部分的对数概率的张量，形状为 (batch_size, max_response_length)，其中 max_response_length 是响应长度的最大值。
    - response_entropy: 包含响应部分的熵的张量，形状为 (batch_size, max_response_length)，其中 max_response_length 是响应长度的最大值。
    """
    torch, _, _, _ = require_hf_dependencies()
    # logits[:, :-1, :] 的形状为 (batch_size, sequence_length - 1, vocab_size)，
    # 因为在语言模型中，通常会将输入序列的最后一个 token 的 logits 排除在外，因为它没有对应的下一个 token 来计算概率。
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1) # 计算每个时间步的熵，形状为 (batch_size, sequence_length - 1)
    targets = input_ids[:, 1:]
    token_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # 计算每个时间步的对数概率，形状为 (batch_size, sequence_length - 1)

    batch_size = input_ids.shape[0]
    max_response_len = max(response_lens, default=0)
    # new_zeros 创建一个新的张量，形状为 (batch_size, max_response_len)，并用 0 填充。
    # 这个新生成的张量会与调用它的张量（源张量）保持以下三个属性一致: 数据类型、设备和梯度跟踪状态。
    response_log_probs = token_log_probs.new_zeros((batch_size, max_response_len))
    response_entropy = entropy.new_zeros((batch_size, max_response_len))

    for row_index, (prompt_len, response_len) in enumerate(zip(prompt_lens, response_lens)):
        if response_len <= 0:
            continue
        start = max(prompt_len - 1, 0)
        end = start + response_len
        response_log_probs[row_index, :response_len] = token_log_probs[row_index, start:end]
        response_entropy[row_index, :response_len] = entropy[row_index, start:end]
    return response_log_probs, response_entropy


def tensor_to_list_rows(matrix, lengths: Sequence[int]) -> List[List[float]]:
    rows: List[List[float]] = []
    for row, length in zip(matrix.detach().cpu().tolist(), lengths):
        rows.append([float(value) for value in row[:length]])
    return rows


def count_valid_tokens(rows: Sequence[Sequence[int]]) -> int:
    return sum(sum(int(value) for value in row) for row in rows)


def aggregate_weight(batch, loss_agg_mode: str) -> float:
    """
    Function:
        根据指定的 loss_agg_mode 聚合权重。支持两种模式：
        - "token-mean": 返回批次中有效 token 的总数（即 response_mask 中值为 1 的数量）。这种模式适用于按 token 平均的损失计算。
        - 其他模式: 返回批次中的样本数量。这种模式适用于按样本平均的损失计算。
    """
    if loss_agg_mode == "token-mean":
        return float(count_valid_tokens(batch.batch["response_mask"]))
    return float(len(batch))


def mean_of_values(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def clone_model_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    torch, _, _, _ = require_hf_dependencies()
    cloned: Dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = value
    return cloned

