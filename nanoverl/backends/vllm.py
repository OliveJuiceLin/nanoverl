"""Shared helpers for the optional local vLLM rollout backend."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from nanoverl.rollout.base import SamplingParams


class MissingDependencyError(RuntimeError):
    """Raised when an optional local vLLM backend dependency is unavailable."""


def require_vllm_dependencies():
    # This helper is new in Phase 3 because the trainer now supports a second
    # inference backend, but we still want `nanoverl` imports to stay cheap when
    # vLLM is not installed.
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised when torch is installed
        raise MissingDependencyError(
            "The vLLM rollout backend requires torch. Install nanoverl with the training dependencies first."
        ) from exc
    try:
        from vllm import LLM, SamplingParams as VLLMSamplingParams
    except ImportError as exc:  # pragma: no cover - exercised when vllm is installed
        raise MissingDependencyError(
            "The vLLM rollout backend requires vllm. Activate the vllm environment or install vllm first."
        ) from exc
    return torch, LLM, VLLMSamplingParams


def build_vllm_sampling_params(sampling: SamplingParams, max_response_length: int):
    # This adapter is new in Phase 3 because the trainer already has one
    # backend-neutral sampling config, and vLLM should plug into that contract
    # instead of introducing its own rollout-specific settings surface.
    _, _, vllm_sampling_params = require_vllm_dependencies()
    top_k = sampling.top_k if sampling.top_k and sampling.top_k > 0 else -1
    return vllm_sampling_params(
        n=1,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=top_k,
        max_tokens=max_response_length,
        skip_special_tokens=True,
    )


def build_vllm_ipc_weight_update_request(
    model_state: Dict[str, Any],
    device,
) -> Tuple[Dict[str, Dict[str, object]], List[object]]:
    # This helper is new in Phase 3 because HF rollout could load a PyTorch
    # state dict directly, while vLLM exposes policy refresh through
    # `LLM.update_weights(...)` and expects one IPC-formatted update payload.
    torch, _, _ = require_vllm_dependencies()
    if device.type != "cuda":
        raise RuntimeError("The thin vLLM rollout backend currently requires CUDA for policy sync.")

    from torch.multiprocessing.reductions import reduce_tensor

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    gpu_uuid = str(torch.cuda.get_device_properties(device_index).uuid)

    tensor_names: List[str] = []
    tensor_dtype_names: List[str] = []
    tensor_shapes: List[List[int]] = []
    resident_tensors: List[object] = []
    ipc_handles: List[Dict[str, object]] = []

    for tensor_name, tensor_value in model_state.items():
        if not isinstance(tensor_value, torch.Tensor):
            continue
        tensor_on_device = tensor_value.to(device=device, non_blocking=True).contiguous()
        resident_tensors.append(tensor_on_device)
        tensor_names.append(tensor_name)
        tensor_dtype_names.append(str(tensor_on_device.dtype).split(".")[-1])
        tensor_shapes.append(list(tensor_on_device.shape))
        ipc_handles.append({gpu_uuid: reduce_tensor(tensor_on_device)})

    return {
        "update_info": {
            "names": tensor_names,
            "dtype_names": tensor_dtype_names,
            "shapes": tensor_shapes,
            "ipc_handles": ipc_handles,
        }
    }, resident_tensors


__all__ = [
    "MissingDependencyError",
    "build_vllm_ipc_weight_update_request",
    "build_vllm_sampling_params",
    "require_vllm_dependencies",
]
