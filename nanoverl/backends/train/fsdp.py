"""
Single-node FSDP worker implementations built on the local HF worker logic.
这个包里面的每一个类都继承自原始的 HF worker 类，并且混入了 FSDPWorkerMixin 来添加 FSDP 支持。每个类都重写了 state_dict 和 load_state_dict 方法，以正确处理 FSDP 模型的状态保存和加载。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Optional

from nanoverl.backends.hf import clone_model_state, get_default_device, require_hf_dependencies
from nanoverl.distributed import TorchDistributedRuntime
from nanoverl.workers.base import PolicyWorker, ReferenceWorker, ValueWorker
from nanoverl.workers.hf import HFPolicyWorker, HFReferenceWorker, HFValueWorker


class MissingDependencyError(RuntimeError):
    """Raised when an optional backend dependency is unavailable."""


def _require_fsdp_dependencies():
    torch, _, _, _ = require_hf_dependencies()
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel,
            FullStateDictConfig,
            ShardingStrategy,
            StateDictType,
        )
    except ImportError as exc:  # pragma: no cover - exercised only when FSDP is unavailable
        raise MissingDependencyError(
            "The FSDP backend requires torch.distributed.fsdp support. Install nanoverl with the 'train' extra first."
        ) from exc
    return torch, FullyShardedDataParallel, StateDictType, FullStateDictConfig, ShardingStrategy


def _fsdp_state_dict_context(module):
    """
    Function:
        - 为 PyTorch FSDP（完全分片数据并行）模型提供一个统一的 state_dict 上下文管理器
    Logic:
        1. 判断传入的 module 是否是一个分片后的模型（即 fully_sharded_data_parallel 实例）。
        2. 如果是分片模型，使用 fully_sharded_data_parallel.state_dict_type 创建一个上下文管理器
    """
    _, fully_sharded_data_parallel, state_dict_type, full_state_dict_config, _ = _require_fsdp_dependencies()
    if isinstance(module, fully_sharded_data_parallel):
        return fully_sharded_data_parallel.state_dict_type(
            module,
            state_dict_type.FULL_STATE_DICT, # 指示在保存时将分布在各个 GPU 上的参数合并为完整的权重
            full_state_dict_config(offload_to_cpu=True, rank0_only=False), # 将合并后的权重卸载到 CPU 内存，防止在大模型合并时因 GPU 显存不足（OOM）而崩溃。
        )
    return nullcontext()


class FSDPWorkerMixin:
    """
    是一个混入类（Mixin），它的主要作用是将单机单卡的 Hugging Face Worker（Phase 1、2 中的本地 Worker）
    无缝升级为支持多卡分布式训练的 FSDP Worker，而无需将底层的 RL 逻辑重写一遍。
    """
    def __init__(self, model_config):
        self.model_config = model_config
        self.runtime = TorchDistributedRuntime.from_environment()
        self.device = self._resolve_device() # 根据 local_rank 绑定显卡
        self._maybe_initialize_process_group() # 初始化分布式环境（如果启用的话）

    def _resolve_device(self):
        # This helper is new in Phase 3 because the local HF path only needed one
        # default device. The FSDP path needs rank-aware device placement so each
        # torchrun process lands on its own GPU without changing the trainer loop.
        torch, _, _, _ = require_hf_dependencies()
        if torch.cuda.is_available():
            return torch.device("cuda", self.runtime.local_rank)
        return get_default_device()

    def _maybe_initialize_process_group(self) -> None:
        # This helper is new in Phase 3 because distributed setup should stay in
        # one explicit place instead of leaking init logic into every worker class.
        if not self.runtime.enabled:
            return
        torch, _, _, _ = require_hf_dependencies()
        import torch.distributed as dist

        if dist.is_initialized(): # 防止重复初始化
            return
        backend = self.runtime.backend or ("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.runtime.local_rank) # 这确保了当前进程只“看到”并控制分配给它的那块特定显卡（由 local_rank 决定），避免多进程竞争同一显卡。
        # 这一步会建立各个进程之间的网络连接，是分布式计算的起点。
        dist.init_process_group(backend=backend, rank=self.runtime.rank, world_size=self.runtime.world_size)

    def _wrap_train_module(self, module):
        """
        Function:
            - 用于训练模型（如 Policy Actor、Value Critic）。
            - 将普通的 PyTorch 模型用 FullyShardedDataParallel 包装，并使用 FULL_SHARD 策略。
        Return:
            - 包装后的模型，适合训练使用。
        """
        torch, fully_sharded_data_parallel, _, _, sharding_strategy = _require_fsdp_dependencies()
        module = module.to(self.device)
        if not self.runtime.enabled or self.device.type != "cuda":
            return module
        return fully_sharded_data_parallel(
            module,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            sharding_strategy=sharding_strategy.FULL_SHARD, # 相当于 ZeRO-3，把参数、梯度、优化器状态全部切片打散到各个 GPU 上，最大化节省显存。
        )

    def _wrap_eval_module(self, module):
        """
        Function:
            - 用于评估模型（如 Reference Actor）。
            - 将普通的 PyTorch 模型用 FullyShardedDataParallel 包装，并使用 SHARD_GRAD_OP 策略。
        Return:
            - 包装后的模型，适合评估使用。
        """
        torch, fully_sharded_data_parallel, _, _, sharding_strategy = _require_fsdp_dependencies()
        module = module.to(self.device)
        if not self.runtime.enabled or self.device.type != "cuda":
            return module
        return fully_sharded_data_parallel(
            module,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            sharding_strategy=sharding_strategy.SHARD_GRAD_OP, # 相当于 ZeRO-2，只切片打散参数和梯度，优化器状态不切片，保留在每个 GPU 上。评估时不需要优化器状态，所以这样做可以减少通信开销，同时仍然节省显存。
        )

    def _module_state_dict(self, module) -> Dict[str, Any]:
        """
        Function:
            - 保存 Checkpoint 时，FSDP 默认只能拿到本卡的切片权重。
            - 这个方法结合外部的 _fsdp_state_dict_context 上下文，将各个 GPU 上的参数分片实时收集拼装成一个完整的 Hugging Face 权重，
            - 再进行 Clone，从而保证保存出的模型依然是可以通过常规手段加载的单机格式。
        """
        with _fsdp_state_dict_context(module):
            return clone_model_state(module.state_dict())

    def _load_module_state_dict(self, module, state: Optional[Dict[str, Any]]) -> None:
        """
        加载时同样通过上下文，将传入的统一完整权重自动切片并分发到各个 GPU 的分片模型上。
        """
        if state is None:
            return
        with _fsdp_state_dict_context(module):
            module.load_state_dict(state)


class FSDPReferenceWorker(FSDPWorkerMixin, HFReferenceWorker):
    def __init__(self, model_config, config):
        FSDPWorkerMixin.__init__(self, model_config)
        self.config = config
        from nanoverl.backends.hf import load_causal_lm

        model = load_causal_lm(model_config)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = self._wrap_eval_module(model)
        self.model.eval()

    def state_dict(self) -> Dict[str, Any]:
        return {"model_state": self._module_state_dict(self.model)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._load_module_state_dict(self.model, state.get("model_state"))
        self.model.to(self.device)
        self.model.eval()

# 等价：class FSDPPolicyWorker(FSDPWorkerMixin, HFPolicyWorker, PolicyWorker):
class FSDPPolicyWorker(FSDPWorkerMixin, HFPolicyWorker):
    def __init__(self, model_config, config):
        FSDPWorkerMixin.__init__(self, model_config)
        torch, _, _, _ = require_hf_dependencies()
        self.actor_config = config
        from nanoverl.backends.hf import load_causal_lm

        self.model = self._wrap_train_module(load_causal_lm(model_config))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            betas=tuple(config.betas),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        self.update_steps = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model_state": self._module_state_dict(self.model),
            "optimizer_state": self.optimizer.state_dict(),
            "update_steps": self.update_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._load_module_state_dict(self.model, state.get("model_state"))
        if state.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(state["optimizer_state"])
        self.update_steps = int(state.get("update_steps", 0))
        self.model.to(self.device)
        self.model.eval()


class FSDPValueWorker(FSDPWorkerMixin, HFValueWorker):
    def __init__(self, model_config, config):
        FSDPWorkerMixin.__init__(self, model_config)
        torch, _, _, _ = require_hf_dependencies()
        from nanoverl.backends.hf import load_backbone_model
        from nanoverl.workers.hf import _infer_hidden_size

        self.value_config = config
        backbone = load_backbone_model(model_config, path=model_config.critic_path)
        value_head = torch.nn.Linear(_infer_hidden_size(backbone.config), 1)
        self.backbone = self._wrap_train_module(backbone)
        self.value_head = self._wrap_train_module(value_head)
        self.optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.value_head.parameters()),
            lr=config.lr,
            betas=tuple(config.betas),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        self.update_steps = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "backbone_state": self._module_state_dict(self.backbone),
            "value_head_state": self._module_state_dict(self.value_head),
            "optimizer_state": self.optimizer.state_dict(),
            "update_steps": self.update_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._load_module_state_dict(self.backbone, state.get("backbone_state"))
        self._load_module_state_dict(self.value_head, state.get("value_head_state"))
        if state.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(state["optimizer_state"])
        self.update_steps = int(state.get("update_steps", 0))
        self.backbone.to(self.device)
        self.value_head.to(self.device)
        self.backbone.eval()
        self.value_head.eval()


__all__ = [
    "FSDPPolicyWorker",
    "FSDPReferenceWorker",
    "FSDPValueWorker",
    "MissingDependencyError",
]
