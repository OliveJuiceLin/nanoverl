"""Typed configuration objects for nanoverl.
目前的实现用户必须配置这里已有的字段，暂时不支持配置文件里没有的字段，如果想增加配置，必须在这里也增加
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar, get_args, get_origin, get_type_hints


class ConfigError(ValueError):
    """Raised when a config file or config tree is invalid."""


def _load_mapping(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised when PyYAML is installed
            raise ConfigError(
                "YAML config support requires PyYAML. Install nanoverl with the 'train' extra "
                "or pass a JSON config."
            ) from exc
        data = yaml.safe_load(text)
    else:
        raise ConfigError("Unsupported config format. Use .json, .yaml, or .yml.")
    if not isinstance(data, dict):
        raise ConfigError("Trainer config root must be a mapping.")
    return data


T = TypeVar("T")


def _coerce_dataclass(cls: Type[T], value: Any) -> T:
    if isinstance(value, cls):
        return value
    if value is None:
        return cls()  # type: ignore[misc]
    if not isinstance(value, Mapping):
        raise ConfigError("Expected a mapping while building config dataclass.")
    type_hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for field_name, field_info in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
        if field_name not in value:
            continue
        raw = value[field_name]
        field_type = type_hints.get(field_name, field_info.type)
        origin = get_origin(field_type)
        if hasattr(field_type, "__dataclass_fields__"):
            kwargs[field_name] = _coerce_dataclass(field_type, raw)
        elif origin in (tuple, Tuple):
            kwargs[field_name] = tuple(raw)
        elif origin is not None:
            args = [arg for arg in get_args(field_type) if arg is not type(None)]
            nested_type = args[0] if args else None
            if nested_type is not None and hasattr(nested_type, "__dataclass_fields__"):
                kwargs[field_name] = _coerce_dataclass(nested_type, raw)
            else:
                kwargs[field_name] = raw
        else:
            kwargs[field_name] = raw
    return cls(**kwargs)  # type: ignore[arg-type]


@dataclass
class DataConfig:
    train_path: str = "examples/data/debug_prompts.jsonl"
    val_path: Optional[str] = "examples/data/debug_prompts.jsonl"
    train_batch_size: int = 2
    val_batch_size: int = 2
    shuffle: bool = True
    seed: int = 7
    max_prompt_length: int = 1024
    max_response_length: int = 256


@dataclass
class ModelConfig:
    path: str = "debug-model"
    tokenizer_path: Optional[str] = None
    critic_path: Optional[str] = None
    trust_remote_code: bool = False
    use_remove_padding: bool = True
    dtype: str = "float32"
    chat_template_path: Optional[str] = None
    attn_implementation: str = "eager"


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    do_sample: bool = True
    n: int = 1


@dataclass
class AlgorithmConfig:
    name: str = "ppo"
    advantage_estimator: str = "gae"
    gamma: float = 1.0
    lam: float = 1.0
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_coef: float = 0.0
    norm_adv_by_std_in_grpo: bool = True


@dataclass
class ActorConfig:
    backend: str = "debug"
    device: Optional[str] = None
    ppo_mini_batch_size: int = 2
    ppo_epochs: int = 1
    micro_batch_size: Optional[int] = None
    clip_ratio: float = 0.2
    clip_ratio_low: Optional[float] = None
    clip_ratio_high: Optional[float] = None
    clip_ratio_c: float = 3.0
    entropy_coeff: float = 0.0
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    loss_agg_mode: str = "token-mean"
    update_step_size: float = 0.02
    lr: float = 1e-5
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95) # or (0.9, 0.999)?
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    shuffle: bool = True


@dataclass
class CriticConfig:
    backend: str = "debug"
    device: Optional[str] = None
    enable: bool = True
    ppo_mini_batch_size: int = 2
    ppo_epochs: int = 1
    micro_batch_size: Optional[int] = None
    cliprange_value: float = 0.5
    loss_agg_mode: str = "token-mean"
    lr: float = 1e-5
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    shuffle: bool = True


@dataclass
class ReferenceConfig:
    backend: str = "debug"
    device: Optional[str] = None
    enable: bool = True
    fixed_kl_offset: float = -0.15


@dataclass
class RolloutConfig:
    backend: str = "debug"
    device: Optional[str] = None
    response_length: int = 64
    train: SamplingConfig = field(default_factory=SamplingConfig)
    validation: SamplingConfig = field(
        default_factory=lambda: SamplingConfig(temperature=0.0, do_sample=False, n=1)
    )
    balance_by_length: bool = False
    gpu_memory_utilization: float = 0.5
    tensor_model_parallel_size: int = 1
    enforce_eager: bool = False
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardConfig:
    type: str = "python"
    function_path: Optional[str] = None
    function_name: str = "compute_reward"
    reward_key: str = "data_source"


@dataclass
class TrainerRuntimeConfig:
    total_epochs: int = 1
    total_training_steps: Optional[int] = None
    validate_before_train: bool = True
    validate_only: bool = False
    test_freq: int = 0
    save_freq: int = 0
    critic_warmup: int = 0
    balance_batch: bool = False
    train_dump_freq: int = 0
    validation_dump_freq: int = 0
    dump_max_rows: int = 8
    project_name: str = "nanoverl"
    experiment_name: str = "debug"
    default_local_dir: str = "checkpoints"
    loggers: Tuple[str, ...] = ("console",)


@dataclass
class RayConfig:
    enabled: bool = False
    address: Optional[str] = None
    num_cpus: Optional[int] = None


@dataclass
class TrainerConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    trainer: TrainerRuntimeConfig = field(default_factory=TrainerRuntimeConfig)
    ray: RayConfig = field(default_factory=RayConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainerConfig":
        cfg = cls(
            data=_coerce_dataclass(DataConfig, data.get("data")),
            model=_coerce_dataclass(ModelConfig, data.get("model")),
            algorithm=_coerce_dataclass(AlgorithmConfig, data.get("algorithm")),
            actor=_coerce_dataclass(ActorConfig, data.get("actor")),
            critic=_coerce_dataclass(CriticConfig, data.get("critic")),
            reference=_coerce_dataclass(ReferenceConfig, data.get("reference")),
            rollout=_coerce_dataclass(RolloutConfig, data.get("rollout")),
            reward=_coerce_dataclass(RewardConfig, data.get("reward")),
            trainer=_coerce_dataclass(TrainerRuntimeConfig, data.get("trainer")),
            ray=_coerce_dataclass(RayConfig, data.get("ray")),
        )
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "TrainerConfig":
        return cls.from_dict(_load_mapping(Path(path)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def total_training_steps(self, train_batches_per_epoch: int) -> int:
        if self.trainer.total_training_steps is not None:
            return self.trainer.total_training_steps
        return train_batches_per_epoch * self.trainer.total_epochs

    def validate(self) -> None:
        if self.data.train_batch_size <= 0:
            raise ConfigError("data.train_batch_size must be positive.")
        if self.data.val_path and self.data.val_batch_size <= 0:
            raise ConfigError("data.val_batch_size must be positive when validation is enabled.")
        if self.data.max_prompt_length <= 0:
            raise ConfigError("data.max_prompt_length must be positive.")
        if self.data.max_response_length <= 0:
            raise ConfigError("data.max_response_length must be positive.")
        if self.algorithm.name not in {"ppo", "grpo"}:
            raise ConfigError("algorithm.name must be either 'ppo' or 'grpo'.")
        if self.rollout.train.n <= 0:
            raise ConfigError("rollout.train.n must be positive.")
        if self.rollout.validation.n <= 0:
            raise ConfigError("rollout.validation.n must be positive.")
        if self.rollout.response_length <= 0:
            raise ConfigError("rollout.response_length must be positive.")
        if self.trainer.total_epochs <= 0 and self.trainer.total_training_steps is None:
            raise ConfigError("trainer.total_epochs must be positive when trainer.total_training_steps is not set.")
        if self.trainer.total_training_steps is not None and self.trainer.total_training_steps < 0:
            raise ConfigError("trainer.total_training_steps must be non-negative when set.")
        if self.trainer.test_freq < 0:
            raise ConfigError("trainer.test_freq must be non-negative.")
        if self.trainer.save_freq < 0:
            raise ConfigError("trainer.save_freq must be non-negative.")
        if self.trainer.critic_warmup < 0:
            raise ConfigError("trainer.critic_warmup must be non-negative.")
        if self.trainer.train_dump_freq < 0:
            raise ConfigError("trainer.train_dump_freq must be non-negative.")
        if self.trainer.validation_dump_freq < 0:
            raise ConfigError("trainer.validation_dump_freq must be non-negative.")
        if self.trainer.dump_max_rows <= 0:
            raise ConfigError("trainer.dump_max_rows must be positive.")
        if self.actor.ppo_mini_batch_size <= 0:
            raise ConfigError("actor.ppo_mini_batch_size must be positive.")
        if self.actor.micro_batch_size is not None and self.actor.micro_batch_size <= 0:
            raise ConfigError("actor.micro_batch_size must be positive when set.")
        uses_critic = self.critic.enable and self.algorithm.advantage_estimator != "grpo" and self.algorithm.name != "grpo"
        if self.actor.micro_batch_size is not None:
            if self.actor.micro_batch_size > self.actor.ppo_mini_batch_size:
                raise ConfigError("actor.micro_batch_size must not exceed actor.ppo_mini_batch_size.")
            if self.actor.ppo_mini_batch_size % self.actor.micro_batch_size != 0:
                raise ConfigError("actor.ppo_mini_batch_size must be divisible by actor.micro_batch_size.")
        if uses_critic and self.critic.ppo_mini_batch_size <= 0:
            raise ConfigError("critic.ppo_mini_batch_size must be positive when critic is enabled.")
        if uses_critic and self.critic.micro_batch_size is not None and self.critic.micro_batch_size <= 0:
            raise ConfigError("critic.micro_batch_size must be positive when set.")
        if uses_critic and self.critic.micro_batch_size is not None:
            if self.critic.micro_batch_size > self.critic.ppo_mini_batch_size:
                raise ConfigError("critic.micro_batch_size must not exceed critic.ppo_mini_batch_size.")
            if self.critic.ppo_mini_batch_size % self.critic.micro_batch_size != 0:
                raise ConfigError("critic.ppo_mini_batch_size must be divisible by critic.micro_batch_size.")
        if self.algorithm.name == "grpo" and self.algorithm.advantage_estimator == "gae":
            self.algorithm.advantage_estimator = "grpo"
        if self.algorithm.advantage_estimator == "grpo" and self.rollout.train.n < 2:
            raise ConfigError("GRPO requires rollout.train.n >= 2.")
        if self.algorithm.advantage_estimator == "grpo" and self.actor.ppo_mini_batch_size % self.rollout.train.n != 0:
            raise ConfigError("GRPO requires actor.ppo_mini_batch_size to be divisible by rollout.train.n.")
        if self.algorithm.advantage_estimator == "grpo" and self.actor.ppo_mini_batch_size < self.rollout.train.n:
            raise ConfigError("GRPO requires actor.ppo_mini_batch_size to cover at least one rollout group.")
        if self.algorithm.use_kl_in_reward and not self.reference.enable:
            raise ConfigError("Reference worker must be enabled when KL-in-reward is enabled.")
        if self.actor.use_kl_loss and not self.reference.enable:
            raise ConfigError("Reference worker must be enabled when actor KL loss is enabled.")
        if self.trainer.validate_only and not self.data.val_path:
            raise ConfigError("trainer.validate_only requires data.val_path.")
        if self.trainer.validate_only:
            self.trainer.validate_before_train = True
        if self.trainer.test_freq > 0 and not self.data.val_path:
            raise ConfigError("trainer.test_freq requires data.val_path.")
        if self.trainer.validation_dump_freq > 0 and not self.data.val_path:
            raise ConfigError("trainer.validation_dump_freq requires data.val_path.")
        if self.actor.backend == "hf" and self.rollout.backend not in {"hf", "vllm"}:
            raise ConfigError("actor.backend='hf' requires rollout.backend to be 'hf' or 'vllm'.")
        if self.actor.backend == "hf" and self.reference.enable and self.reference.backend != "hf":
            raise ConfigError("reference.backend must be 'hf' when actor.backend is 'hf'.")
        if self.actor.backend == "hf" and uses_critic and self.critic.backend != "hf":
            raise ConfigError("critic.backend must be 'hf' when actor.backend is 'hf' and critic is used.")
        if self.actor.backend == "fsdp" and self.rollout.backend not in {"hf", "vllm"}:
            raise ConfigError("actor.backend='fsdp' requires rollout.backend to be 'hf' or 'vllm'.")
        if self.actor.backend == "fsdp" and self.reference.enable and self.reference.backend != "fsdp":
            raise ConfigError("reference.backend must be 'fsdp' when actor.backend is 'fsdp'.")
        if self.actor.backend == "fsdp" and uses_critic and self.critic.backend != "fsdp":
            raise ConfigError("critic.backend must be 'fsdp' when actor.backend is 'fsdp' and critic is used.")
        if self.rollout.backend == "vllm":
            if self.actor.backend == "debug":
                raise ConfigError("rollout.backend='vllm' requires actor.backend to be 'hf' or 'fsdp'.")
            if self.rollout.tensor_model_parallel_size <= 0:
                raise ConfigError("rollout.tensor_model_parallel_size must be positive.")
            if self.rollout.gpu_memory_utilization <= 0.0 or self.rollout.gpu_memory_utilization > 1.0:
                raise ConfigError("rollout.gpu_memory_utilization must be in the range (0, 1].")
            # This guard is new in Phase 3 because the thin vLLM slice keeps one
            # local rollout engine per trainer process. Wider tensor parallel
            # rollout needs a larger weight-transfer/runtime design than we want here.
            if self.rollout.tensor_model_parallel_size != 1:
                raise ConfigError("rollout.tensor_model_parallel_size is currently supported only for value 1.")
        if self.rollout.backend in {"hf", "vllm"} and self.model.tokenizer_path is None:
            self.model.tokenizer_path = self.model.path
        if self.rollout.balance_by_length:
            self.trainer.balance_batch = True
        if self.trainer.balance_batch and uses_critic:
            if self.actor.ppo_mini_batch_size != self.critic.ppo_mini_batch_size:
                raise ConfigError(
                    "trainer.balance_batch currently requires actor and critic mini-batch sizes to match."
                )
        if self.ray.enabled and self.actor.backend == "debug":
            # This is intentionally permissive; debug workers still run locally.
            return


__all__ = [
    "ActorConfig",
    "AlgorithmConfig",
    "ConfigError",
    "CriticConfig",
    "DataConfig",
    "ModelConfig",
    "RayConfig",
    "ReferenceConfig",
    "RewardConfig",
    "RolloutConfig",
    "SamplingConfig",
    "TrainerConfig",
    "TrainerRuntimeConfig",
]
