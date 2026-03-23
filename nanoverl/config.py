"""Typed configuration objects for nanoverl."""

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
    prompt_key: str = "prompt"
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
    trust_remote_code: bool = False
    use_remove_padding: bool = True


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
    ppo_mini_batch_size: int = 2
    ppo_epochs: int = 1
    clip_ratio: float = 0.2
    clip_ratio_low: Optional[float] = None
    clip_ratio_high: Optional[float] = None
    clip_ratio_c: float = 3.0
    entropy_coeff: float = 0.0
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    loss_agg_mode: str = "token-mean"
    update_step_size: float = 0.02


@dataclass
class CriticConfig:
    backend: str = "debug"
    enable: bool = True
    ppo_mini_batch_size: int = 2
    ppo_epochs: int = 1
    cliprange_value: float = 0.5
    loss_agg_mode: str = "token-mean"


@dataclass
class ReferenceConfig:
    backend: str = "debug"
    enable: bool = True
    fixed_kl_offset: float = -0.15


@dataclass
class RolloutConfig:
    backend: str = "debug"
    response_length: int = 64
    train: SamplingConfig = field(default_factory=SamplingConfig)
    validation: SamplingConfig = field(
        default_factory=lambda: SamplingConfig(temperature=0.0, do_sample=False, n=1)
    )
    balance_by_length: bool = False


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
        if self.rollout.train.n <= 0:
            raise ConfigError("rollout.train.n must be positive.")
        if self.actor.ppo_mini_batch_size <= 0:
            raise ConfigError("actor.ppo_mini_batch_size must be positive.")
        if self.critic.enable and self.critic.ppo_mini_batch_size <= 0:
            raise ConfigError("critic.ppo_mini_batch_size must be positive when critic is enabled.")
        if self.algorithm.name == "grpo" and self.algorithm.advantage_estimator == "gae":
            self.algorithm.advantage_estimator = "grpo"
        if self.algorithm.advantage_estimator == "grpo" and self.rollout.train.n < 2:
            raise ConfigError("GRPO requires rollout.train.n >= 2.")
        if self.algorithm.use_kl_in_reward and not self.reference.enable:
            raise ConfigError("Reference worker must be enabled when KL-in-reward is enabled.")
        if self.actor.use_kl_loss and not self.reference.enable:
            raise ConfigError("Reference worker must be enabled when actor KL loss is enabled.")
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
