"""Backend registry for rollout engines."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from nanoverl.rollout.base import RolloutEngine


RolloutEngineFactory = Callable[[Any, Any, Any], RolloutEngine]
_ROLLOUT_ENGINES: Dict[str, RolloutEngineFactory] = {}


def _normalize_backend(name: str) -> str:
    return str(name).lower()


def register_rollout_engine(backend: str, factory: RolloutEngineFactory) -> None:
    backend = _normalize_backend(backend)
    if backend in _ROLLOUT_ENGINES:
        raise ValueError("Rollout backend is already registered: %s" % backend)
    _ROLLOUT_ENGINES[backend] = factory


def get_rollout_engine(backend: str) -> RolloutEngineFactory:
    backend = _normalize_backend(backend)
    try:
        return _ROLLOUT_ENGINES[backend]
    except KeyError as exc:
        registered = ", ".join(sorted(_ROLLOUT_ENGINES)) or "<none>"
        raise ValueError(
            "Unknown rollout backend: %s. Registered rollout backends: %s" % (backend, registered)
        ) from exc


def registered_rollout_backends() -> Tuple[str, ...]:
    return tuple(sorted(_ROLLOUT_ENGINES))


def create_rollout_engine(backend: str, model_config, data_config, rollout_config) -> RolloutEngine:
    return get_rollout_engine(backend)(model_config, data_config, rollout_config)


def _create_debug_rollout_engine(model_config, data_config, rollout_config) -> RolloutEngine:
    del model_config, data_config
    from nanoverl.rollout.debug import DebugRolloutEngine

    return DebugRolloutEngine(max_response_length=rollout_config.response_length)


def _create_hf_rollout_engine(model_config, data_config, rollout_config) -> RolloutEngine:
    from nanoverl.rollout.hf import HFRolloutEngine

    return HFRolloutEngine(model_config, data_config, rollout_config)


def _create_vllm_rollout_engine(model_config, data_config, rollout_config) -> RolloutEngine:
    from nanoverl.rollout.vllm import VLLMRolloutEngine

    return VLLMRolloutEngine(model_config, data_config, rollout_config)


register_rollout_engine("debug", _create_debug_rollout_engine)
register_rollout_engine("hf", _create_hf_rollout_engine)
register_rollout_engine("vllm", _create_vllm_rollout_engine)


__all__ = [
    "RolloutEngineFactory",
    "create_rollout_engine",
    "get_rollout_engine",
    "register_rollout_engine",
    "registered_rollout_backends",
]
