"""Backend registry for policy, reference, and value workers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from nanoverl.workers.base import PolicyWorker, ReferenceWorker, ValueWorker


PolicyWorkerFactory = Callable[[Any, Any], PolicyWorker]
ReferenceWorkerFactory = Callable[[Any, Any], ReferenceWorker]
ValueWorkerFactory = Callable[[Any, Any], ValueWorker]

_POLICY_WORKERS: Dict[str, PolicyWorkerFactory] = {}
_REFERENCE_WORKERS: Dict[str, ReferenceWorkerFactory] = {}
_VALUE_WORKERS: Dict[str, ValueWorkerFactory] = {}


def _normalize_backend(name: str) -> str:
    return str(name).lower()


def _unknown_backend_message(role: str, backend: str, registry: Dict[str, object]) -> str:
    registered = ", ".join(sorted(registry)) or "<none>"
    return "Unknown %s backend: %s. Registered %s backends: %s" % (role, backend, role, registered)


def register_policy_worker(backend: str, factory: PolicyWorkerFactory) -> None:
    backend = _normalize_backend(backend)
    if backend in _POLICY_WORKERS:
        raise ValueError("Policy backend is already registered: %s" % backend)
    _POLICY_WORKERS[backend] = factory


def register_reference_worker(backend: str, factory: ReferenceWorkerFactory) -> None:
    backend = _normalize_backend(backend)
    if backend in _REFERENCE_WORKERS:
        raise ValueError("Reference backend is already registered: %s" % backend)
    _REFERENCE_WORKERS[backend] = factory


def register_value_worker(backend: str, factory: ValueWorkerFactory) -> None:
    backend = _normalize_backend(backend)
    if backend in _VALUE_WORKERS:
        raise ValueError("Value backend is already registered: %s" % backend)
    _VALUE_WORKERS[backend] = factory


def get_policy_worker(backend: str) -> PolicyWorkerFactory:
    backend = _normalize_backend(backend)
    try:
        return _POLICY_WORKERS[backend]
    except KeyError as exc:
        raise ValueError(_unknown_backend_message("policy", backend, _POLICY_WORKERS)) from exc


def get_reference_worker(backend: str) -> ReferenceWorkerFactory:
    backend = _normalize_backend(backend)
    try:
        return _REFERENCE_WORKERS[backend]
    except KeyError as exc:
        raise ValueError(_unknown_backend_message("reference", backend, _REFERENCE_WORKERS)) from exc


def get_value_worker(backend: str) -> ValueWorkerFactory:
    backend = _normalize_backend(backend)
    try:
        return _VALUE_WORKERS[backend]
    except KeyError as exc:
        raise ValueError(_unknown_backend_message("value", backend, _VALUE_WORKERS)) from exc


def registered_worker_backends() -> Dict[str, Tuple[str, ...]]:
    return {
        "policy": tuple(sorted(_POLICY_WORKERS)),
        "reference": tuple(sorted(_REFERENCE_WORKERS)),
        "value": tuple(sorted(_VALUE_WORKERS)),
    }


def create_policy_worker(backend: str, model_config, actor_config) -> PolicyWorker:
    return get_policy_worker(backend)(model_config, actor_config)


def create_reference_worker(backend: str, model_config, ref_config) -> ReferenceWorker:
    return get_reference_worker(backend)(model_config, ref_config)


def create_value_worker(backend: str, model_config, critic_config) -> ValueWorker:
    return get_value_worker(backend)(model_config, critic_config)


def _create_debug_policy_worker(model_config, actor_config) -> PolicyWorker:
    del model_config
    from nanoverl.workers.debug import DebugPolicyWorker

    return DebugPolicyWorker(actor_config)


def _create_debug_reference_worker(model_config, ref_config) -> ReferenceWorker:
    del model_config
    from nanoverl.workers.debug import DebugReferenceWorker

    return DebugReferenceWorker(ref_config)


def _create_debug_value_worker(model_config, critic_config) -> ValueWorker:
    del model_config
    from nanoverl.workers.debug import DebugValueWorker

    return DebugValueWorker(critic_config)


def _create_hf_policy_worker(model_config, actor_config) -> PolicyWorker:
    from nanoverl.workers.hf import HFPolicyWorker

    return HFPolicyWorker(model_config, actor_config)


def _create_hf_reference_worker(model_config, ref_config) -> ReferenceWorker:
    from nanoverl.workers.hf import HFReferenceWorker

    return HFReferenceWorker(model_config, ref_config)


def _create_hf_value_worker(model_config, critic_config) -> ValueWorker:
    from nanoverl.workers.hf import HFValueWorker

    return HFValueWorker(model_config, critic_config)


def _create_fsdp_policy_worker(model_config, actor_config) -> PolicyWorker:
    from nanoverl.backends.train.fsdp import FSDPPolicyWorker

    return FSDPPolicyWorker(model_config, actor_config)


def _create_fsdp_reference_worker(model_config, ref_config) -> ReferenceWorker:
    from nanoverl.backends.train.fsdp import FSDPReferenceWorker

    return FSDPReferenceWorker(model_config, ref_config)


def _create_fsdp_value_worker(model_config, critic_config) -> ValueWorker:
    from nanoverl.backends.train.fsdp import FSDPValueWorker

    return FSDPValueWorker(model_config, critic_config)


register_policy_worker("debug", _create_debug_policy_worker)
register_reference_worker("debug", _create_debug_reference_worker)
register_value_worker("debug", _create_debug_value_worker)
register_policy_worker("hf", _create_hf_policy_worker)
register_reference_worker("hf", _create_hf_reference_worker)
register_value_worker("hf", _create_hf_value_worker)
register_policy_worker("fsdp", _create_fsdp_policy_worker)
register_reference_worker("fsdp", _create_fsdp_reference_worker)
register_value_worker("fsdp", _create_fsdp_value_worker)


__all__ = [
    "PolicyWorkerFactory",
    "ReferenceWorkerFactory",
    "ValueWorkerFactory",
    "create_policy_worker",
    "create_reference_worker",
    "create_value_worker",
    "get_policy_worker",
    "get_reference_worker",
    "get_value_worker",
    "register_policy_worker",
    "register_reference_worker",
    "register_value_worker",
    "registered_worker_backends",
]
