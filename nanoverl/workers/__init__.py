"""Worker interfaces and debug implementations."""

from nanoverl.backends.train.fsdp import FSDPPolicyWorker, FSDPReferenceWorker, FSDPValueWorker
from nanoverl.workers.base import LogProbResult, PolicyWorker, ReferenceWorker, UpdateResult, ValueResult, ValueWorker
from nanoverl.workers.debug import DebugPolicyWorker, DebugReferenceWorker, DebugValueWorker
from nanoverl.workers.hf import HFPolicyWorker, HFReferenceWorker, HFValueWorker


def create_policy_worker(backend: str, model_config, config) -> PolicyWorker:
    if backend == "debug":
        return DebugPolicyWorker(config)
    if backend == "hf":
        return HFPolicyWorker(model_config, config)
    if backend == "fsdp":
        return FSDPPolicyWorker(model_config, config)
    raise ValueError("Unknown policy backend: %s" % backend)


def create_reference_worker(backend: str, model_config, config) -> ReferenceWorker:
    if backend == "debug":
        return DebugReferenceWorker(config)
    if backend == "hf":
        return HFReferenceWorker(model_config, config)
    if backend == "fsdp":
        return FSDPReferenceWorker(model_config, config)
    raise ValueError("Unknown reference backend: %s" % backend)


def create_value_worker(backend: str, model_config, config) -> ValueWorker:
    if backend == "debug":
        return DebugValueWorker(config)
    if backend == "hf":
        return HFValueWorker(model_config, config)
    if backend == "fsdp":
        return FSDPValueWorker(model_config, config)
    raise ValueError("Unknown value backend: %s" % backend)


__all__ = [
    "DebugPolicyWorker",
    "DebugReferenceWorker",
    "DebugValueWorker",
    "HFPolicyWorker",
    "HFReferenceWorker",
    "HFValueWorker",
    "LogProbResult",
    "PolicyWorker",
    "ReferenceWorker",
    "UpdateResult",
    "ValueResult",
    "ValueWorker",
    "create_policy_worker",
    "create_reference_worker",
    "create_value_worker",
]
