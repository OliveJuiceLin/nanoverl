"""Worker interfaces and debug implementations."""

from nanoverl.backends.train.fsdp import FSDPPolicyWorker, FSDPReferenceWorker, FSDPValueWorker
from nanoverl.workers.base import LogProbResult, PolicyWorker, ReferenceWorker, UpdateResult, ValueResult, ValueWorker
from nanoverl.workers.debug import DebugPolicyWorker, DebugReferenceWorker, DebugValueWorker
from nanoverl.workers.hf import HFPolicyWorker, HFReferenceWorker, HFValueWorker


def create_policy_worker(backend: str, model_config, actor_config) -> PolicyWorker:
    if backend == "debug":
        return DebugPolicyWorker(actor_config)
    if backend == "hf":
        return HFPolicyWorker(model_config, actor_config)
    if backend == "fsdp":
        return FSDPPolicyWorker(model_config, actor_config)
    raise ValueError("Unknown policy backend: %s" % backend)


def create_reference_worker(backend: str, model_config, ref_config) -> ReferenceWorker:
    if backend == "debug":
        return DebugReferenceWorker(ref_config)
    if backend == "hf":
        return HFReferenceWorker(model_config, ref_config)
    if backend == "fsdp":
        return FSDPReferenceWorker(model_config, ref_config)
    raise ValueError("Unknown reference backend: %s" % backend)


def create_value_worker(backend: str, model_config, critic_config) -> ValueWorker:
    if backend == "debug":
        return DebugValueWorker(critic_config)
    if backend == "hf":
        return HFValueWorker(model_config, critic_config)
    if backend == "fsdp":
        return FSDPValueWorker(model_config, critic_config)
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
