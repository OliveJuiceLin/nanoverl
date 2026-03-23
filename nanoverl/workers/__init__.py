"""Worker interfaces and debug implementations."""

from nanoverl.backends.train.fsdp import FSDPPolicyWorker, FSDPReferenceWorker, FSDPValueWorker
from nanoverl.workers.base import LogProbResult, PolicyWorker, ReferenceWorker, UpdateResult, ValueResult, ValueWorker
from nanoverl.workers.debug import DebugPolicyWorker, DebugReferenceWorker, DebugValueWorker


def create_policy_worker(backend: str, config) -> PolicyWorker:
    if backend == "debug":
        return DebugPolicyWorker(config)
    if backend == "fsdp":
        return FSDPPolicyWorker(config)
    raise ValueError("Unknown policy backend: %s" % backend)


def create_reference_worker(backend: str, config) -> ReferenceWorker:
    if backend == "debug":
        return DebugReferenceWorker(config)
    if backend == "fsdp":
        return FSDPReferenceWorker(config)
    raise ValueError("Unknown reference backend: %s" % backend)


def create_value_worker(backend: str, config) -> ValueWorker:
    if backend == "debug":
        return DebugValueWorker(config)
    if backend == "fsdp":
        return FSDPValueWorker(config)
    raise ValueError("Unknown value backend: %s" % backend)


__all__ = [
    "DebugPolicyWorker",
    "DebugReferenceWorker",
    "DebugValueWorker",
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
