"""FSDP backend entry points.

These classes are explicit placeholders so the package shape matches the intended architecture.
They fail loudly until the torch-backed implementation lands.
"""

from __future__ import annotations

from nanoverl.workers.base import PolicyWorker, ReferenceWorker, ValueWorker


class MissingDependencyError(RuntimeError):
    """Raised when an optional backend dependency is unavailable."""


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError(
            "The FSDP backend requires torch. Install nanoverl with the 'train' extra first."
        ) from exc


class FSDPPolicyWorker(PolicyWorker):
    def __init__(self, config):
        _require_torch()
        raise NotImplementedError("FSDP policy integration is scaffolded but not wired to a model engine yet.")


class FSDPReferenceWorker(ReferenceWorker):
    def __init__(self, config):
        _require_torch()
        raise NotImplementedError("FSDP reference integration is scaffolded but not wired to a model engine yet.")


class FSDPValueWorker(ValueWorker):
    def __init__(self, config):
        _require_torch()
        raise NotImplementedError("FSDP value integration is scaffolded but not wired to a model engine yet.")


__all__ = [
    "FSDPPolicyWorker",
    "FSDPReferenceWorker",
    "FSDPValueWorker",
    "MissingDependencyError",
]
