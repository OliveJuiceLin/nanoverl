"""RL algorithm building blocks."""

from nanoverl.algos.advantages import (
    compute_gae_advantages,
    compute_grpo_advantages,
    compute_rloo_advantages,
    get_advantage_estimator,
    register_advantage_estimator,
)
from nanoverl.algos.base import AlgorithmStepContext, RLAlgorithm
from nanoverl.algos.kl import apply_kl_penalty, compute_kl_penalty
from nanoverl.algos.ppo import (
    compute_policy_loss,
    compute_reinforce_policy_loss,
    compute_value_loss,
    get_policy_loss_fn,
    register_policy_loss,
)
from nanoverl.algos.registry import create_algorithm, get_algorithm_class, register_algorithm

import nanoverl.algos.on_policy as _on_policy  # noqa: F401

__all__ = [
    "AlgorithmStepContext",
    "RLAlgorithm",
    "apply_kl_penalty",
    "compute_gae_advantages",
    "compute_grpo_advantages",
    "compute_kl_penalty",
    "compute_policy_loss",
    "compute_reinforce_policy_loss",
    "compute_rloo_advantages",
    "compute_value_loss",
    "create_algorithm",
    "get_advantage_estimator",
    "get_algorithm_class",
    "get_policy_loss_fn",
    "register_advantage_estimator",
    "register_algorithm",
    "register_policy_loss",
]
