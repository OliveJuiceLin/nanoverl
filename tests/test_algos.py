"""Tests for PPO and GRPO math."""

import unittest

from nanoverl.algos.advantages import compute_gae_advantages, compute_grpo_advantages
from nanoverl.algos.ppo import compute_policy_loss, compute_value_loss


class AlgoTest(unittest.TestCase):
    def test_gae_advantages(self):
        advantages, returns = compute_gae_advantages(
            token_level_rewards=[[1.0, 0.0]],
            values=[[0.0, 0.0]],
            response_mask=[[1, 1]],
            gamma=1.0,
            lam=1.0,
        )
        self.assertEqual(advantages, [[1.0, 0.0]])
        self.assertEqual(returns, [[1.0, 0.0]])

    def test_grpo_advantages(self):
        advantages, _ = compute_grpo_advantages(
            token_level_rewards=[[1.0], [0.0]],
            response_mask=[[1], [1]],
            group_ids=["g", "g"],
            normalize_by_std=True,
        )
        self.assertAlmostEqual(advantages[0][0], 1.0, places=6)
        self.assertAlmostEqual(advantages[1][0], -1.0, places=6)

    def test_policy_loss(self):
        loss, metrics = compute_policy_loss(
            old_log_probs=[[0.0]],
            log_probs=[[0.0]],
            advantages=[[1.0]],
            response_mask=[[1]],
            cliprange=0.2,
        )
        self.assertAlmostEqual(loss, -1.0, places=6)
        self.assertAlmostEqual(metrics["policy_approx_kl"], 0.0, places=6)

    def test_value_loss(self):
        loss, metrics = compute_value_loss(
            values=[[0.5]],
            returns=[[1.0]],
            response_mask=[[1]],
            cliprange_value=0.5,
        )
        self.assertAlmostEqual(loss, 0.25, places=6)
        self.assertAlmostEqual(metrics["value_abs_error"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
