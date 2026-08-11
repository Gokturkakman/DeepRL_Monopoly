from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monopoly_game_engine.agent_ddqn import DDQNAgent  # noqa: E402
from monopoly_game_engine.agent_ppo import PPOAgent  # noqa: E402
from monopoly_game_engine.env import MonopolyEnv  # noqa: E402
from monopoly_game_engine.self_play import SelfPlayPool  # noqa: E402


class SelfPlayPoolTests(unittest.TestCase):
    def _env(self) -> MonopolyEnv:
        env = MonopolyEnv(agent_ids=[0], max_rounds=2)
        env.turn_order = [0, 1, 2, 3]
        env.current_turn_idx = 1
        return env

    def test_ddqn_snapshot_acts_legally(self) -> None:
        agent = DDQNAgent(player_id=1, hybrid=True, device="cpu")
        pool = SelfPlayPool(max_size=4)
        pool.register(agent)

        opponent = pool.sample_opponent(1, rng=random.Random(0))
        env = self._env()
        action = opponent.choose_action(env)
        self.assertIn(action, env.get_allowed_actions(1))

    def test_ppo_snapshot_acts_legally(self) -> None:
        agent = PPOAgent(player_id=1, hybrid=True, device="cpu")
        pool = SelfPlayPool(max_size=4)
        pool.register(agent)

        opponent = pool.sample_opponent(1, rng=random.Random(0))
        env = self._env()
        action = opponent.choose_action(env)
        self.assertIn(action, env.get_allowed_actions(1))

    def test_snapshot_is_frozen_not_aliased_to_live_agent(self) -> None:
        agent = PPOAgent(player_id=1, hybrid=False, device="cpu")
        pool = SelfPlayPool(max_size=4)
        pool.register(agent)

        before = {k: v.clone() for k, v in pool._snapshots[0]["state_dict"].items()}
        with torch.no_grad():
            for p in agent.actor.parameters():
                p.add_(1.0)

        after = pool._snapshots[0]["state_dict"]
        for k in before:
            self.assertTrue(torch.equal(before[k], after[k]))

    def test_rejects_agent_without_known_network(self) -> None:
        pool = SelfPlayPool(max_size=4)
        with self.assertRaises(TypeError):
            pool.register(object())


if __name__ == "__main__":
    unittest.main()
