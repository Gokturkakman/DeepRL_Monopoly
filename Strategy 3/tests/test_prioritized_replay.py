from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monopoly_game_engine.prioritized_replay import PrioritizedReplayBuffer, SumTree  # noqa: E402
from monopoly_game_engine.state import STATE_DIM  # noqa: E402


def _push(buf: PrioritizedReplayBuffer, marker: int) -> None:
    state = np.full(STATE_DIM, marker, dtype=np.float32)
    buf.push(state, action=0, reward=float(marker), next_state=state, done=False, next_allowed=[0])


class SumTreeTests(unittest.TestCase):
    def test_total_matches_sum_of_updates(self) -> None:
        tree = SumTree(capacity=4)
        for p in (1.0, 2.0, 3.0, 4.0):
            tree.add(p)
        self.assertAlmostEqual(tree.total(), 10.0)

    def test_overwrite_at_capacity_replaces_oldest(self) -> None:
        tree = SumTree(capacity=2)
        tree.add(1.0)
        tree.add(2.0)
        tree.add(5.0)  # wraps, replaces the first 1.0
        self.assertAlmostEqual(tree.total(), 7.0)
        self.assertEqual(tree.n_entries, 2)

    def test_get_returns_matching_leaf_for_cumulative_value(self) -> None:
        tree = SumTree(capacity=3)
        tree.add(1.0)
        tree.add(1.0)
        tree.add(8.0)
        # third leaf owns the [2, 10) range
        _, priority, data_idx = tree.get(5.0)
        self.assertAlmostEqual(priority, 8.0)
        self.assertEqual(data_idx, 2)


class PrioritizedReplayBufferTests(unittest.TestCase):
    def test_len_and_capacity_wraparound(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=3, alpha=0.6)
        for i in range(5):
            _push(buf, i)
        self.assertEqual(len(buf), 3)  # capped at capacity, not total pushes

    def test_sample_returns_is_weights_bounded_by_one(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=8, alpha=0.6, beta_start=0.4)
        for i in range(8):
            _push(buf, i)
        random.seed(0)
        *_, leaf_indices, weights = buf.sample(4)
        self.assertEqual(len(leaf_indices), 4)
        self.assertTrue((weights <= 1.0 + 1e-6).all())
        self.assertTrue((weights > 0).all())

    def test_high_priority_transition_sampled_more_often(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=4, alpha=1.0, beta_start=1.0)
        for i in range(4):
            _push(buf, i)
        # index 0's leaf sits at data-index 0; give it a much higher priority
        target_leaf = buf.tree.capacity - 1
        buf.tree.update(target_leaf, 100.0)

        random.seed(0)
        counts = {0: 0, "other": 0}
        for _ in range(200):
            _, _, _, _, _, _, _, leaf_indices, _ = buf.sample(1)
            counts[0 if leaf_indices[0] == target_leaf else "other"] += 1

        self.assertGreater(counts[0], counts["other"])

    def test_update_priorities_raises_max_priority_for_future_inserts(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=4, alpha=0.6)
        for i in range(4):
            _push(buf, i)
        buf.update_priorities([buf.tree.capacity - 1], [50.0])
        self.assertGreater(buf.max_priority, 1.0)

    def test_mutate_last_reward_updates_only_the_latest_transition(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=4, alpha=0.6)
        _push(buf, 1)
        _push(buf, 2)
        buf.mutate_last_reward(10.0)
        self.assertAlmostEqual(buf.data[0][2], 1.0)
        self.assertAlmostEqual(buf.data[1][2], 12.0)

    def test_state_dict_round_trip_preserves_content_and_priorities(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=4, alpha=0.6, beta_start=0.4, beta_frames=10)
        for i in range(6):  # forces one wraparound
            _push(buf, i)
        buf.update_priorities([buf.tree.capacity - 1], [7.0])
        payload = buf.state_dict()

        restored = PrioritizedReplayBuffer(capacity=1)  # constructor args overwritten by load
        restored.load_state_dict(payload)

        self.assertEqual(len(restored), len(buf))
        self.assertEqual(restored.capacity, buf.capacity)
        original_order = buf._temporal_order()
        restored_order = restored._temporal_order()
        original_rewards = sorted(buf.data[i][2] for i in original_order)
        restored_rewards = sorted(restored.data[i][2] for i in restored_order)
        self.assertEqual(original_rewards, restored_rewards)


if __name__ == "__main__":
    unittest.main()
