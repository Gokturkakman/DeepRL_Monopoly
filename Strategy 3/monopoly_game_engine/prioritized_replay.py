"""
Proportional prioritized experience replay (Schaul et al. 2016,
arXiv:1511.05952) for DDQNAgent.

Why: with a 2,958-action space, most transitions carry little learning
signal (the TD-error is already small -- the agent had it roughly right),
while a minority (e.g. the transition right before a bankruptcy or a large
trade) carry a lot. Uniform sampling spends the same number of gradient
steps on both; sampling proportional to |TD-error| spends more steps where
the value estimate is most wrong. Importance-sampling weights correct the
resulting sampling bias so the Bellman update stays an unbiased estimator
in expectation (annealed toward full correction via ``beta``, per the
paper: early training tolerates the bias in exchange for prioritizing hard
transitions when the value estimates are least reliable anyway).
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

import numpy as np
import torch

from .state import STATE_DIM


class SumTree:
    """
    Binary tree stored as a flat array: internal nodes hold the sum of
    their children, leaves hold individual priorities. ``get(s)`` walks
    down from the root in O(log capacity), giving O(log n) prioritized
    sampling instead of the O(n) rebuild-a-CDF approach.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write = 0
        self.n_entries = 0

    def update(self, leaf_idx: int, priority: float) -> None:
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        idx = leaf_idx
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def add(self, priority: float) -> int:
        """Insert at the current write head; returns the leaf index used."""
        leaf_idx = self.write + self.capacity - 1
        self.update(leaf_idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return leaf_idx

    def total(self) -> float:
        return float(self.tree[0])

    def leaf_priority(self, data_idx: int) -> float:
        return float(self.tree[data_idx + self.capacity - 1])

    def get(self, cumulative: float) -> tuple[int, float, int]:
        """Find the leaf whose cumulative-priority range contains ``cumulative``."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if cumulative <= self.tree[left]:
                idx = left
            else:
                cumulative -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), data_idx


class PrioritizedReplayBuffer:
    """
    Drop-in replacement for DDQNAgent's uniform ``ReplayBuffer``: same
    ``push``/``__len__``/``state_dict``/``load_state_dict`` surface, plus
    ``sample`` returning leaf indices and importance-sampling weights, and
    ``update_priorities`` for writing back fresh TD-errors after a
    gradient step.
    """

    def __init__(
        self,
        capacity: int = 10_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        priority_eps: float = 1e-3,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.priority_eps = priority_eps
        self.beta = beta_start
        self.max_priority = 1.0
        self._frame = 0
        self.tree = SumTree(capacity)
        self.data: List[Optional[tuple]] = [None] * capacity
        self._last_idx: Optional[int] = None

    def push(self, state, action, reward, next_state, done, next_allowed, teacher_action=None):
        transition = (
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            bool(done),
            tuple(int(a) for a in next_allowed),
            None if teacher_action is None else int(teacher_action),
        )
        data_idx = self.tree.write
        self.data[data_idx] = transition
        # New transitions get max priority so every one is sampled at least
        # once before its real TD-error is known.
        self.tree.add(self.max_priority ** self.alpha)
        self._last_idx = data_idx

    def mutate_last_reward(self, delta: float) -> None:
        """Add ``delta`` to the most recently pushed transition's reward (win/loss bonus)."""
        if self._last_idx is None:
            return
        s, a, r, ns, d, allowed, teacher_action = self.data[self._last_idx]
        self.data[self._last_idx] = (s, a, r + delta, ns, d, allowed, teacher_action)
        # The reward just changed materially -- make sure it gets replayed
        # soon rather than waiting on its stale priority.
        self.tree.update(self._last_idx + self.capacity - 1, self.max_priority ** self.alpha)

    def __len__(self) -> int:
        return self.tree.n_entries

    def sample(self, batch_size: int):
        self.beta = min(
            1.0, self.beta_start + self._frame * (1.0 - self.beta_start) / self.beta_frames
        )
        self._frame += 1

        total = self.tree.total()
        segment = total / batch_size
        leaf_indices: List[int] = []
        priorities: List[float] = []
        batch: List[tuple] = []
        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            cumulative = random.uniform(lo, hi)
            leaf_idx, priority, data_idx = self.tree.get(cumulative)
            leaf_indices.append(leaf_idx)
            priorities.append(priority)
            batch.append(self.data[data_idx])

        probs = np.array(priorities, dtype=np.float64) / total
        weights = (len(self) * probs) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones, next_allowed, teacher_actions = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
            next_allowed,
            teacher_actions,
            leaf_indices,
            torch.tensor(weights, dtype=torch.float32),
        )

    def update_priorities(self, leaf_indices: Sequence[int], td_errors: Sequence[float]) -> None:
        for leaf_idx, td_error in zip(leaf_indices, td_errors):
            priority = (abs(float(td_error)) + self.priority_eps) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(leaf_idx, priority)

    # ── temporal ordering (oldest-first, matching a deque's iteration order) ──

    def _temporal_order(self) -> List[int]:
        n = self.tree.n_entries
        if n < self.capacity:
            return list(range(n))
        w = self.tree.write
        return list(range(w, self.capacity)) + list(range(0, w))

    def state_dict(self) -> dict:
        order = self._temporal_order()
        transitions = [self.data[i] for i in order]
        priorities = [self.tree.leaf_priority(i) for i in order]
        states = (
            np.stack([t[0] for t in transitions])
            if transitions
            else np.empty((0, STATE_DIM), dtype=np.float32)
        )
        next_states = (
            np.stack([t[3] for t in transitions])
            if transitions
            else np.empty((0, STATE_DIM), dtype=np.float32)
        )
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_frames": self.beta_frames,
            "priority_eps": self.priority_eps,
            "beta": self.beta,
            "frame": self._frame,
            "max_priority": self.max_priority,
            "states": torch.from_numpy(states),
            "actions": torch.tensor([t[1] for t in transitions], dtype=torch.long),
            "rewards": torch.tensor([t[2] for t in transitions], dtype=torch.float32),
            "next_states": torch.from_numpy(next_states),
            "dones": torch.tensor([t[4] for t in transitions], dtype=torch.bool),
            "next_allowed": [list(t[5]) for t in transitions],
            "teacher_actions": [t[6] for t in transitions],
            "priorities": priorities,
        }

    def load_state_dict(self, payload: dict) -> None:
        self.capacity = int(payload["capacity"])
        self.alpha = float(payload.get("alpha", self.alpha))
        self.beta_start = float(payload.get("beta_start", self.beta_start))
        self.beta_frames = int(payload.get("beta_frames", self.beta_frames))
        self.priority_eps = float(payload.get("priority_eps", self.priority_eps))
        self.beta = float(payload.get("beta", self.beta_start))
        self._frame = int(payload.get("frame", 0))
        self.max_priority = 1.0
        self.tree = SumTree(self.capacity)
        self.data = [None] * self.capacity
        self._last_idx = None

        teacher_actions = payload.get("teacher_actions") or [None] * len(payload["actions"])
        priorities = payload.get("priorities") or []
        for state, action, reward, next_state, done, next_allowed, teacher_action, priority in zip(
            payload["states"].numpy(),
            payload["actions"].tolist(),
            payload["rewards"].tolist(),
            payload["next_states"].numpy(),
            payload["dones"].tolist(),
            payload["next_allowed"],
            teacher_actions,
            priorities,
        ):
            transition = (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
                tuple(int(a) for a in next_allowed),
                None if teacher_action is None else int(teacher_action),
            )
            data_idx = self.tree.write
            self.data[data_idx] = transition
            self.tree.add(float(priority))
            self._last_idx = data_idx
            self.max_priority = max(self.max_priority, float(priority))


__all__ = ["SumTree", "PrioritizedReplayBuffer"]
