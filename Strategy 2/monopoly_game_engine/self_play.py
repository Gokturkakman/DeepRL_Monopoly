"""
Self-play opponent pool: rotating snapshots of the learner's own past
weights, played epsilon-greedy rather than argmax.

Why not argmax: pairing the learner against a purely deterministic copy of
itself invites the same failure mode as the fixed personalities -- learning
to counter one exact decision boundary rather than playing well in general.
Why not the *current* weights: training against a live-updating copy of
yourself has no fixed target to learn against and tends to cycle (see
AlphaStar/OpenAI Five's league play -- the reason for sampling from a
history of past checkpoints rather than "best response to latest self").

Snapshots are kept in memory (CPU state_dict copies) so registering one is
cheap enough to call every few hundred games; disk persistence is optional
and only for cross-run resumoption, not required for a single training run.
"""

import random
from pathlib import Path
from typing import List, Optional

import torch

from .actions import ActionType
from .agent_ppo import fixed_accept_trade_decision, fixed_buy_decision
from .networks import DDQNNetwork

HYBRID_FIXED_ACTIONS = {int(ActionType.BUY_PROPERTY), int(ActionType.ACCEPT_TRADE)}


class SelfPlayOpponent:
    """Frozen snapshot of a past learner checkpoint, played epsilon-greedy."""

    def __init__(
        self,
        player_id: int,
        network: DDQNNetwork,
        hybrid: bool,
        epsilon: float = 0.15,
        rng: Optional[random.Random] = None,
    ):
        self.player_id = player_id
        self.network = network
        self.hybrid = hybrid
        self.epsilon = epsilon
        self._rng = rng if rng is not None else random

    def choose_action(self, env) -> int:
        pid = self.player_id
        allowed = env.get_allowed_actions(pid)
        if not allowed:
            return int(ActionType.DO_NOTHING)

        if self.hybrid and int(ActionType.BUY_PROPERTY) in allowed:
            if fixed_buy_decision(env, pid):
                return int(ActionType.BUY_PROPERTY)

        if self.hybrid and int(ActionType.ACCEPT_TRADE) in allowed:
            pending = env._incoming_trade(pid)
            if pending is not None:
                return (
                    int(ActionType.ACCEPT_TRADE)
                    if fixed_accept_trade_decision(env, pid)
                    else int(ActionType.DECLINE_TRADE)
                )

        nn_allowed = (
            [a for a in allowed if a not in HYBRID_FIXED_ACTIONS] if self.hybrid else list(allowed)
        )
        if not nn_allowed:
            nn_allowed = [int(ActionType.DO_NOTHING)]
        state = env._get_state(pid)
        return self.network.get_action(state, nn_allowed, self.epsilon)


class SelfPlayPool:
    """
    Rotating snapshots of a DDQNAgent's online_net. ``register(agent)``
    copies current weights into memory (and optionally disk); the pool
    keeps the most recent ``max_size`` snapshots so training samples a
    *history* of past selves rather than only the newest one.
    """

    def __init__(self, max_size: int = 8, save_dir: Optional[str] = None):
        self.max_size = max_size
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots: List[dict] = []

    def register(self, agent) -> None:
        state = {k: v.detach().cpu().clone() for k, v in agent.online_net.state_dict().items()}
        snapshot = {
            "state_dict": state,
            "hidden_dim": agent.hidden_dim,
            "hybrid": agent.hybrid,
            "games_trained": agent.games_trained,
        }
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.max_size:
            self._snapshots.pop(0)
        if self.save_dir:
            torch.save(snapshot, self.save_dir / f"snapshot_{agent.games_trained:07d}.pt")

    def __len__(self) -> int:
        return len(self._snapshots)

    def sample_opponent(
        self, player_id: int, rng: random.Random, epsilon: float = 0.15
    ) -> SelfPlayOpponent:
        snapshot = rng.choice(self._snapshots)
        network = DDQNNetwork(hidden_dim=snapshot["hidden_dim"])
        network.load_state_dict(snapshot["state_dict"])
        network.eval()
        return SelfPlayOpponent(
            player_id, network, snapshot["hybrid"], epsilon=epsilon, rng=rng
        )


__all__ = ["SelfPlayOpponent", "SelfPlayPool"]
