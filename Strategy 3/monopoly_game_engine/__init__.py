"""
monopoly_game_engine – Shared ppo-plus-v2 Monopoly simulator
=============================================================

Based on:
  "Decision Making in Monopoly Using a Hybrid Deep Reinforcement
   Learning Approach"
  Bonjour et al., IEEE TETCI, Vol. 6, No. 6, December 2022.

Quick start
-----------
>>> from monopoly_game_engine import train_ppo, train_ddqn, evaluate_agent
>>> agent, history = train_ppo(hybrid=True, n_games=2000)
>>> results = evaluate_agent(agent, is_ppo=True, n_games=2000)
"""

import random

import numpy as np
import torch

from .env          import MonopolyEnv
from .agent_ppo    import PPOAgent
from .agent_ddqn   import DDQNAgent
from .agents_fixed import FPAgentA, FPAgentB, FPAgentC
from .train        import (
    train,
    evaluate,
    evaluate_in_pool,
    evaluate_held_out,
    TRAINING_POOL_CLASSES,
    HELD_OUT_CLASS,
)
from .state        import build_state_vector
from .actions      import ACTION_SPACE_SIZE, action_to_description


def train_ppo(
    hybrid: bool = True,
    player_id: int = 0,
    n_games: int = 2000,
    log_every: int = 100,
    checkpoint_every: int = 0,
    checkpoint_path: str | None = None,
    watchdog=None,
    seed: int = 42,
    resume_path: str | None = None,
    opponent_pool=None,
    opponent_epsilon: float = 0.1,
    opponent_threshold_jitter: float = 0.2,
    held_out_eval_games: int = 20,
    real_eval_games: int = 20,
    asu_factory=None,
    asu_probability: float = 0.0,
    self_play_pool=None,
    self_play_probability: float = 0.0,
    self_play_epsilon: float = 0.15,
    self_play_register_every: int = 200,
    **kwargs,
):
    """Train a PPO agent. Set hybrid=True for the hybrid approach."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = PPOAgent(player_id=player_id, hybrid=hybrid, **kwargs)
    if resume_path is not None:
        agent.load(resume_path)
        n_games = max(0, n_games - agent.games_trained)
    history = train(
        agent,
        is_ppo=True,
        hybrid=hybrid,
        n_games=n_games,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        checkpoint_path=checkpoint_path,
        watchdog=watchdog,
        seed=seed,
        opponent_pool=opponent_pool,
        opponent_epsilon=opponent_epsilon,
        opponent_threshold_jitter=opponent_threshold_jitter,
        held_out_eval_games=held_out_eval_games,
        real_eval_games=real_eval_games,
        asu_factory=asu_factory,
        asu_probability=asu_probability,
        self_play_pool=self_play_pool,
        self_play_probability=self_play_probability,
        self_play_epsilon=self_play_epsilon,
        self_play_register_every=self_play_register_every,
    )
    return agent, history


def train_ddqn(
    hybrid: bool = True,
    player_id: int = 0,
    n_games: int = 10_000,
    log_every: int = 100,
    checkpoint_every: int = 0,
    checkpoint_path: str | None = None,
    watchdog=None,
    seed: int = 42,
    resume_path: str | None = None,
    opponent_pool=None,
    opponent_epsilon: float = 0.1,
    opponent_threshold_jitter: float = 0.2,
    held_out_eval_games: int = 20,
    real_eval_games: int = 20,
    asu_factory=None,
    asu_probability: float = 0.0,
    self_play_pool=None,
    self_play_probability: float = 0.0,
    self_play_epsilon: float = 0.15,
    self_play_register_every: int = 200,
    **kwargs,
):
    """Train a DDQN agent. Set hybrid=True for the hybrid approach."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = DDQNAgent(player_id=player_id, hybrid=hybrid, **kwargs)
    if resume_path is not None:
        agent.load(resume_path)
        n_games = max(0, n_games - agent.games_trained)
    history = train(
        agent,
        is_ppo=False,
        hybrid=hybrid,
        n_games=n_games,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        checkpoint_path=checkpoint_path,
        watchdog=watchdog,
        seed=seed,
        opponent_pool=opponent_pool,
        opponent_epsilon=opponent_epsilon,
        opponent_threshold_jitter=opponent_threshold_jitter,
        held_out_eval_games=held_out_eval_games,
        real_eval_games=real_eval_games,
        asu_factory=asu_factory,
        asu_probability=asu_probability,
        self_play_pool=self_play_pool,
        self_play_probability=self_play_probability,
        self_play_epsilon=self_play_epsilon,
        self_play_register_every=self_play_register_every,
    )
    return agent, history


def evaluate_agent(agent, is_ppo: bool, n_games: int = 2000, n_runs: int = 5):
    """Evaluate a trained agent against the fixed A/B/C baseline (legacy, deterministic)."""
    return evaluate(agent, is_ppo=is_ppo, n_games=n_games, n_runs=n_runs)


__all__ = [
    "MonopolyEnv",
    "PPOAgent", "DDQNAgent",
    "FPAgentA", "FPAgentB", "FPAgentC",
    "train_ppo", "train_ddqn", "evaluate_agent",
    "evaluate_in_pool", "evaluate_held_out",
    "TRAINING_POOL_CLASSES", "HELD_OUT_CLASS",
    "build_state_vector", "ACTION_SPACE_SIZE", "action_to_description",
]
