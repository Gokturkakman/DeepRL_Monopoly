"""
Training loop (Section VII).

Trains one learning agent (PPO or DDQN, standard or hybrid) against
three fixed-policy opponents. Logs win rates every log_every games.

Call-site fixes applied (companions to agent_ppo.py fixes)
──────────────────────────────────────────────────────────
Fix 1 – choose_action() now returns 4 values for PPO: (action, log_prob,
    value, nn_allowed).  log_prob is None when the hybrid fixed policy
    fired; in that case we skip buffer storage entirely.

Fix 2 – store() now receives nn_allowed so it can record the per-step
    action mask alongside the transition.

Fix 3 – update() now receives (last_next_state, last_done) so that
    mid-game rollout boundaries are bootstrapped correctly by the critic
    instead of defaulting to zero.

Fix 5 – bounded potential shaping replaces repeated absolute state rewards.
    Each neural transition receives gamma*Phi(next)-Phi(current), and terminal
    transitions use zero terminal potential before the explicit win/loss bonus.
    This prevents policies from earning reward merely by taking extra actions.
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .actions import ActionType
from .agents_fixed import (
    FixedPolicyAgent,
    FPAgentA,
    FPAgentB,
    FPAgentC,
    TheBlocker,
    TheBuilder,
    TheDealMaker,
    TheGambler,
    TheHoarder,
    TheRailBaron,
)
from .constants import NUM_PLAYERS
from .env import MonopolyEnv

POTENTIAL_REWARD_LIMIT = 2.0

# Opponent pool for training: five noised personalities. TheRailBaron is
# deliberately excluded and reserved as a held-out generalization probe
# (see evaluate_held_out) -- a policy that only beats personalities it
# trained against has learned to exploit their specific decision
# boundaries, not to play Monopoly well in general.
TRAINING_POOL_CLASSES = [TheHoarder, TheDealMaker, TheGambler, TheBuilder, TheBlocker]
HELD_OUT_CLASS = TheRailBaron

DEFAULT_OPPONENT_EPSILON = 0.1
DEFAULT_OPPONENT_THRESHOLD_JITTER = 0.2


def _sample_opponents(
    other_pids: List[int],
    rng: random.Random,
    pool: List[type] | None = None,
    epsilon: float = DEFAULT_OPPONENT_EPSILON,
    threshold_jitter: float = DEFAULT_OPPONENT_THRESHOLD_JITTER,
    asu_factory=None,
    asu_probability: float = 0.0,
    self_play_pool=None,
    self_play_probability: float = 0.0,
    self_play_epsilon: float = 0.15,
) -> List[FixedPolicyAgent]:
    """
    Sample one noised fixed-policy opponent per seat, with replacement.

    ``asu_factory`` (e.g. ASU_FROZEN_TEACHER.opponent.ASUOpponent) is kept
    optional and out of this module's imports on purpose: train.py has no
    dependency on ASU_FROZEN_TEACHER, the caller supplies the class. Playing
    against ASU is allowed by competition rules; training on its output is
    not, and that constraint belongs to the caller, not to this sampler. At
    most one seat per game becomes ASU, at probability ``asu_probability``,
    since ASU's own decision cost is 1-2 orders of magnitude above a fixed
    heuristic's and can dominate wall-clock if given every seat.

    ``self_play_pool`` (monopoly_game_engine.self_play.SelfPlayPool) has no
    such cost cap -- a snapshot's forward pass costs about the same as the
    learner's own, so more than one self-play seat per game is fine.
    """
    classes = pool if pool is not None else TRAINING_POOL_CLASSES
    agents = []
    asu_seat_used = False
    for pid in other_pids:
        if asu_factory is not None and not asu_seat_used and rng.random() < asu_probability:
            agents.append(
                asu_factory(pid, epsilon=epsilon, threshold_jitter=threshold_jitter, rng=rng)
            )
            asu_seat_used = True
        elif (
            self_play_pool is not None
            and len(self_play_pool) > 0
            and rng.random() < self_play_probability
        ):
            agents.append(
                self_play_pool.sample_opponent(pid, rng, epsilon=self_play_epsilon)
            )
        else:
            agents.append(
                rng.choice(classes)(
                    pid, epsilon=epsilon, threshold_jitter=threshold_jitter, rng=rng
                )
            )
    return agents


def run_episode(
    env: MonopolyEnv,
    learning_agent,
    fp_agents: List[FixedPolicyAgent],
    agent_pid: int,
    is_ppo: bool,
    update_online: bool = True,
) -> Dict:
    """
    Run one complete game. The learning agent occupies position agent_pid,
    fixed-policy agents fill the other three slots.

    Returns metrics dict including:
      won, reward, steps, stats,
      trades_initiated, trades_accepted, trades_declined, properties_acquired
    """
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    update_stats = {}

    # ── Per-episode metric counters ──────────────────────────────────────────
    trades_initiated = 0
    trades_accepted = 0
    trades_declined = 0
    properties_acquired = 0

    prev_prop_count = len(env.players[agent_pid].properties)

    agents_map = {fp.player_id: fp for fp in fp_agents}
    agents_map[agent_pid] = learning_agent

    pending_transition = None

    def potential_delta(start: float, terminal: bool = False) -> float:
        next_potential = 0.0 if terminal else env._compute_reward(agent_pid)
        gamma = getattr(learning_agent, "gamma", 0.99)
        decision_penalty = getattr(learning_agent, "decision_penalty", 0.0)
        return float(
            np.clip(
                gamma * next_potential - start - decision_penalty,
                -POTENTIAL_REWARD_LIMIT,
                POTENTIAL_REWARD_LIMIT,
            )
        )

    max_steps = env.max_rounds * NUM_PLAYERS * 30
    step_count = 0

    while not done and step_count < max_steps:
        step_count += 1
        pid = env.whose_turn()

        if env.players[pid].bankrupt:
            env._advance_turn()
            continue

        allowed = env.get_allowed_actions(pid)
        if not allowed:
            allowed = [int(ActionType.DO_NOTHING)]

        if pid == agent_pid:
            # ── Learning agent ────────────────────────────────────────────
            if is_ppo:
                action, log_prob, value, nn_allowed = learning_agent.choose_action(
                    state, env, allowed
                )

                # A neural transition spans opponent and hybrid-policy actions
                # until the next state where the actor is actually consulted.
                if log_prob is not None and pending_transition is not None:
                    reward = potential_delta(pending_transition[4])
                    total_reward += reward
                    if update_online:
                        learning_agent.store(
                            pending_transition[0],
                            pending_transition[1],
                            pending_transition[2],
                            reward,
                            pending_transition[3],
                            False,
                            pending_transition[5],
                        )
                        if len(learning_agent.buffer) >= learning_agent.n_steps:
                            update_stats = learning_agent.update(
                                last_next_state=state,
                                last_done=False,
                            )
                            # The sampled action used the pre-update actor.
                            # Resample so the next rollout begins on-policy.
                            action, log_prob, value, nn_allowed = (
                                learning_agent.choose_action(state, env, allowed)
                            )
                    pending_transition = None
            else:
                action, nn_allowed = learning_agent.choose_training_action(
                    state, env, allowed
                )
                if nn_allowed is not None and pending_transition is not None:
                    reward = potential_delta(pending_transition[2])
                    total_reward += reward
                    if update_online:
                        learning_agent.store_transition(
                            pending_transition[0],
                            pending_transition[1],
                            reward,
                            state,
                            False,
                            nn_allowed,
                        )
                        update_stats = learning_agent.update()
                    pending_transition = None
                log_prob, value = 0.0, 0.0

            if action not in allowed:
                raise ValueError(
                    f"Learning agent selected illegal action {action}; allowed={allowed}"
                )

            # ── Count the action BEFORE stepping ──────────────────────────
            a = action
            buy_offset = int(ActionType.BUY_PROPERTY)
            acc_offset = int(ActionType.ACCEPT_TRADE)
            dec_offset = int(ActionType.DECLINE_TRADE)
            from .actions import OFFSETS as _OFF

            is_trade_offer = (
                _OFF["buy_trade"] <= a < _OFF["buy_trade"] + 252
                or _OFF["sell_trade"] <= a < _OFF["sell_trade"] + 252
                or _OFF["exch_trade"] <= a < _OFF["exch_trade"] + 2268
            )

            if a == buy_offset:
                properties_acquired += 1
            elif a == acc_offset:
                trades_accepted += 1
            elif a == dec_offset:
                trades_declined += 1
            elif is_trade_offer:
                trades_initiated += 1

            transition_state = state.copy()
            potential_before = env._compute_reward(agent_pid)
            next_state, _, done, info = env.step(action)

            # Detect property gained via accepted trade
            new_prop_count = len(env.players[agent_pid].properties)
            if a == acc_offset and new_prop_count > prev_prop_count:
                properties_acquired += new_prop_count - prev_prop_count
            prev_prop_count = new_prop_count

            steps += 1

            if is_ppo and log_prob is not None:
                pending_transition = (
                    transition_state,
                    action,
                    log_prob,
                    value,
                    potential_before,
                    nn_allowed,
                )
            elif not is_ppo and nn_allowed is not None:
                pending_transition = (
                    transition_state,
                    action,
                    potential_before,
                )
            state = next_state

        else:
            # ── Fixed-policy agent ────────────────────────────────────────
            agent = agents_map.get(pid)
            action = agent.choose_action(env) if agent else int(ActionType.END_TURN)
            if action not in allowed:
                action = (
                    int(ActionType.END_TURN)
                    if int(ActionType.END_TURN) in allowed
                    else allowed[0]
                )

            next_state, _, done, _ = env.step(action)
            state = next_state

    # ── Game over ─────────────────────────────────────────────────────────────
    winner = env.winner()
    won = winner == agent_pid

    if pending_transition is not None:
        reward = potential_delta(
            pending_transition[4] if is_ppo else pending_transition[2],
            terminal=True,
        )
        total_reward += reward
        if update_online and is_ppo:
            learning_agent.store(
                pending_transition[0],
                pending_transition[1],
                pending_transition[2],
                reward,
                pending_transition[3],
                True,
                pending_transition[5],
            )
        elif update_online:
            reward += getattr(learning_agent, "win_loss_bonus", 0.0) * (
                1 if won else -1
            )
            total_reward += getattr(learning_agent, "win_loss_bonus", 0.0) * (
                1 if won else -1
            )
            learning_agent.store_transition(
                pending_transition[0],
                pending_transition[1],
                reward,
                state,
                True,
                (),
            )

    if update_online:
        if is_ppo:
            learning_agent.add_win_loss(won)
            total_reward += getattr(learning_agent, "win_loss_bonus", 0.0) * (
                1 if won else -1
            )
            if len(learning_agent.buffer) > 0:
                update_stats.update(
                    learning_agent.update(last_next_state=state, last_done=True)
                )
        else:
            update_stats = learning_agent.update()
            learning_agent.finish_episode()

    return {
        "won": won,
        "reward": total_reward,
        "steps": steps,
        "stats": update_stats,
        "trades_initiated": trades_initiated,
        "trades_accepted": trades_accepted,
        "trades_declined": trades_declined,
        "properties_acquired": properties_acquired,
    }


def train(
    learning_agent,
    is_ppo: bool,
    hybrid: bool,
    n_games: int = 2000,
    log_every: int = 50,
    seed: int = 42,
    checkpoint_every: int = 0,
    checkpoint_path: str | None = None,
    watchdog=None,
    opponent_pool: List[type] | None = None,
    opponent_epsilon: float = DEFAULT_OPPONENT_EPSILON,
    opponent_threshold_jitter: float = DEFAULT_OPPONENT_THRESHOLD_JITTER,
    held_out_eval_games: int = 20,
    asu_factory=None,
    asu_probability: float = 0.0,
    self_play_pool=None,
    self_play_probability: float = 0.0,
    self_play_epsilon: float = 0.15,
    self_play_register_every: int = 200,
) -> Dict:
    """
    Main training function.

    Non-learner seats are resampled every game from ``opponent_pool``
    (default TRAINING_POOL_CLASSES) with epsilon-noise and threshold
    jitter, so the learner faces a shifting mix of personalities instead
    of the same three deterministic opponents for the whole run. Every
    ``log_every`` games a short held-out probe against noised TheRailBaron
    (never in the training pool) is logged alongside the in-pool win rate,
    so overfitting to the training opponents shows up during training
    instead of only at final evaluation.

    Returns:
        history: dict with win_rates (list per log_every games) and other metrics
    """
    random.seed(seed)
    np.random.seed(seed)

    agent_pid = learning_agent.player_id
    env = MonopolyEnv(agent_ids=[agent_pid], max_rounds=200)

    other_pids = [i for i in range(NUM_PLAYERS) if i != agent_pid]
    opponent_rng = random.Random(seed)

    history = defaultdict(list)
    wins_window = 0
    window_games = 0

    window_trades_initiated = 0
    window_trades_accepted = 0
    window_trades_declined = 0
    window_props_acquired = 0

    print(f"\n{'=' * 60}")
    print(
        f"Training {'Hybrid' if hybrid else 'Standard'} "
        f"{'PPO' if is_ppo else 'DDQN'} agent (player {agent_pid})"
    )
    print(f"Total games: {n_games}  |  Log every: {log_every}")
    print(f"{'=' * 60}")

    starting_games = int(getattr(learning_agent, "games_trained", 0))
    games_completed = 0
    for game_num in range(1, n_games + 1):
        absolute_game = starting_games + game_num
        episode_seed = seed + absolute_game - 1
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if watchdog is not None:
            try:
                watchdog.check()
            except RuntimeError as exc:
                if checkpoint_path:
                    path = Path(checkpoint_path)
                    emergency = path.with_name(
                        f"{path.stem}_emergency{path.suffix}"
                    )
                    learning_agent.save(str(emergency))
                    print(f"Memory watchdog stopped training: {exc}")
                    print(f"Emergency checkpoint: {emergency}")
                history["stopped_early"] = True
                history["stop_reason"] = str(exc)
                break

        fp_agents = _sample_opponents(
            other_pids,
            opponent_rng,
            pool=opponent_pool,
            epsilon=opponent_epsilon,
            threshold_jitter=opponent_threshold_jitter,
            asu_factory=asu_factory,
            asu_probability=asu_probability,
            self_play_pool=self_play_pool,
            self_play_probability=self_play_probability,
            self_play_epsilon=self_play_epsilon,
        )
        result = run_episode(env, learning_agent, fp_agents, agent_pid, is_ppo)
        games_completed = game_num
        learning_agent.games_trained = absolute_game

        if (
            self_play_pool is not None
            and hasattr(learning_agent, "online_net")
            and self_play_register_every > 0
            and absolute_game % self_play_register_every == 0
        ):
            self_play_pool.register(learning_agent)

        if (
            checkpoint_path
            and checkpoint_every > 0
            and absolute_game % checkpoint_every == 0
        ):
            learning_agent.save(checkpoint_path)

        if result["won"]:
            wins_window += 1
        window_games += 1

        window_trades_initiated += result["trades_initiated"]
        window_trades_accepted += result["trades_accepted"]
        window_trades_declined += result["trades_declined"]
        window_props_acquired += result["properties_acquired"]

        if game_num % log_every == 0:
            win_rate = wins_window / window_games * 100

            avg_trades_init = window_trades_initiated / window_games
            avg_trades_acc = window_trades_accepted / window_games
            avg_trades_dec = window_trades_declined / window_games
            avg_props = window_props_acquired / window_games

            history["win_rates"].append(win_rate)
            history["games"].append(absolute_game)
            history["rewards"].append(result["reward"])
            history["avg_trades_initiated"].append(avg_trades_init)
            history["avg_trades_accepted"].append(avg_trades_acc)
            history["avg_trades_declined"].append(avg_trades_dec)
            history["avg_properties_acquired"].append(avg_props)

            held_out_str = ""
            if held_out_eval_games > 0:
                saved_epsilon = getattr(learning_agent, "epsilon", None)
                held_out = evaluate_held_out(
                    learning_agent,
                    is_ppo,
                    n_games=held_out_eval_games,
                    n_runs=1,
                    seed=absolute_game,
                )
                if saved_epsilon is not None:
                    learning_agent.epsilon = saved_epsilon
                history["held_out_win_rate"].append(held_out["mean_win_rate"])
                held_out_str = f"  HeldOut%: {held_out['mean_win_rate']:5.1f}%"

            eps_str = (
                f"  ε={learning_agent.epsilon:.3f}"
                if hasattr(learning_agent, "epsilon")
                else ""
            )
            print(
                f"  Game {absolute_game:5d} | "
                f"Win%: {win_rate:5.1f}%{held_out_str}{eps_str} | "
                f"Props: {avg_props:.1f} | "
                f"Trades init/acc/dec: "
                f"{avg_trades_init:.1f}/{avg_trades_acc:.1f}/{avg_trades_dec:.1f}"
            )

            wins_window = 0
            window_games = 0
            window_trades_initiated = 0
            window_trades_accepted = 0
            window_trades_declined = 0
            window_props_acquired = 0

    history["resumed_from_games"] = starting_games
    history["games_completed_this_run"] = games_completed
    history["games_completed"] = int(
        getattr(learning_agent, "games_trained", games_completed)
    )
    return dict(history)


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    learning_agent,
    is_ppo: bool,
    n_games: int = 2000,
    n_runs: int = 5,
    seed: int = 0,
    opponent_classes: List[type] | None = None,
    opponent_epsilon: float = 0.0,
    opponent_threshold_jitter: float = 0.0,
) -> Dict:
    """
    Evaluate a trained agent over n_runs × n_games.
    Sets epsilon=0 for DDQN automatically.
    Returns win rates plus per-game averages of all tracked metrics.

    ``opponent_classes`` defaults to the original fixed A/B/C baseline
    (deterministic, no noise) for backward-compatible comparison against
    past TRAINING_RESULTS.md figures. Pass TRAINING_POOL_CLASSES or
    ``[HELD_OUT_CLASS]`` with nonzero noise for the diversity-aware probes.
    """
    if hasattr(learning_agent, "epsilon"):
        learning_agent.epsilon = 0.0

    agent_pid = learning_agent.player_id
    env = MonopolyEnv(agent_ids=[agent_pid], max_rounds=200)
    other_pids = [i for i in range(NUM_PLAYERS) if i != agent_pid]
    classes = (
        opponent_classes if opponent_classes is not None else [FPAgentA, FPAgentB, FPAgentC]
    )
    eval_rng = random.Random(seed)
    fp_agents = [
        classes[i % len(classes)](
            other_pids[i],
            epsilon=opponent_epsilon,
            threshold_jitter=opponent_threshold_jitter,
            rng=eval_rng,
        )
        for i in range(3)
    ]

    all_wins = []
    all_ti, all_ta, all_td, all_pa = [], [], [], []

    for run in range(n_runs):
        random.seed(seed + run)
        np.random.seed(seed + run)
        torch.manual_seed(seed + run)
        wins = 0
        run_ti = run_ta = run_td = run_pa = 0
        for _ in range(n_games):
            result = run_episode(
                env, learning_agent, fp_agents, agent_pid, is_ppo, update_online=False
            )
            if result["won"]:
                wins += 1
            run_ti += result["trades_initiated"]
            run_ta += result["trades_accepted"]
            run_td += result["trades_declined"]
            run_pa += result["properties_acquired"]

        rate = wins / n_games * 100
        all_wins.append(rate)
        all_ti.append(run_ti / n_games)
        all_ta.append(run_ta / n_games)
        all_td.append(run_td / n_games)
        all_pa.append(run_pa / n_games)
        print(
            f"  Run {run + 1}/{n_runs}: "
            f"win={rate:.1f}%  "
            f"props={run_pa / n_games:.1f}  "
            f"trades init/acc/dec="
            f"{run_ti / n_games:.1f}/{run_ta / n_games:.1f}/{run_td / n_games:.1f}"
        )

    mean_wr = float(np.mean(all_wins))
    std_wr = float(np.std(all_wins))
    print(f"\n  Overall win rate: {mean_wr:.2f}% ± {std_wr:.2f}%")
    print(f"  Avg props/game : {np.mean(all_pa):.2f} ± {np.std(all_pa):.2f}")
    print(f"  Avg trades initiated/game: {np.mean(all_ti):.2f}")
    print(f"  Avg trades accepted/game : {np.mean(all_ta):.2f}")
    print(f"  Avg trades declined/game : {np.mean(all_td):.2f}")

    return {
        "win_rates": all_wins,
        "mean_win_rate": mean_wr,
        "std_win_rate": std_wr,
        "avg_properties_acquired": float(np.mean(all_pa)),
        "avg_trades_initiated": float(np.mean(all_ti)),
        "avg_trades_accepted": float(np.mean(all_ta)),
        "avg_trades_declined": float(np.mean(all_td)),
    }


def evaluate_in_pool(
    learning_agent,
    is_ppo: bool,
    n_games: int = 2000,
    n_runs: int = 5,
    seed: int = 0,
    epsilon: float = DEFAULT_OPPONENT_EPSILON,
    threshold_jitter: float = DEFAULT_OPPONENT_THRESHOLD_JITTER,
) -> Dict:
    """Win rate against the noised training-pool mix (in-distribution)."""
    return evaluate(
        learning_agent,
        is_ppo,
        n_games=n_games,
        n_runs=n_runs,
        seed=seed,
        opponent_classes=TRAINING_POOL_CLASSES,
        opponent_epsilon=epsilon,
        opponent_threshold_jitter=threshold_jitter,
    )


def evaluate_held_out(
    learning_agent,
    is_ppo: bool,
    n_games: int = 500,
    n_runs: int = 3,
    seed: int = 0,
    epsilon: float = DEFAULT_OPPONENT_EPSILON,
    threshold_jitter: float = DEFAULT_OPPONENT_THRESHOLD_JITTER,
) -> Dict:
    """
    Generalization probe. TheRailBaron never appears in TRAINING_POOL_CLASSES,
    so a win rate here tracking the in-pool win rate is evidence of general
    play; a win rate that lags badly is evidence of overfitting to the
    training opponents' specific decision boundaries rather than to Monopoly
    itself. This is a separate number from vs-ASU: ASU is the distillation
    teacher, so vs-ASU alone cannot detect this failure mode.
    """
    return evaluate(
        learning_agent,
        is_ppo,
        n_games=n_games,
        n_runs=n_runs,
        seed=seed,
        opponent_classes=[HELD_OUT_CLASS],
        opponent_epsilon=epsilon,
        opponent_threshold_jitter=threshold_jitter,
    )
