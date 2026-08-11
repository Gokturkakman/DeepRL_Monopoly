"""
train_and_save.py
-----------------
Trains a hybrid PPO or DDQN agent against the three fixed-policy opponents
and saves a resumable model checkpoint.

Usage:
    python tools/train_and_save.py                        # default 2000 games
    python tools/train_and_save.py --games 5000           # more training
    python tools/train_and_save.py --algo ddqn --games 10000
    python tools/train_and_save.py --algo ppo --games 2000 --out my_model.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_guard import GIB, MemoryWatchdog
from monopoly_game_engine import train_ddqn, train_ppo


def _history_segment(history: dict) -> dict:
    return {
        key: history.get(key)
        for key in (
            "resumed_from_games",
            "games_completed_this_run",
            "games_completed",
            "elapsed_seconds",
            "peak_rss_gib",
            "peak_cuda_gib",
        )
    }


def merge_training_history(previous: dict | None, current: dict) -> dict:
    """Merge a completed resume segment into its checkpoint history."""
    if not previous:
        current["training_segments"] = [_history_segment(current)]
        return current

    expected = int(current.get("resumed_from_games", 0))
    actual = int(previous.get("games_completed", 0))
    if actual != expected:
        raise ValueError(
            f"History/checkpoint mismatch: history has {actual} games, "
            f"checkpoint resumed from {expected}."
        )

    merged = dict(current)
    series_keys = {
        key
        for history in (previous, current)
        for key, value in history.items()
        if isinstance(value, list) and key != "training_segments"
    }
    for key in series_keys:
        merged[key] = list(previous.get(key, [])) + list(current.get(key, []))

    previous_segments = previous.get("training_segments")
    if previous_segments is None:
        previous_segments = [_history_segment(previous)]
    merged["training_segments"] = list(previous_segments) + [
        _history_segment(current)
    ]
    merged["resumed_from_games"] = int(previous.get("resumed_from_games", 0))
    merged["games_completed_this_run"] = int(
        previous.get("games_completed_this_run", actual)
    ) + int(current.get("games_completed_this_run", 0))
    merged["elapsed_seconds"] = float(previous.get("elapsed_seconds", 0.0)) + float(
        current.get("elapsed_seconds", 0.0)
    )
    if merged["games_completed_this_run"]:
        merged["seconds_per_game"] = (
            merged["elapsed_seconds"] / merged["games_completed_this_run"]
        )
    for key in ("peak_rss_gib", "peak_cuda_gib"):
        merged[key] = max(
            float(previous.get(key, 0.0)), float(current.get(key, 0.0))
        )
    return merged


def main():
    parser = argparse.ArgumentParser(description="Train and save a Monopoly DRL agent")
    parser.add_argument(
        "--algo",
        choices=["ppo", "ddqn"],
        default="ppo",
        help="Algorithm to use (default: ppo)",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        default=True,
        help="Use hybrid mode (default: True)",
    )
    parser.add_argument(
        "--no-hybrid",
        dest="hybrid",
        action="store_false",
        help="Disable hybrid mode (use standard DRL)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=2000,
        help="Number of training games (default: 2000)",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Output path for saved model weights"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="PyTorch device (default: auto)",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=None,
        help=(
            "Per-game exploration decay (DDQN only). Default None keeps DDQNAgent's "
            "0.9995 (tuned for the paper's 10,000-game run -- reaches its epsilon "
            "floor around game ~6000). For shorter budgets, pass a faster decay so "
            "the run actually reaches an exploitation-dominated phase; e.g. 0.9985 "
            "reaches floor ~game 2000."
        ),
    )
    parser.add_argument("--epsilon-end", type=float, default=None)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "DDQN learning rate (default None: DDQNAgent's 1e-5, tuned for the "
            "paper's 10,000-game run). Diagnosed 2026-08-11: a 1000-game run at "
            "1e-5 with target_update_freq=500 games (2 syncs total) stayed at 0%% "
            "win rate despite correctly-signed rewards throughout -- the network "
            "wasn't absorbing credit fast enough for the run length. Try 1e-4 to "
            "1e-3 for shorter runs, together with --target-update-freq-steps."
        ),
    )
    parser.add_argument(
        "--target-update-freq-steps",
        type=int,
        default=None,
        help=(
            "Sync target network every N gradient steps instead of every "
            "target-update-freq GAMES (default None: games-based only, ~500 "
            "games/sync). Composes additively with the games-based sync."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help=(
            "Fixed games-per-log-window (default None: games//50). Also gates the "
            "held-out probe, so on a resumed multi-chunk run prefer a fixed value "
            "over the games//50 default -- that default rescales with the CURRENT "
            "chunk's --games target, so early small chunks fire the held-out probe "
            "(and its held_out_eval_games extra simulated games) far more often "
            "than intended."
        ),
    )
    parser.add_argument("--stop-rss-gib", type=float, default=3)
    parser.add_argument("--hard-rss-gib", type=float, default=4)
    parser.add_argument("--min-available-gib", type=float, default=2)
    parser.add_argument(
        "--opponent-epsilon",
        type=float,
        default=0.1,
        help="Per-decision random-action probability for training opponents (default: 0.1)",
    )
    parser.add_argument(
        "--opponent-threshold-jitter",
        type=float,
        default=0.2,
        help="Fractional randomization of fixed-opponent cash thresholds (default: 0.2)",
    )
    parser.add_argument(
        "--held-out-eval-games",
        type=int,
        default=20,
        help="Games vs held-out TheRailBaron logged every log-every games (default: 20; 0 disables)",
    )
    parser.add_argument(
        "--asu-opponent-probability",
        type=float,
        default=0.0,
        help=(
            "Per-game probability one opponent seat becomes ASU (default: 0.0, off). "
            "Rules allow playing against ASU, never training on its output -- this "
            "only ever adds ASU as an adversary, nothing here records its decisions. "
            "ASU's own decision cost is 1-2 orders of magnitude above a fixed "
            "heuristic's (measured ~0.045s/decision, occasional multi-second tail), "
            "so keep this low (0.01-0.02) unless you have compute to spare."
        ),
    )
    parser.add_argument(
        "--asu-decision-timeout",
        type=float,
        default=5.0,
        help="Per-decision wall-clock cap in seconds for the ASU opponent seat (default: 5.0)",
    )
    parser.add_argument(
        "--self-play-probability",
        type=float,
        default=0.0,
        help="Per-seat probability of a past learner checkpoint as opponent (default: 0.0, off)",
    )
    parser.add_argument("--self-play-pool-size", type=int, default=8)
    parser.add_argument("--self-play-epsilon", type=float, default=0.15)
    parser.add_argument(
        "--self-play-register-every",
        type=int,
        default=200,
        help="Snapshot the learner into the self-play pool every N games (default: 200)",
    )
    args = parser.parse_args()

    asu_factory = None
    if args.asu_opponent_probability > 0:
        from ASU_FROZEN_TEACHER.opponent import ASUOpponent
        from functools import partial

        asu_factory = partial(ASUOpponent, decision_timeout=args.asu_decision_timeout)

    self_play_pool = None
    if args.self_play_probability > 0 and args.algo == "ddqn":
        from monopoly_game_engine.self_play import SelfPlayPool

        self_play_pool = SelfPlayPool(max_size=args.self_play_pool_size)

    # Default output filename
    if args.out is None:
        mode = "hybrid" if args.hybrid else "standard"
        args.out = str(ROOT / "artifacts" / "ppo_plus" / f"{args.algo}_{mode}_model.pt")
    if args.resume and not Path(args.out).exists():
        parser.error(f"resume checkpoint does not exist: {args.out}")
    history_path = str(Path(args.out).with_suffix("")) + "_history.json"
    previous_history = None
    if args.resume and Path(history_path).exists():
        with open(history_path) as f:
            previous_history = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"  Algorithm : {args.algo.upper()}")
    print(f"  Mode      : {'Hybrid' if args.hybrid else 'Standard'}")
    print(f"  Games     : {args.games}")
    print(f"  Device    : {args.device}")
    print(f"  Save to   : {args.out}")
    print(f"{'=' * 60}\n")

    start = time.time()
    watchdog = MemoryWatchdog(
        stop_rss_gib=args.stop_rss_gib,
        hard_rss_gib=args.hard_rss_gib,
        min_available_gib=args.min_available_gib,
    )

    if args.algo == "ppo":
        agent, history = train_ppo(
            hybrid=args.hybrid,
            player_id=0,
            n_games=args.games,
            log_every=(args.log_every if args.log_every is not None else max(1, args.games // 50)),
            device=args.device,
            checkpoint_every=args.checkpoint_every,
            checkpoint_path=args.out,
            watchdog=watchdog,
            seed=args.seed,
            resume_path=args.out if args.resume else None,
            opponent_epsilon=args.opponent_epsilon,
            opponent_threshold_jitter=args.opponent_threshold_jitter,
            held_out_eval_games=args.held_out_eval_games,
            asu_factory=asu_factory,
            asu_probability=args.asu_opponent_probability,
        )
    else:
        ddqn_kwargs = {}
        if args.epsilon_decay is not None:
            ddqn_kwargs["epsilon_decay"] = args.epsilon_decay
        if args.epsilon_end is not None:
            ddqn_kwargs["epsilon_end"] = args.epsilon_end
        if args.lr is not None:
            ddqn_kwargs["lr"] = args.lr
        if args.target_update_freq_steps is not None:
            ddqn_kwargs["target_update_freq_steps"] = args.target_update_freq_steps
        agent, history = train_ddqn(
            hybrid=args.hybrid,
            player_id=0,
            n_games=args.games,
            log_every=(args.log_every if args.log_every is not None else max(1, args.games // 50)),
            device=args.device,
            checkpoint_every=args.checkpoint_every,
            checkpoint_path=args.out,
            watchdog=watchdog,
            seed=args.seed,
            resume_path=args.out if args.resume else None,
            opponent_epsilon=args.opponent_epsilon,
            opponent_threshold_jitter=args.opponent_threshold_jitter,
            held_out_eval_games=args.held_out_eval_games,
            asu_factory=asu_factory,
            asu_probability=args.asu_opponent_probability,
            self_play_pool=self_play_pool,
            self_play_probability=args.self_play_probability,
            self_play_epsilon=args.self_play_epsilon,
            self_play_register_every=args.self_play_register_every,
            **ddqn_kwargs,
        )

    elapsed = time.time() - start
    games_completed = history.get("games_completed", 0)
    games_this_run = history.get("games_completed_this_run", games_completed)
    peak_rss_gib = watchdog.peak_rss / GIB
    peak_cuda_gib = 0.0
    status = "stopped early" if history.get("stopped_early") else "complete"
    print(f"\nTraining {status} in {elapsed:.1f}s")
    print(f"Games completed: {games_completed}/{args.games}")
    if games_this_run:
        print(f"Mean wall time: {elapsed / games_this_run:.3f}s/game")
    print(f"Peak process RSS: {peak_rss_gib:.2f} GiB")
    if getattr(agent, "device", torch.device("cpu")).type == "cuda":
        peak_cuda_gib = torch.cuda.max_memory_allocated() / GIB
        print(f"Peak CUDA memory: {peak_cuda_gib:.2f} GiB")

    history["elapsed_seconds"] = elapsed
    history["seconds_per_game"] = elapsed / games_this_run if games_this_run else None
    history["peak_rss_gib"] = peak_rss_gib
    history["peak_cuda_gib"] = peak_cuda_gib

    # Save model weights
    agent.save(args.out)
    print(f"Model saved to: {args.out}")

    # Save training history
    history = merge_training_history(previous_history, history)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Print final win rate
    if history.get("win_rates"):
        final_wr = history["win_rates"][-1]
        best_wr = max(history["win_rates"])
        print(f"\nFinal win rate : {final_wr:.1f}%")
        print(f"Best win rate  : {best_wr:.1f}%")


if __name__ == "__main__":
    main()
