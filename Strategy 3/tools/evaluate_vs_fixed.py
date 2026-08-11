"""
evaluate_vs_fixed.py
---------------------
Strategy 3's decided evaluation metric (PLAN.md §1, "(b) baseline-relative"):
seat-balanced win rate of a MonopolyZero (monopoly_bench) candidate against
Fixed-A/B/C, with a Wilson lower bound. This is the number to compare
against the measured asu_value_v1 baseline of 72/100
(Strategy 1/REPO_STUDY_NOTES.md, section 1) -- ASU is not seated as an
opponent here; it never is for this metric (PLAN.md §1).

The candidate is loaded through MonopolyZeroNet.load_inference, which
rejects a ruleset/state/action mismatch outright (CLAUDE.md's checkpoint-
identity rule) rather than silently loading into the wrong shape.

Usage:
    python tools/evaluate_vs_fixed.py --candidate RUN_DIR/champion.pt --games 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monopoly_bench.adapters import FixedAdapter
from monopoly_bench.arena import balanced_single_seats, play_game, summarize
from monopoly_bench.config import SearchConfig
from monopoly_bench.model import MonopolyZeroNet
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC


FIXED_TRIO = (FPAgentA, FPAgentB, FPAgentC)


def build_policies(candidate_seat: int, model: MonopolyZeroNet, search: SearchConfig):
    from monopoly_bench.adapters import SearchAdapter

    policies = {candidate_seat: SearchAdapter(model, search, self_play=False)}
    other_seats = [seat for seat in range(4) if seat != candidate_seat]
    for seat, agent_class in zip(other_seats, FIXED_TRIO):
        policies[seat] = FixedAdapter(agent_class)
    return policies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="MonopolyZero champion .pt (save_inference format)")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=9_100_000, help="Default: BenchmarkConfig.seeds.gate")
    parser.add_argument("--simulations", type=int, default=None, help="Override SearchConfig.simulations")
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--out", default=None, help="Optional path to write the JSON summary")
    args = parser.parse_args()

    model = MonopolyZeroNet.load_inference(args.candidate)
    search = SearchConfig() if args.simulations is None else SearchConfig(simulations=args.simulations)

    target_seats = balanced_single_seats(args.games)
    results = []
    for game_id, seats in enumerate(target_seats):
        candidate_seat = next(iter(seats))
        seed = args.seed_base + game_id
        policies = build_policies(candidate_seat, model, search)
        results.append(play_game(game_id=game_id, seed=seed, policies=policies, max_rounds=args.max_rounds))

    summary = summarize(results, target_seats)
    report = {
        "opponents": "Fixed-A/B/C (monopoly_game_engine.agents_fixed)",
        "candidate": str(args.candidate),
        "seat_balanced": True,
        "games": summary.games,
        "completed": summary.completed,
        "wins": summary.wins,
        "losses": summary.losses,
        "win_rate": summary.win_rate,
        "wilson_lower_95": summary.wilson_lower,
        "illegal_actions": summary.illegal_actions,
        "crashes": summary.crashes,
        "search_latency_p95_s": summary.latency_p95_s,
        "asu_value_v1_reference": "72/100 seat-balanced vs Fixed-A/B/C (Strategy 1/REPO_STUDY_NOTES.md)",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
