"""
evaluate_vs_fixed.py
---------------------
Strategy 3's decided evaluation metric (PLAN.md §1, "(b) baseline-relative"):
seat-balanced win rate of a trained PPO/DDQN checkpoint against Fixed-A/B/C,
with a Wilson interval. Compare the reported win_rate against the measured
asu_value_v1 baseline of 72/100 (Strategy 1/REPO_STUDY_NOTES.md §1).

Reuses ASU_FROZEN_TEACHER.evaluate.evaluate_lineup -- the exact function that
produced that 72/100 number -- so the comparison is the same code path, not
just the same statistic. 25 paired seeds x 4 seats = 100 games matches that
artifact's methodology.

ASU is never an opponent here (PLAN.md §1: baseline-relative, not
head-to-head). It is also never a training input for this checkpoint --
PLAN.md §2 restricts ASU to a training-time opponent seat only
(--asu-opponent-probability in tools/train_and_save.py); the MonopolyZero/
monopoly_bench self-play track was dropped because its Trainer bootstraps on
ASU's chosen actions unconditionally, which the Hard Rules in CLAUDE.md do
not permit outside the SLM/Gemma track.

Usage:
    python tools/evaluate_vs_fixed.py --checkpoint ppo:artifacts/ppo_plus/ppo_hybrid_2000_v2.pt
    python tools/evaluate_vs_fixed.py --checkpoint ddqn:artifacts/ddqn_plus/ddqn_hybrid_2000_v2.pt --seeds 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ASU_FROZEN_TEACHER.evaluate import evaluate_lineup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="ppo:/path or ddqn:/path")
    parser.add_argument("--seeds", type=int, default=25, help="Paired seeds; games = seeds * 4 (seat-balanced)")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    seeds = tuple(range(args.seed_base, args.seed_base + args.seeds))
    result = evaluate_lineup(args.checkpoint, ("fixed-a", "fixed-b", "fixed-c"), seeds=seeds)

    focus_id = result["policy_ids"]["focus"]
    focus_summary = result["win_rates"][focus_id]
    report = {
        "checkpoint": args.checkpoint,
        "opponents": "fixed-a, fixed-b, fixed-c",
        "seat_balanced": True,
        "games": focus_summary["games"],
        "wins": focus_summary["wins"],
        "win_rate": focus_summary["win_rate"],
        "wilson_95_interval": focus_summary["wilson_95"],
        "truncations": result["truncations"],
        "asu_value_v1_reference": "72/100 seat-balanced vs Fixed-A/B/C (Strategy 1/REPO_STUDY_NOTES.md)",
        "raw": result,
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
