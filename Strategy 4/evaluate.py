"""
evaluate.py -- TheStatistician vs Fixed-A/B/C
-----------------------------------------------
Same protocol as the measured asu_value_v1 baseline of 72/100
(Strategy 1/REPO_STUDY_NOTES.md): seat-balanced, paired seeds, 25 seeds x 4
seats = 100 games, Wilson interval. Reuses the real harness pieces
(_new_seeded_game, _run_game, _ScriptedAdapter, wilson_interval) from
ASU_FROZEN_TEACHER.evaluate instead of a hand-rolled loop -- an
unwrapped-agent smoke test showed Fixed-A/B/C themselves can return an
illegal action in a rare state and rely on _ScriptedAdapter's fallback,
so this must go through the same wrapper.

Does not modify Strategy 3/ASU_FROZEN_TEACHER/evaluate.py -- this is a
standalone script that imports from it.

Usage:
    python evaluate.py --seeds 25
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

STRATEGY_3 = Path(__file__).resolve().parents[1] / "Strategy 3"
if str(STRATEGY_3) not in sys.path:
    sys.path.insert(0, str(STRATEGY_3))

from agent import TheStatistician  # noqa: E402

from ASU_FROZEN_TEACHER.evaluate import _ScriptedAdapter, _run_game, wilson_interval  # noqa: E402
from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES  # noqa: E402
from monopoly_game_engine.constants import NUM_PLAYERS  # noqa: E402


@dataclass(frozen=True)
class _Spec:
    kind: str
    policy_id: str


class _Factory:
    """Minimal stand-in for AgentFactory that also knows how to build
    TheStatistician; Fixed-A/B/C go through the real FP_AGENT_CLASSES."""

    def build(self, spec: _Spec, player_id: int):
        if spec.kind == "statistician":
            return _ScriptedAdapter(TheStatistician(player_id), player_id)
        letter_index = ord(spec.kind[-1]) - ord("a")
        agent = FP_AGENT_CLASSES[letter_index](player_id)
        return _ScriptedAdapter(agent, player_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--out", default=str(Path(__file__).parent / "eval_vs_fixed_abc_100.json"))
    args = parser.parse_args()

    focus_spec = _Spec("statistician", "statistician")
    opponent_specs = (_Spec("fixed-a", "fixed-a"), _Spec("fixed-b", "fixed-b"), _Spec("fixed-c", "fixed-c"))
    factory = _Factory()
    max_decisions = args.max_rounds * NUM_PLAYERS * 40

    games = []
    for seed in range(args.seed_base, args.seed_base + args.seeds):
        for focus_seat in range(NUM_PLAYERS):
            specs = [None] * NUM_PLAYERS
            specs[focus_seat] = focus_spec
            other_seats = [s for s in range(NUM_PLAYERS) if s != focus_seat]
            for seat, spec in zip(other_seats, opponent_specs):
                specs[seat] = spec
            result = _run_game(tuple(specs), focus_seat, int(seed), max_decisions, factory)
            games.append(result)

    completed = [g for g in games if not g["truncated"]]
    wins = sum(1 for g in completed if g["focus_won"])
    total = len(completed)
    lower, upper = wilson_interval(wins, total)
    fallbacks = sum(sum(g["scripted_compatibility_fallbacks"]) for g in games)

    report = {
        "agent": "TheStatistician (Strategy 4)",
        "opponents": "fixed-a, fixed-b, fixed-c",
        "seat_balanced": True,
        "games_requested": len(games),
        "games_completed": total,
        "truncations": len(games) - total,
        "wins": wins,
        "win_rate": wins / total if total else 0.0,
        "wilson_95_interval": [lower, upper],
        "scripted_compatibility_fallbacks": fallbacks,
        "asu_value_v1_reference": "72/100 seat-balanced vs Fixed-A/B/C (Strategy 1/REPO_STUDY_NOTES.md)",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
