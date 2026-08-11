"""
measure_landing_frequency.py
-----------------------------
Empirical landing frequency per board square, measured directly on this
repo's ppo-plus-v2 engine (Strategy 3/monopoly_game_engine) instead of taken
from web sources. This matters because the engine has NO Chance/Community
Chest card effects (PPO_PLUS_RULES.md) -- so the common "orange properties
are best because of the Go Back 3 Spaces card" argument does not apply here.
Only what the engine actually implements (dice, doubles, jail, Go To Jail)
can shift the distribution away from a flat 1/40.

Runs games with the six Fixed personalities (A-F) round-robin so the
distribution reflects realistic play, not four copies of one policy.

Usage:
    python measure_landing_frequency.py --games 300
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

STRATEGY_3 = Path(__file__).resolve().parents[1] / "Strategy 3"
if str(STRATEGY_3) not in sys.path:
    sys.path.insert(0, str(STRATEGY_3))

from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES
from monopoly_game_engine.env import MonopolyEnv


SQUARE_NAMES = [
    "GO", "Mediterranean Ave", "Community Chest 1", "Baltic Ave", "Income Tax",
    "Reading Railroad", "Oriental Ave", "Chance 1", "Vermont Ave",
    "Connecticut Ave", "Jail/Just Visiting", "St. Charles Place",
    "Electric Company", "States Ave", "Virginia Ave", "Pennsylvania Railroad",
    "St. James Place", "Community Chest 2", "Tennessee Ave", "New York Ave",
    "Free Parking", "Kentucky Ave", "Chance 2", "Indiana Ave", "Illinois Ave",
    "B&O Railroad", "Atlantic Ave", "Ventnor Ave", "Water Works",
    "Marvin Gardens", "Go To Jail", "Pacific Ave", "North Carolina Ave",
    "Community Chest 3", "Pennsylvania Ave", "Short Line Railroad", "Chance 3",
    "Park Place", "Luxury Tax", "Boardwalk",
]

COLOR_GROUPS = {
    "brown": [1, 3],
    "light_blue": [6, 8, 9],
    "pink": [11, 13, 14],
    "orange": [16, 18, 19],
    "red": [21, 23, 24],
    "yellow": [26, 27, 29],
    "green": [31, 32, 34],
    "dark_blue": [37, 39],
    "railroad": [5, 15, 25, 35],
    "utility": [12, 28],
}


def run_games(n_games: int, max_rounds: int, seed_base: int) -> Counter:
    landings: Counter = Counter()
    classes = list(FP_AGENT_CLASSES)
    for game_index in range(n_games):
        env = MonopolyEnv(agent_ids=[0, 1, 2, 3], max_rounds=max_rounds)
        agents = [classes[(game_index + seat) % len(classes)](seat) for seat in range(4)]
        last_positions = [p.position for p in env.players]
        decisions = 0
        max_decisions = max_rounds * 4 * 40
        while not env.done and decisions < max_decisions:
            pid = env.whose_turn()
            allowed = env.get_allowed_actions(pid)
            action = agents[pid].choose_action(env)
            if action not in allowed:
                action = allowed[0]
            env.step(action)
            decisions += 1
            for seat, player in enumerate(env.players):
                if player.position != last_positions[seat]:
                    landings[player.position] += 1
                    last_positions[seat] = player.position
    return landings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--out", default=str(Path(__file__).parent / "landing_frequency.json"))
    args = parser.parse_args()

    landings = run_games(args.games, args.max_rounds, args.seed_base)
    total = sum(landings.values())
    ranked = sorted(range(40), key=lambda sq: -landings.get(sq, 0))

    report = {
        "games": args.games,
        "total_landings": total,
        "per_square": [
            {
                "square": sq,
                "name": SQUARE_NAMES[sq],
                "landings": landings.get(sq, 0),
                "frequency": landings.get(sq, 0) / total if total else 0.0,
            }
            for sq in range(40)
        ],
        "ranked_squares": [
            {"square": sq, "name": SQUARE_NAMES[sq], "landings": landings.get(sq, 0)}
            for sq in ranked
        ],
        "group_landings": {
            group: sum(landings.get(sq, 0) for sq in squares)
            for group, squares in COLOR_GROUPS.items()
        },
    }
    print(json.dumps(report["group_landings"], indent=2, sort_keys=True))
    print("\nTop 10 squares:")
    for entry in report["ranked_squares"][:10]:
        print(f"  {entry['square']:2d} {entry['name']:<22} {entry['landings']}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
