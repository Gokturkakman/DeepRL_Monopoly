"""
agent.py -- TheStatistician
----------------------------
Hand-written heuristic Monopoly agent. Deliberately built on a DIFFERENT
basis than ASU_FROZEN_TEACHER's asu_value_v1 (which is
M_assets + R_short + R_long + M_monopoly, cash excluded from assets,
five-lap dice-enumeration rent projection, 2**missing_deeds monopoly
discount -- see Strategy 3/ASU_FROZEN_TEACHER/spec.py). This agent uses
none of that term structure. Per the team's relayed competition rule,
"ASU'yu birebir kullanmak yasak" / "ASU'yu birebir output klonlamak yasak"
covers structural cloning, not just calling ASU's code -- so this agent's
scoring is built from a different, independently-measured basis:

  score(square) = empirical_landing_frequency(square) * base_rent(square) / price(square)

i.e. "how often does anyone land here, times return per dollar invested."
The landing frequencies are NOT taken from a web source -- they are
measured directly on this repo's ppo-plus-v2 engine
(measure_landing_frequency.py), because that engine has no Chance/Community
Chest card effects (PPO_PLUS_RULES.md), which invalidates the common
"orange is best because of the Go Back 3 Spaces card" web claim for this
specific ruleset. See README.md for the measured numbers and sources.

Structurally this is a FixedPolicyAgent subclass (same framework as
agents_fixed.py's A-F personalities in Strategy 3/monopoly_game_engine),
reusing its shared trade/mortgage helpers -- only the buy/build/trade
priority function differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

STRATEGY_3 = Path(__file__).resolve().parents[1] / "Strategy 3"
if str(STRATEGY_3) not in sys.path:
    sys.path.insert(0, str(STRATEGY_3))

from monopoly_game_engine.actions import AUCTION_ACTION_TO_INCREMENT, OFFSETS, ActionType, AuctionAction
from monopoly_game_engine.agents_fixed import (
    FixedPolicyAgent,
    _buy_trade_action,
    _incoming_offer,
)
from monopoly_game_engine.constants import (
    COLOR_GROUPS,
    JAIL_BAIL,
    PROPERTIES,
    PROPERTY_IDS,
    RAILROAD_IDS,
    REAL_ESTATE_IDS,
    UTILITY_IDS,
)

_FREQ_PATH = Path(__file__).parent / "landing_frequency.json"


def _load_square_frequencies() -> dict[int, float]:
    """Empirical landing frequency per square, measured on this engine.
    Falls back to a flat distribution if the measurement hasn't been run."""
    if _FREQ_PATH.exists():
        data = json.loads(_FREQ_PATH.read_text())
        return {entry["square"]: entry["frequency"] for entry in data["per_square"]}
    return {sq: 1.0 / 40 for sq in range(40)}


_SQUARE_FREQ = _load_square_frequencies()


def _property_score(sq: int) -> float:
    """traffic (empirical) * return-per-dollar (base rent / price)."""
    info = PROPERTIES[sq]
    price = info["price"]
    if price <= 0:
        return 0.0
    if info["color"] == "railroad":
        base_rent = 25  # one-railroad-owned rent
    elif info["color"] == "utility":
        base_rent = 4 * 3.5  # 4x average dice roll, one utility owned
    else:
        base_rent = info["rent"][0]
    return _SQUARE_FREQ.get(sq, 1.0 / 40) * base_rent / price


_SCORES = {sq: _property_score(sq) for sq in PROPERTY_IDS}
_GROUP_SCORE = {
    color: sum(_SCORES[sq] for sq in squares) for color, squares in COLOR_GROUPS.items()
}
# Highest-traffic-per-dollar groups first; drives buy/build/trade priority.
_GROUP_PRIORITY = sorted(_GROUP_SCORE, key=lambda color: -_GROUP_SCORE[color])
_TOP_GROUPS = set(_GROUP_PRIORITY[:4])  # top quartile of color groups by measured score


class TheStatistician(FixedPolicyAgent):
    """
    Traits
    ------
    Buying     : score-driven (empirical traffic x rent/price), always buys
                 railroads/utilities (board-wide reachability), buys
                 denial pieces even off-priority when it would hand an
                 opponent a monopoly.
    Building   : develops completed monopolies in the top-scoring groups
                 first; cash floor before spending on houses.
    Trading    : only pursues/accepts trades that complete a monopoly in a
                 top-scoring group.
    Jail       : leaves immediately if cheap and cash is comfortable,
                 otherwise waits it out.
    Mortgaging : mortgages the lowest-scoring non-monopoly holding first.
    """

    _CASH_FLOOR = 150
    _BUILD_CASH_FLOOR = 100
    _JITTER_ATTRS = ("_CASH_FLOOR", "_BUILD_CASH_FLOOR")

    def _would_complete_opponent_monopoly(self, sq: int, env) -> bool:
        color = PROPERTIES[sq]["color"]
        if color in ("railroad", "utility"):
            return False
        group = COLOR_GROUPS[color]
        owners = {env.properties[s].owner for s in group if s != sq}
        if len(owners) == 1 and None not in owners:
            (owner,) = owners
            return owner != self.player_id
        return False

    def _should_buy(self, player, prop, env) -> bool:
        sq = prop.square_id
        price = PROPERTIES[sq]["price"]
        if not player.can_afford(price + self._CASH_FLOOR):
            return False
        if self._would_complete_opponent_monopoly(sq, env):
            return False  # never hand an opponent a free monopoly
        if PROPERTIES[sq]["color"] in ("railroad", "utility"):
            return True
        color = PROPERTIES[sq]["color"]
        group = COLOR_GROUPS[color]
        owned_by_me = sum(1 for s in group if env.properties[s].owner == self.player_id)
        if owned_by_me + 1 == len(group):
            return True  # completes my own monopoly regardless of tier
        return color in _TOP_GROUPS

    def _handle_jail(self, allowed, player) -> Optional[int]:
        if int(ActionType.USE_GOOJ_CARD) in allowed:
            return int(ActionType.USE_GOOJ_CARD)
        if int(ActionType.PAY_BAIL) in allowed and player.cash >= JAIL_BAIL * 3:
            return int(ActionType.PAY_BAIL)
        return None

    def _best_build_action(self, allowed, env) -> Optional[int]:
        player = env.players[self.player_id]
        candidates = []
        for i, sq in enumerate(REAL_ESTATE_IDS):
            prop = env.properties[sq]
            if prop.owner != self.player_id or not prop.is_monopoly:
                continue
            candidates.append((sq, i, prop))
        candidates.sort(key=lambda item: -_SCORES.get(item[0], 0.0))
        for sq, i, prop in candidates:
            house_price = PROPERTIES[sq]["house_price"]
            if not player.can_afford(house_price + self._BUILD_CASH_FLOOR):
                continue
            for action_key in ("improve_hotel", "improve_house"):
                action = OFFSETS[action_key] + i
                if action in allowed:
                    return action
        return None

    def _make_trade_offer(self, allowed, env) -> Optional[int]:
        pid = self.player_id
        for color in _GROUP_PRIORITY[:4]:
            group = COLOR_GROUPS[color]
            owned = [s for s in group if env.properties[s].owner == pid]
            if len(owned) + 1 != len(group):
                continue
            need = [
                s
                for s in group
                if env.properties[s].owner not in (pid, None)
                and not env.players[env.properties[s].owner].bankrupt
            ]
            if need:
                sq = need[0]
                target = env.properties[sq].owner
                action = _buy_trade_action(pid, target, sq, 2, env, allowed)
                if action is not None:
                    return action
        return None

    def _should_accept_trade(self, offer, env) -> bool:
        pid = self.player_id
        if offer.offered_prop:
            color = offer.offered_prop.color
            if color in _TOP_GROUPS:
                group = COLOR_GROUPS[color]
                would_own = sum(1 for s in group if env.properties[s].owner == pid) + 1
                if would_own == len(group):
                    return True
        return False

    def _maybe_mortgage(self, allowed, env) -> Optional[int]:
        player = env.players[self.player_id]
        if player.cash >= self._CASH_FLOOR:
            return None
        candidates = []
        for sq in PROPERTY_IDS:
            prop = env.properties.get(sq)
            if prop is None or prop.owner != self.player_id or prop.is_monopoly or prop.houses > 0:
                continue
            candidates.append(sq)
        candidates.sort(key=lambda sq: _SCORES.get(sq, 0.0))  # lowest score first
        for sq in candidates:
            idx = PROPERTY_IDS.index(sq)
            action = OFFSETS["mortgage"] + idx
            if action in allowed:
                return action
        return None


__all__ = ["TheStatistician"]
