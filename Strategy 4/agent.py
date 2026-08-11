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

import copy
import json
import random
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

    _JEOPARDY_WEIGHT = 0.5

    def _lookahead_best(self, candidates: List[int], env) -> Optional[int]:
        """1-ply lookahead over a short candidate list: clone env, apply
        each candidate, keep the one with the best resulting position --
        net-worth potential (``env._compute_reward``, bounded
        self-vs-mean-opponent net worth) minus a jeopardy penalty (see
        ``_jeopardy``: fraction of opponent squares whose current rent
        would bankrupt us). Every caller here restricts candidates to
        deterministic state mutations (build/sell-house/mortgage) -- no
        dice roll, no opponent turn in between -- so this is an exact
        one-step comparison, not an approximate rollout. Restores global
        RNG state per CLAUDE.md's cloning-for-lookahead rule and never
        mutates ``env`` itself (`_lookahead_best` only ever operates on
        `copy.deepcopy` clones)."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        pid = self.player_id
        outer_state = random.getstate()
        try:
            best_action, best_value = candidates[0], float("-inf")
            for action in candidates:
                clone = copy.deepcopy(env)
                clone.step(action)
                value = clone._compute_reward(pid) - self._JEOPARDY_WEIGHT * self._jeopardy(clone)
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_action
        finally:
            random.setstate(outer_state)

    def _is_opponent_denial_target(self, sq: int, env) -> bool:
        """True if every other square in this square's colour group is
        already owned by a single opponent -- sq is their last missing
        piece. Buying it ourselves permanently denies that opponent the
        monopoly (a 3-piece group split 2-1 can never complete); refusing
        it just leaves the piece for them to pick up later. This corrects
        an earlier inverted reading of the same condition that treated
        "opponent is one piece short" as a reason to pass instead of the
        strongest possible reason to buy."""
        color = PROPERTIES[sq]["color"]
        if color in ("railroad", "utility"):
            return False
        group = COLOR_GROUPS[color]
        owners = {env.properties[s].owner for s in group if s != sq}
        if len(owners) == 1 and None not in owners:
            (owner,) = owners
            return owner != self.player_id
        return False

    def _jeopardy(self, env) -> float:
        """Fraction of opponent-owned squares whose *current* rent (houses,
        railroad count, utility count all included) exceeds our cash right
        now -- how much of the board would bankrupt us on a single bad
        landing. Independently sourced from a published, non-ASU academic
        source (Khan, "AI for Board Games" final-year project, University
        of Leeds School of Computing) rather than derived from ASU's own
        rent-projection formula -- a simple current-rent-vs-cash fraction,
        not ASU's dice-enumeration multi-lap projection. Dynamic: recomputed
        every call from live state, unlike this agent's static per-square
        traffic score."""
        pid = self.player_id
        player = env.players[pid]
        deadly = 0
        for sq in PROPERTY_IDS:
            prop = env.properties[sq]
            if prop.owner is None or prop.owner == pid:
                continue
            owner = env.players[prop.owner]
            rent = prop.get_rent(
                dice_roll=7,
                num_railroads=owner.railroads_owned(),
                num_utilities=owner.utilities_owned(),
            )
            if rent > player.cash:
                deadly += 1
        return deadly / len(PROPERTY_IDS)

    def _should_buy(self, player, prop, env) -> bool:
        sq = prop.square_id
        price = PROPERTIES[sq]["price"]
        if not player.can_afford(price + self._CASH_FLOOR):
            return False
        if self._is_opponent_denial_target(sq, env):
            return True  # deny an opponent's last piece -- always worth it
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
        """Gather every affordable build action across owned monopolies,
        then 1-ply lookahead (`_lookahead_best`) to pick the one that
        actually leaves the best net-worth potential -- rather than the
        static traffic x rent/price score, which ranks squares in
        isolation and ignores how much of the build budget each one
        consumes relative to its immediate payoff."""
        player = env.players[self.player_id]
        candidates = []  # (sq, action) -- sorted by score before lookahead
        for i, sq in enumerate(REAL_ESTATE_IDS):
            prop = env.properties[sq]
            if prop.owner != self.player_id or not prop.is_monopoly:
                continue
            house_price = PROPERTIES[sq]["house_price"]
            if not player.can_afford(house_price + self._BUILD_CASH_FLOOR):
                continue
            for action_key in ("improve_hotel", "improve_house"):
                action = OFFSETS[action_key] + i
                if action in allowed:
                    candidates.append((sq, action))
                    break
        candidates.sort(key=lambda item: -_SCORES.get(item[0], 0.0))
        return self._lookahead_best([action for _, action in candidates], env)

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
        """Routine low-cash management is unchanged in spirit from baseline
        (mortgage a bare property, otherwise do nothing -- a dip under the
        floor is not an emergency), except the choice of *which* bare
        property now goes through 1-ply lookahead instead of the static
        score. Only in the engine's actual forced-debt phase
        (``env.debt_player == self.player_id``, unpaid rent that must be
        settled before the turn can continue) does this escalate through
        houses/hotels and monopoly mortgages -- without that escalation the
        agent has nothing left to offer, `choose_action` falls through to an
        illegal END_TURN, and the harness's compatibility fallback picks an
        arbitrary legal action (`allowed[0]`) instead of a deliberate one."""
        player = env.players[self.player_id]
        pid = self.player_id
        forced_debt = getattr(env, "debt_player", None) == pid

        if not forced_debt and player.cash >= self._CASH_FLOOR:
            return None

        owned = [sq for sq in PROPERTY_IDS if env.properties[sq].owner == pid]
        owned.sort(key=lambda sq: _SCORES.get(sq, 0.0))  # weakest first (tie-break)

        # 1. Mortgage a bare, non-monopoly property -- baseline behaviour,
        #    safe in both the routine and forced cases. Lookahead picks
        #    which one among however many are legal.
        bare_candidates = []
        for sq in owned:
            prop = env.properties[sq]
            if prop.is_monopoly or prop.houses > 0 or prop.mortgaged:
                continue
            idx = PROPERTY_IDS.index(sq)
            action = OFFSETS["mortgage"] + idx
            if action in allowed:
                bare_candidates.append(action)
        if bare_candidates:
            return self._lookahead_best(bare_candidates, env)

        if not forced_debt:
            return None  # routine dip, nothing safe to mortgage -- do nothing

        # Forced debt with no bare property left: escalate.
        # 2. Sell houses/hotels off a developed property.
        house_candidates = []
        for sq in owned:
            prop = env.properties[sq]
            if prop.houses <= 0:
                continue
            i = REAL_ESTATE_IDS.index(sq)
            action_key = "sell_hotel" if prop.houses == 5 else "sell_house"
            action = OFFSETS[action_key] + i
            if action in allowed:
                house_candidates.append(action)
        if house_candidates:
            return self._lookahead_best(house_candidates, env)

        # 3. Last resort: mortgage a monopoly holding too (only legal once
        #    step 2 has cleared any houses on that group).
        monopoly_candidates = []
        for sq in owned:
            prop = env.properties[sq]
            if prop.mortgaged:
                continue
            idx = PROPERTY_IDS.index(sq)
            action = OFFSETS["mortgage"] + idx
            if action in allowed:
                monopoly_candidates.append(action)
        return self._lookahead_best(monopoly_candidates, env)


__all__ = ["TheStatistician"]
