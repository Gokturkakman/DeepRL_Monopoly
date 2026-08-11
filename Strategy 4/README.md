# Strategy 4 — TheStatistician (hand-written algorithm, not RL, not ASU)

Built same-day (2026-08-11) as a Plan B candidate per the team's relayed
competition rules: an open-sourced, hand-written algorithm is a plausible
path, and the team explicitly suggested an "anti-ASU" model is smart since
most teams will converge toward approximating ASU (`Strategy 3/CLAUDE.md`,
"Competition rules" section).

## Basis — deliberately different from ASU

`ASU_FROZEN_TEACHER.asu_value_v1` (`Strategy 3/ASU_FROZEN_TEACHER/spec.py`)
scores states as `M_assets + R_short + R_long + M_monopoly`: list-price
assets with cash excluded, a five-lap dice-enumeration rent projection, and
a `2 ** missing_deeds` monopoly-completion discount. "ASU'yu birebir
kullanmak/klonlamak yasak" (the team's relayed rule) is read here as
covering that term structure, not just calling ASU's code, so
`TheStatistician` uses none of it.

Instead:

```
score(square) = empirical_landing_frequency(square) * base_rent(square) / price(square)
```

"How often does anyone land here, times return per dollar invested" — a
traffic-times-ROI heuristic, structurally unrelated to ASU's rent-projection
formula.

## The landing frequencies are measured, not looked up

A web search for "best Monopoly property" turns up the standard claim that
orange properties dominate because of the "Go Back 3 Spaces" Chance card
routing players there from Electric Company. **That card effect does not
exist in this engine** — `Strategy 3/PPO_PLUS_RULES.md`: "Chance and
Community Chest spaces have no card effect. There is no shuffled card
deck." Citing the web number here would have been wrong for this specific
ruleset.

`measure_landing_frequency.py` instead runs 200 games (Fixed-A/B/C/D/E/F
round robin, up to 100 rounds each) directly on `Strategy 3`'s
`monopoly_game_engine` and counts landings per square. Result
(`landing_frequency.json`, group totals):

| Group | Landings |
|---|---:|
| Railroad | 6595 |
| Orange | 5454 |
| Yellow | 5284 |
| Red | 5236 |
| Green | 5063 |
| Pink | 4898 |
| Light blue | 4560 |
| Utility | 3418 |
| Brown | 3016 |
| Dark blue | 2814 |

Railroads dominate for a mechanism-independent reason: every square on the
board reaches a railroad within one roll. Orange still ranks highest among
color groups even without the Chance-card mechanism, driven by jail
dynamics (jail is the single most-landed square: 3485 landings) — doubles,
Go To Jail, and bail-outs are all implemented in this ruleset and shift
traffic even without cards.

## Agent

`agent.py`'s `TheStatistician` is a `FixedPolicyAgent` subclass (same
framework as `Strategy 3/monopoly_game_engine/agents_fixed.py`'s A-F
personalities, reusing its shared trade/mortgage helpers) with a
score-driven buy/build/trade/mortgage policy: buys railroads/utilities
unconditionally (board-wide reachability), buys real estate by the
traffic x rent/price score above a top-quartile-of-groups threshold, always
buys a square that is an opponent's last missing piece for a monopoly
(permanent denial -- see `_is_opponent_denial_target`), and prioritizes
house-building and trade offers by the same score. Jail: leaves immediately
if cash is comfortable, otherwise waits. Cash management escalates only in
the engine's genuine forced-debt phase (`env.debt_player`): sell houses,
then mortgage even a monopoly holding, rather than falling through to the
evaluation harness's undefined `allowed[0]` compatibility fallback.

Build and liquidation decisions now use a 1-ply lookahead
(`_lookahead_best`): clone the env, apply each legal candidate action, read
`env._compute_reward(pid)` off the clone, keep the best, restore global RNG
state per `CLAUDE.md`'s cloning-for-lookahead rule, never mutate the live
`env`. Scoped to two decision points only -- build (which monopoly to
develop) and liquidation (which property to mortgage/sell-house) -- because
both are single deterministic state mutations (no dice, no opponent reply
in between), so the comparison is exact, not an approximation. The buy/pass
decision was deliberately left out of scope: passing routes through a
multi-bidder auction, which would need a full deterministic auction
sub-simulation to evaluate honestly, and that was judged too much
additional engineering risk for the remaining time. No deeper multi-turn
rollout was attempted either, given the measured cost of search on this
engine (ASU's own rollout variant runs ~0.045s/decision with multi-second
tails; the CFR track measured 25 min for one game).

**Caveat on what the build lookahead actually optimizes**: `_compute_reward`
is built on `Property.calculate_net_worth()`, where a developed property
contributes `houses * house_price * (1 + 0.5*houses)`. Going from 0->1
house nets `+0.5*house_price` in immediate accounted value; 1->2 nets
`+1.5*house_price`. That multiplier grows with existing house count, so the
lookahead systematically favors *continuing to develop the
already-most-built property* over starting a new one -- a reasonable
Monopoly instinct (concentrate rent), but it is a side effect of the
engine's net-worth accounting formula, not a rediscovery of the traffic x
rent/price logic from measured landing frequency. Ties (common, since the
metric is quantized in `0.5*house_price` steps) fall back to the traffic
score via a pre-sort, so the original heuristic still breaks ties.

## Measured result

`evaluate.py` — same protocol as the measured `asu_value_v1` baseline
(`Strategy 1/REPO_STUDY_NOTES.md`): seat-balanced, 25 paired seeds x 4 seats
= 100 games vs Fixed-A/B/C, Wilson 95% interval. Directly comparable to the
72/100 ASU figure, which was also measured on 100 games.

```
wins: 44 / 100
win_rate: 0.44
wilson_95_interval: [0.347, 0.538]
```

(`eval_vs_fixed_abc_100.json`.) Because 100 games is a small sample (the
project's own reporting rule flags exactly this), the same build was also
run at 4x the sample size, 100 seeds x 4 seats = 400 games
(`eval_vs_fixed_abc_400.json`), alongside the pre-lookahead build on the
identical protocol (`eval_vs_fixed_abc_400_pre_fix_baseline.json`):

| Build | N=100 | N=400 |
|---|---:|---:|
| Original (pre any fix in this document) | 45/100 (0.356–0.548) | 161/400 = 40.25% (0.356–0.451) |
| Current (fixes + lookahead) | 44/100 (0.347–0.538) | 169/400 = 42.25% (0.375–0.471) |

The N=400 comparison is the one to trust: +8 games on a 4x larger,
independent sample, same direction as the smaller run, but the Wilson
intervals overlap substantially. This is not strong evidence of a real
improvement -- it is weak, directionally-consistent evidence, and should be
reported as such. For comparison: `asu_value_v1` measured 72/100 in the
same setup; PPO/DDQN (`Strategy 3/TRAINING_RESULTS.md`) measured 0-2.5%.
**TheStatistician does not beat ASU** -- it sits roughly 30 points below it
at both sample sizes, and nothing tried here closed a meaningful fraction
of that gap.

## Tuning attempts, 2026-08-11/12 (mostly ruled out or neutral)

Eight changes were tried against the 25-seed protocol (deterministic given
fixed seeds, so a win-count delta *on that specific 100-game sample* is
reproducible -- but reproducible is not the same as a real effect; see the
N=400 comparison above, which is what actually supports keeping the
lookahead change). Kept whatever fixed a real bug or showed a
directionally consistent effect at larger N; dropped everything that
regressed win rate outright.

| Change | Result (N=100) | Verdict |
|---|---:|---|
| Baseline (pre-fix) | 45/100, 1297 illegal-action fallbacks | reference |
| + denial-buy on any opponent-touched group + naive debt escalation | 30/100 | reverted -- see below |
| Debt-gated liquidation escalation only (`env.debt_player`-gated) | 45/100, 1059 fallbacks | **kept** |
| + denial-target fix (`_is_opponent_denial_target`) | 44/100, 862 fallbacks | **kept** (real bug, ~flat win rate) |
| Cash floor sweep: 100 / 150 / 300 / 400 (build floor fixed) | 41 / 44 / 44 / 33 | 150 (current) already best |
| Build floor sweep: 100 / 250 / 400 (cash floor fixed) | 44 / 41 / 33 | 100 (current) already best |
| `_TOP_GROUPS` size sweep: top 3/4/5/6/7/8 of 10 | 38/44/39/32/32/38 | top-4 (current) already best |
| + 1-ply lookahead on build/liquidation (`_lookahead_best`) | 44/100 (N=400: 42.25% vs 40.25%) | **kept**, weak positive signal at N=400 |

Two real fixes landed (independent of the lookahead question):

1. **Liquidation was either too timid or too trigger-happy depending on
   game phase.** The original `_maybe_mortgage` only ever tried mortgaging
   a bare non-monopoly property; when none existed during the engine's
   actual forced-debt phase (`env.debt_player == pid`, unpaid rent that
   must be resolved before the turn continues), it returned `None`,
   `choose_action` fell through to an illegal `END_TURN`, and
   `evaluate.py`'s `_ScriptedAdapter` silently substituted an arbitrary
   legal action (`allowed[0]`) -- 1297 times across 100 games. The first
   fix attempt escalated through house-selling and monopoly-mortgaging
   *unconditionally* whenever cash dipped under the routine $150 floor,
   which touched houses/monopolies even in harmless, non-emergency dips
   and cost 15 games (45 -> 30). Gating the escalation strictly to
   `env.debt_player == pid` fixed that: routine dips behave exactly as
   before (mortgage-or-do-nothing), and only genuine forced debt escalates.
   Net effect on illegal-action fallbacks: real and large (1297 -> ~880
   after all fixes, and 4842 -> 3211 at N=400).
2. **`_would_complete_opponent_monopoly` was inverted.** It detected when
   a square was an opponent's last missing piece for a monopoly, then
   used that to *refuse* the purchase ("never hand an opponent a free
   monopoly"). But refusing doesn't protect anyone -- it leaves the piece
   for the opponent to buy instead, handing them exactly the monopoly the
   check was meant to prevent. Buying it ourselves is the actual denial
   move (a 3-piece group split 2-1 can never complete). Renamed to
   `_is_opponent_denial_target` and flipped to a forced buy. Logically
   correct; measured win-rate effect was within noise.

**Conclusion**: threshold tuning on the one-ply traffic x rent/price
heuristic is exhausted -- six independent levers (buy priority, cash
floors, group-tier width) all landed at or below the existing defaults.
Adding 1-ply lookahead to build/liquidation decisions produced a small,
directionally consistent gain at N=400 (40.25% -> 42.25%) that does not
rise to a confident improvement given the overlapping intervals, and even
if fully credited, closes only a small fraction of the ~30-point gap to
ASU's 72/100.

## Tuning attempts, 2026-08-12 (round 2: dynamic/jeopardy and buy lookahead)

Following an academic (non-ASU) source -- Khan, *AI for Board Games*,
University of Leeds School of Computing final-year project -- which
describes a published, six-factor **dynamic** state-evaluation heuristic
(`value = money + total_rent*rm - opp_money*opmm - opp_rent*oprm -
jeopardy*ja`, recomputed every decision from live state) as the general
reason a dynamic evaluator beats a static one, two follow-up changes were
tried and measured (25-seed protocol):

| Change | Result (N=100) | Verdict |
|---|---:|---|
| `_jeopardy`: fraction of opponent squares whose current rent > our cash, scaling the buy-decision cash floor | 44/100, 877 fallbacks (bit-identical to before) | reverted -- zero measured effect |
| Same, also scaling the routine mortgage-trigger floor | 44/100, 877 fallbacks (still identical) | reverted -- zero measured effect |
| `_jeopardy` folded into the existing build/liquidation `_lookahead_best` value (`reward - 0.5*jeopardy`) instead of a floor multiplier | 44/100, 873 fallbacks | **kept** -- cheap, theoretically sound, marginally fewer fallbacks, no downside measured |
| 1-ply lookahead on the buy decision itself for groups below the top quartile (buy if `_compute_reward` after buying beats before) | **39/100** | reverted -- regression |

Two real findings, not just parameter noise:

1. **Bankruptcy-risk avoidance isn't the bottleneck for this matchup.**
   `_jeopardy` verified working in isolation (a standalone debug script
   showed it reaching up to 0.25 mid-game), yet gating buy/mortgage
   decisions on it changed *zero* games out of 100, twice, at two
   different injection points. Combined with `truncations: 0` (every game
   ends by elimination, never the round cap), the honest read is: losses
   here come from falling behind on board position early, not from acute
   bankruptcy events a safety margin could have prevented. This redirects
   effort away from safety-margin tuning and toward acquisition strategy.
2. **A pure net-worth lookahead is a bad buy-decision proxy, and this
   explains itself from the accounting formula.** `Property.calculate_net_worth()`
   values *any* unmortgaged property at `price * 2.5` (or `*5.0` once it's
   part of a monopoly) regardless of which color group it's in -- so
   `env._compute_reward(pid)` after any affordable purchase is almost
   always higher than before, independent of whether the property is
   actually a good one by traffic/rent economics. A lookahead built on
   this metric degenerates toward "buy everything affordable," diluting
   cash away from the top-tier groups that actually matter and losing 5
   games (44 -> 39). The existing `_TOP_GROUPS` static threshold, despite
   being "just" a fixed priority list, encodes real information (measured
   landing frequency) that the net-worth accounting formula does not
   capture at all -- it was not the weak link.

**Architectural finding, not just a result**: true lookahead on the
buy/auction decision (comparing "buy now" against "decline and let the
auction resolve") was scoped in round 1 as the most promising unexplored
lever, but turns out to be infeasible as originally conceived --
`choose_action(env)` only receives the shared game-state `env`, never the
other seated agents' policy objects, so there is no way to simulate how
Fixed-A/B/C (or ASU) would actually bid in a cloned auction. Any
auction-outcome estimate would have to assume a generic opponent bidding
model rather than exactly replay one, which is a materially different (and
much weaker) kind of lookahead than the deterministic single-action cases
(build/liquidation) that worked cleanly. This rules out "just add auction
lookahead" as a near-term option; a genuine advance here would need either
a state-value function good enough to not need auction simulation at all
(closer to ASU's own approach, and constrained by the same anti-cloning
rule), or accepting an approximate/statistical opponent bidding model.

## Dependency note

This folder is not self-contained -- `agent.py` and `evaluate.py` add
`../Strategy 3` to `sys.path` to reuse its `monopoly_game_engine` and
`ASU_FROZEN_TEACHER.evaluate` harness rather than duplicating ~17k lines of
engine code again. It is committed on the same branch as `Strategy 3`
(`feature/strategy-3-hybrid`) for that reason; it will not run standalone
without `Strategy 3/` present alongside it.

## Open questions, not resolved here

- Whether a hand-written algorithm submission is allowed at all is recorded
  as an open question in `Strategy 3/CLAUDE.md`'s "Competition rules"
  section, not a settled ruling.
- The team's rules also explicitly bless a hybrid: "10 tane edge case için
  algo, kalanı RL" -- this agent could instead become a bounded scripted
  component inside the RL agent rather than a separate submission. Not
  decided here.
