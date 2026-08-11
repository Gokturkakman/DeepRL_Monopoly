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
traffic x rent/price score above a top-quartile-of-groups threshold, never
buys a piece that would hand an opponent a free monopoly, and prioritizes
house-building and trade offers by the same score. Jail: leaves immediately
if cash is comfortable, otherwise waits.

One-ply, no search/lookahead -- deliberately, given the measured cost of
deeper search on this engine (ASU's own truncated rollout variant runs
~0.045s/decision with multi-second tails; the CFR track measured 25 min for
one game). A hand-written multi-ply expectiminimax was considered and
rejected for today's deadline on that basis.

## Measured result

`evaluate.py` — same protocol as the measured `asu_value_v1` baseline
(`Strategy 1/REPO_STUDY_NOTES.md`): seat-balanced, 25 paired seeds x 4 seats
= 100 games vs Fixed-A/B/C, Wilson 95% interval.

```
wins: 45 / 100
win_rate: 0.45
wilson_95_interval: [0.356, 0.548]
```

(`eval_vs_fixed_abc_100.json`.) For comparison: `asu_value_v1` measured
72/100 in the same setup; PPO/DDQN (`Strategy 3/TRAINING_RESULTS.md`)
measured 0-2.5%. This is a same-day first pass, not a finished algorithm --
well above the RL baselines, well below ASU.

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
