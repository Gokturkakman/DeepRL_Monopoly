# CLAUDE.md — DeepRL_Monopoly

Project memory for Claude Code. Read this before touching any code in this
directory tree.

## Project purpose

Build the strongest possible Monopoly-playing agent — highest achievable
win rate, not just "plays legally" — using reinforcement learning, within a
5-day, CPU-trainable budget. "Strongest" is judged by measured win rate
against the fixed-policy and ASU baselines under the rules below, not by
algorithm sophistication for its own sake.

The repo runs three parallel research tracks against one shared game engine:

| Track | Approach | Owns |
|---|---|---|
| CFR | Counterfactual Regret Minimization, rollout-based | `RL_CFR_MONOPOLYMODIFIED/` |
| PPO / DDQN | Model-free deep RL, on-policy and off-policy | `monopoly_game_engine/` |
| SLM | Gemma 4 QLoRA distillation from the ASU teacher | `SLM_HANDMADE_MONOPOLY/` |

`ASU_FROZEN_TEACHER/` is a hand-built heuristic teacher, currently the
strongest measured policy in this repo. It exists to evaluate against and
(for the SLM track only) to distill from — see the prohibition below.
`monopoly_bench/` (MonopolyZero) combines a PPO warm start, ASU imitation,
and self-play search on top of the same engine.

Full architecture, vocabulary, and a traced example decision:
[`REPO_STUDY_NOTES.md`](REPO_STUDY_NOTES.md). Full rules contract:
[`PPO_PLUS_RULES.md`](PPO_PLUS_RULES.md). Measured results so far:
[`TRAINING_RESULTS.md`](TRAINING_RESULTS.md).

## Hard rules

- **The engine is the source of truth.** No policy moves pieces or
  transfers money directly; it returns an action ID and
  `env.get_allowed_actions(pid)` / `env.step(action)` validate and apply it.
- **Legal-action masks are mandatory, not an optimization.** Every neural
  output must be masked to legal actions before sampling. An illegal action
  from an agent is a policy failure — fail closed, never silently
  substitute a "safe" action on its behalf.
- **`ppo-plus-v2` is the ruleset/state/action contract**, not a PPO version
  number. It fixes a 300-float observation and a 2,958-action space. Never
  invent or persist an action ID without going through
  `monopoly_game_engine/actions.py`'s offsets and the current ruleset
  version. A v1 checkpoint or a v1 result is not evidence about v2 — reject
  mismatched ruleset/state/action metadata explicitly on load rather than
  loading into the wrong shape.
- **Playing against ASU is allowed; training on ASU's output is not**,
  except for the SLM/Gemma track, whose entire job is supervised
  distillation from ASU-labeled decisions. PPO, DDQN, CFR, and MonopolyZero
  self-play must learn from game outcomes and their own experience, not
  from ASU's chosen actions. (`_sample_opponents` in `train.py` documents
  this boundary — ASU can occupy an opponent seat, but nothing records its
  decisions as training targets there.)
- **Heavy training, broad search, and Gemma QLoRA belong in Colab.** Local
  runs are for inspection, unit tests, and small smoke checks. The local
  trainer enforces RSS/available-RAM guards for this reason — do not raise
  them to force a heavy run to stay local instead of moving it to Colab.
- **Dataset splits keep whole games together.** Never split individual
  transitions/rows across train/val/test — that leaks a trajectory across
  the split.
- **Report win rates with their full setup**, never a bare percentage:
  opponent identity (which fixed personalities / ASU, at what noise),
  seat balance (focus policy rotated through all seats), round cap and
  truncation/tiebreak behavior, and ideally a Wilson interval for a small
  sample. "72%" alone is not a result.
- **Exact-agreement / imitation metrics are not win rate.** For the SLM
  track, JSON legality rate and teacher-agreement rate are offline gates,
  not evidence of playing strength — only measured game win rate is.
- **Determinism when cloning the engine for lookahead** (ASU rollout,
  MonopolyZero search): restore global RNG state after the clone and never
  mutate the caller's live environment.

## Competition rules (team channel, "Cem Abi", 2026-08-10/11)

Relayed informally over chat; quoted/paraphrased here so they're not lost.
Where the source message was ambiguous, that is noted rather than resolved
by guessing. These are in addition to, not a replacement for, the Hard
Rules above — where they overlap (ASU), they agree with and confirm the
Hard Rules' reading.

- **Using ASU directly is forbidden** ("ASU'yu birebir kullanmak yasak").
- **Playing against ASU is allowed** (you vs B vs C vs ASU is fine), **but
  literally cloning ASU's output is forbidden** ("ASU'ya karşı
  oynayabilirsiniz ama ASU'yu birebir output klonlamak yasak"). This is the
  same distinction as the Hard Rules' "playing against ASU is allowed;
  training on its output is not" — it directly confirms PLAN.md §2's
  corrected reading (opponent seat only, no bootstrap/imitation).
- **Winning criterion**: not most matches won, but the highest *overall* win
  rate against everyone -- the most generalizable model. Since every team
  will likely try to imitate/approximate ASU, deliberately building an
  "anti-ASU" model (diverging from where others converge) was suggested as
  a smart strategy. This is a strategic suggestion, not a rule.
- **Rules may still change until Friday** (this week) -- "monopoly oynuyoruz
  poker değil, ayak üstü tüm düzeni değiştirirsek kimse oyuna
  odaklanmayacak, değişebilir diye cumaya kadar." Treat anything in this
  section as provisional until then.
- **Unclear, not resolved**: "emrullahın kuralları ... bu gece ... askıya
  alıyorum" -- some other, unspecified rule set ("Emrullah's rules") was
  described as suspended for one night. What that rule set contained is not
  in the relayed message; do not assume it is void beyond that night without
  checking with the team.
- **Open question, not yet answered**: whether submitting a hand-written
  (non-RL, non-ASU) algorithm is allowed at all. The relayed message treats
  an open-sourced, group-linked algorithm submission and an explicit hybrid
  (a small hand-written component for a bounded set of edge cases, RL for
  the rest) as plausibly acceptable, but this was framed as a question to
  the team, not a settled ruling.
- **Compute**: Colab Pro was reported to run 5-6 instances in parallel and to
  provide 100+ hours/month; a "Colab CLI" was mentioned as a way to run
  Claude Code there. Unverified by us -- relayed as-is.
- Incentive (Cem Abi's own initiative, outside the competition rules): an
  AWS credit reward for whoever writes a paper or builds a benchmark from
  this work.

## Known-hard problem, current state

Both the completed DDQN and PPO v2 baselines finished near 0% win rate
against Fixed-A/B/C. Diagnosed causes so far: sparse delayed credit over a
long horizon, a huge structured action space dominated by trade
combinatorics, opponent non-stationarity (three deterministic fixed
opponents), and default hyperparameters (DDQN learning rate, target-sync
interval) tuned for a 10,000-game paper run rather than the 1,000–2,000
game budget actually used. This is an active, in-progress fix, not a
closed question — see `Strategy 2/` for the current PPO-focused work.

## Compute

CPU-only locally, 5-day team deadline. Heavy runs move to Google Colab
(see the rule above) — a Colab connection/notebook setup is being added for
this so training can run there instead of tying up a laptop.
