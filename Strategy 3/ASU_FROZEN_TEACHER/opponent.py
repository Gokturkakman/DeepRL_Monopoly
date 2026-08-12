"""
ASU as a legitimate training/eval opponent -- never as a distillation source.

Team ruling (2026-08-10): playing against ASU is fine ("siz vs B vs C vs ASU
ok"); cloning its output into a trained model is not ("ASU birebir output
klonlamak yasak"). This module only ever gives ASU the FixedPolicyAgent-style
``choose_action(env) -> int`` interface so it can occupy a seat in an
opponent pool exactly like TheHoarder etc. Nothing here records ASU's
decisions anywhere; there is no label, buffer, or checkpoint path through
this class.

A hard per-decision timeout guards against ASUValueV1's measured latency
tail: ~0.045s/decision on ordinary states, but 10+ minutes observed on
pathological trade/auction-heavy states. Left unguarded inside a training
loop, one such state stalls Colab-hour burn with zero checkpoint progress.
"""

import random
import signal

from .core import ASUValueV1, preserve_global_rng


class _DecisionTimeout(RuntimeError):
    pass


def _raise_timeout(signum, frame):
    raise _DecisionTimeout()


class ASUOpponent:
    """Timeout-guarded ASU seat. Accepts the same constructor shape as the
    fixed-policy personalities (player_id, epsilon, threshold_jitter, rng)
    so it drops into monopoly_game_engine.train's opponent-pool sampling
    unmodified; threshold_jitter is accepted and ignored since ASU has no
    tunable cash thresholds (that's the point of a frozen spec)."""

    def __init__(
        self,
        player_id: int,
        epsilon: float = 0.0,
        threshold_jitter: float = 0.0,
        rng: random.Random | None = None,
        decision_timeout: float = 10.0,
    ):
        self.player_id = player_id
        self.epsilon = epsilon
        self.decision_timeout = decision_timeout
        self._rng = rng if rng is not None else random
        self._asu = ASUValueV1(player_id)

    def choose_action(self, env) -> int:
        allowed = env.get_allowed_actions(self.player_id)
        if not allowed:
            from monopoly_game_engine.actions import ActionType

            return int(ActionType.DO_NOTHING)

        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            return self._rng.choice(allowed)

        if self.decision_timeout > 0:
            signal.signal(signal.SIGALRM, _raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, self.decision_timeout)
        # except must wrap the finally, not sit beside it: if the alarm fires
        # exactly as self._asu.choose_action returns -- after the try's body
        # succeeded but before the finally's own setitimer(...,0) call runs --
        # _DecisionTimeout is raised from inside the finally, which a sibling
        # except can't catch. Observed in practice: crashed a live Colab run.
        try:
            try:
                with preserve_global_rng():
                    return self._asu.choose_action(env)
            finally:
                if self.decision_timeout > 0:
                    signal.setitimer(signal.ITIMER_REAL, 0)
        except _DecisionTimeout:
            return self._rng.choice(allowed)


__all__ = ["ASUOpponent"]
