"""Play from the CFR blueprint, with the heuristic policy as a safety net.

Choosing among the blueprint's mixed actions
--------------------------------------------
`mode="purify"` plays the most likely action.  `mode="sample"` samples the
thresholded distribution with a deterministic hash of (hand, node), so a given
spot always produces the same action -- reproducible for testing, and it costs
nothing in duplicate variance (measured: our own randomness moves the paired
std by under 3%, see dev/variance_test.py).

Low-probability actions are dropped before sampling (threshold 0.15 by
default): in an abstraction they mostly reflect overfitting or incomplete
convergence, and the published comparisons found thresholding and purification
both beat playing the raw average strategy against real opponents.

Fallbacks: if the history somehow leaves the tree, or the infoset was never
reached in training, the heuristic policy decides.  Neither should happen in
practice; both are guarded so they can never cost an error.
"""
from __future__ import annotations

from . import blueprint as BP
from .obs import CHECK_CALL, FOLD, PREFLOP
from .policy import Policy

NAMES = ("FOLD", "CHECK_CALL", "RAISE_HALF", "RAISE_POT")


def _hash01(a: int, b: int) -> float:
    """Deterministic pseudo-random number in [0, 1) from two ints."""
    x = (a * 0x9E3779B1 + b * 0x85EBCA77 + 0x165667B1) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x ^= x >> 12
    x = (x * 0x297A2D39) & 0xFFFFFFFF
    x ^= x >> 15
    return x / 4294967296.0


class BlueprintPolicy:
    def __init__(self, strength, bp: BP.Blueprint | None = None, fallback: Policy | None = None,
                 mode: str = "sample", threshold: float = 0.15):
        self.s = strength
        self.bp = bp if bp is not None else BP.Blueprint()
        self.fallback = fallback if fallback is not None else Policy(strength)
        self.mode = mode
        self.threshold = threshold
        self.misses = 0

    def act(self, obs, adj=None) -> str:
        try:
            a = self._blueprint_action(obs)
        except Exception:
            a = None
        if a is None:
            self.misses += 1
            return self.fallback.act(obs, adj)
        return a

    def _blueprint_action(self, obs):
        bp = self.bp
        if not bp.ok:
            return None
        node = bp.node(obs.action_history)
        if node < 0:
            return None
        role = 0 if obs.is_button else 1
        if int(bp.actor[node]) != role:
            return None
        street = obs.street
        if street == PREFLOP:
            b = BP.bucket(self.s, obs.hole_cards, (), 0)
        else:
            b = BP.bucket(self.s, obs.hole_cards, obs.board, street)
        probs = bp.probs(node, b)
        if probs is None:
            return None
        legal = obs.legal_actions

        if self.mode == "purify":
            best, bp_ = None, -1.0
            for i, pr in enumerate(probs):
                if pr > bp_ and NAMES[i] in legal:
                    best, bp_ = NAMES[i], pr
            return best

        th = self.threshold
        kept = [(NAMES[i], pr) for i, pr in enumerate(probs) if pr >= th and NAMES[i] in legal]
        if not kept:
            kept = [(NAMES[i], pr) for i, pr in enumerate(probs) if NAMES[i] in legal and pr > 0]
        if not kept:
            return None
        tot = sum(pr for _, pr in kept)
        r = _hash01(obs.hand_index * 7919 + street, node) * tot
        acc = 0.0
        for name, pr in kept:
            acc += pr
            if r < acc:
                return name
        return kept[-1][0]
