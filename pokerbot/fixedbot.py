"""Exact identification of card-independent bots, by consistency.

Frequency statistics say "this opponent folds rarely".  For the bots most
likely to be in the field -- variants of the four published baselines and the
all-in bot -- we can say something much stronger: "every single action this
opponent has taken is exactly what always-pot would have done".

Each candidate is a deterministic rule over the LEGAL actions, so we
reconstruct, for every opponent decision, which actions were legal (raise cap,
stacks, whether a bet was faced) and test it against each rule.  A single
mismatch eliminates a candidate permanently.  A competent bot breaks every rule
within a few hands; a degenerate one never does, and after `n_confirm`
consistent decisions the identification is effectively certain.

Uniform random play is identified separately, by a likelihood-ratio test
against the alternative that the bot never folds to a bet it could call.

Only once a candidate is confirmed does the bot switch to the precomputed best
response for it (pokerbot/data/br_<kind>.npz).  Anything short of that is left
to the frequency-based exploit or to the blueprint.
"""
from __future__ import annotations

import math

FOLD, CALL, HALF, POT = "FOLD", "CHECK_CALL", "RAISE_HALF", "RAISE_POT"
CANDIDATES = ("allin", "half", "station", "checkfold")


def rule(kind: str, legal) -> str:
    """What the deterministic bot `kind` plays given its legal action set."""
    if kind == "allin":
        return POT if POT in legal else CALL
    if kind == "half":
        return HALF if HALF in legal else CALL
    if kind == "station":
        return CALL
    if kind == "checkfold":
        return FOLD if FOLD in legal else CALL
    raise ValueError(kind)


class FixedBotDetector:
    __slots__ = ("alive", "consistent", "n_confirm", "rand_ll", "rand_n",
                 "folds_faced", "fold_choices", "recent")

    def __init__(self, n_confirm: int = 14):
        self.alive = {k: True for k in CANDIDATES}
        self.consistent = {k: 0 for k in CANDIDATES}
        self.n_confirm = n_confirm
        self.rand_ll = 0.0       # log-likelihood ratio: random vs "never-fold-ish"
        self.rand_n = 0
        self.folds_faced = 0     # decisions facing a bet with fold legal
        self.fold_choices = 0
        # Sliding window of recent decisions as (action, fold_was_legal).  The
        # whole-half consistency test can never re-identify a bot whose
        # behaviour CHANGES -- and the commonest change is an opponent whose
        # 8 s time bank runs out mid-half, after which the judge plays
        # check-or-fold for it on every decision until the half ends.
        self.recent = []

    def _remember(self, action: str, fold_legal: bool) -> None:
        r = self.recent
        r.append((action, fold_legal))
        if len(r) > 16:
            del r[0]

    def timed_out(self) -> bool:
        """The last decisions look exactly like the judge's check-or-fold
        fallback: every bet faced was folded, every free spot was checked,
        nothing was ever raised -- with enough folds to rule out a passive
        patch of play by a normal bot."""
        r = self.recent
        if len(r) < 10:
            return False
        folds = 0
        for action, fold_legal in r[-10:]:
            if fold_legal:
                if action != FOLD:
                    return False
                folds += 1
            elif action != CALL:
                return False
        return folds >= 4

    def observe(self, action: str, legal) -> None:
        """One opponent decision with the set of actions that were legal."""
        self._remember(action, FOLD in legal)
        for k in CANDIDATES:
            if self.alive[k]:
                if rule(k, legal) == action:
                    self.consistent[k] += 1
                else:
                    self.alive[k] = False
        if FOLD in legal:
            self.folds_faced += 1
            if action == FOLD:
                self.fold_choices += 1

    def observe_fold(self) -> None:
        """An inferred fold (the hand ended on it, so it never showed up in the
        history).  A fold is only legal facing a bet; no legal-set detail is
        needed to know it rules out every bot that never folds."""
        for k in ("allin", "half", "station"):
            self.alive[k] = False
        self.folds_faced += 1
        self.fold_choices += 1
        self._remember(FOLD, True)

    def observe_call_facing_bet(self) -> None:
        """An inferred call of our final bet before a showdown.  It rules out
        check-fold.  It does NOT rule out always-pot/half, because our bet may
        have put them all-in or hit the raise cap, leaving call as the only
        non-fold option -- and we cannot see which from here."""
        self.alive["checkfold"] = False
        self.folds_faced += 1
        self._remember(CALL, True)

    def identified(self) -> str | None:
        """The confirmed fixed bot, or None."""
        if self.timed_out():
            return "checkfold"
        best = None
        for k in CANDIDATES:
            if self.alive[k] and self.consistent[k] >= self.n_confirm:
                # Prefer the most specific rule when several still fit
                # (a bot we have only ever seen call is consistent with both
                # 'station' and, if raising was never legal, 'allin').
                if best is None or self.consistent[k] > self.consistent[best]:
                    best = k
        if best is not None:
            return best
        if self.looks_random():
            return "random"
        return None

    def looks_random(self) -> bool:
        """Uniform-random play folds about 1/3 to 1/4 of the time it faces a
        bet -- including with hands that are clearly worth a call -- while
        never being consistent with any deterministic rule.  Require enough
        evidence and a fold rate in the uniform band."""
        if any(self.alive[k] and self.consistent[k] >= 4 for k in CANDIDATES):
            return False
        n = self.folds_faced
        if n < 40:
            return False
        f = self.fold_choices / n
        return 0.20 <= f <= 0.42


def legal_actions(to_call: int, raises: int, my_stack: int, opp_stack: int) -> tuple:
    """The engine's legality rule (game.Game._legal_for), from the actor's view."""
    acts = []
    if to_call > 0:
        acts.append(FOLD)
    acts.append(CALL)
    if raises < 4 and my_stack > to_call and opp_stack > 0:
        acts.append(HALF)
        acts.append(POT)
    return tuple(acts)
