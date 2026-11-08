"""Decision logic.

Two different numbers describe a hand and they are not interchangeable:

  eq   our share of the pot at showdown, on an absolute 0..1 scale.  This is
       the number to compare against pot odds, and nothing else.
  pct  where the hand sits in the distribution of all hands in this spot.
       This is the number to put an aggression threshold on.

They have to be kept apart because equity in this variant is extremely
compressed.  Preflop equity spans roughly [0.10, 0.69] with a median of 0.50,
so "bet when equity > 0.62" means "bet the top 2% of hands" -- which is how the
first version of this file ended up almost never raising preflop.  Postflop the
squeeze runs the other way: the current nuts on the flop is worth only 0.88
equity, because two cards are still to come.

Sources of the two numbers:
  preflop   exact table, 282M simulated duels over the 19,998 suit-isomorphic
            hand classes (data/preflop_eq.npy, data/preflop_pct.npy)
  postflop  pct = share of a random opponent-hand pool we beat on the visible
            board; eq = that pushed through data/calibration.npy, fitted
            against Monte Carlo equity over real runouts.

The thresholds were fitted by cross-entropy search against a field of eleven
opponents (dev/tune.py), scored on the judge's own rule rather than on chips.
"""
from __future__ import annotations

import time

from .obs import CHECK_CALL, FLOP, FOLD, PREFLOP, RAISE_HALF, RAISE_POT, RIVER, TURN
from .strength import Strength

STARTING_STACK = 80


class Params:
    """Everything tunable, in one place, so the arena can search over it."""

    # --- aggression: thresholds on pct (hand's rank in the field) ---------
    bet_pct = [0.6467, 0.6489, 0.6979, 0.6338]    # open/lead when nobody has bet
    pot_pct = [0.4086, 0.6393, 0.846, 0.7905]     # prefer pot-sized over half-pot
    raise_pct = [0.5185, 0.7994, 0.7865, 0.8847]  # raise rather than call a bet

    # --- continuing: thresholds on eq (absolute, vs pot odds) ------------
    call_margin = [0.0979, 0.0891, 0.196, 0.0801]  # equity needed above the price
    give_up = [0.0, 0.2248, 0.4682, 0.3043]        # fold below this at any price

    # once this much of the stack is in, stop folding live hands
    committed_frac = 0.4851
    committed_eq = 0.4063

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(Params, k):
                raise AttributeError(f"unknown parameter {k!r}")
            setattr(self, k, v)

    def as_dict(self):
        return {k: getattr(self, k) for k in vars(Params)
                if not k.startswith("_") and not callable(getattr(Params, k))}


class Budget:
    """Keeps us inside the judge's time bank.

    The judge charges `wall_time - 1 ms` per request against 8 seconds a half.
    We cannot see its clock, only our own compute, so we assume a pessimistic
    fixed transport cost per call and project forward from the hands played so
    far.  If the projection runs hot we drop to a cheaper strength estimate
    rather than letting the bank empty -- an empty bank means check-or-fold for
    the rest of the half, which is far more expensive than a coarse decision.
    """

    __slots__ = ("total", "overhead", "safety", "spent", "calls", "num_hands", "tier")

    def __init__(self, total=8.0, overhead=0.0006, safety=0.78):
        self.total = total
        self.overhead = overhead
        self.safety = safety
        self.spent = 0.0
        self.calls = 0
        self.num_hands = 1000
        self.tier = 0

    def update(self, elapsed: float, hand_index: int) -> None:
        self.spent += elapsed
        self.calls += 1
        used = self.spent + self.calls * self.overhead
        done = hand_index + 1
        if done < 8:
            return
        per_call = used / self.calls
        projected = per_call * self.calls * (self.num_hands / done)
        limit = self.safety * self.total
        if projected > limit * 1.35:
            self.tier = 2
        elif projected > limit:
            self.tier = 1
        elif projected < limit * 0.6 and self.tier:
            self.tier -= 1


def _clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


class Policy:
    def __init__(self, strength=None, params=None, budget=None):
        self.s = strength or Strength()
        self.p = params or Params()
        self.budget = budget if budget is not None else Budget()

    # -- strength ----------------------------------------------------------

    def assess(self, obs):
        """(eq, pct) for the current spot."""
        street = obs.street
        if street == PREFLOP:
            return self.s.preflop_pair(obs.hole_cards)
        pb = self.s.pairbest(obs.board)
        ours = self.s.made_score(obs.hole_cards, pb)
        tier = self.budget.tier if self.budget else 0
        if tier >= 2:
            # Cheapest path: rank against live PAIRS instead of sampled hands.
            pct = self.s.q_from_pairbest(obs.board, pb, ours)
        else:
            pct = self.s.sample_equity(obs.hole_cards, obs.board, pb, ours,
                                       n=64 if tier else 0)
        return self.s.calibrated(street, pct), pct

    # -- the decision ------------------------------------------------------

    def act(self, obs, adj=None) -> str:
        """Decide.  Timing is charged by the caller (agent.choose_action), which
        sees the whole decision including opponent modelling; this class only
        reads the resulting budget tier."""
        try:
            return self._decide(obs, adj)
        except Exception:
            legal = obs.legal_actions
            return CHECK_CALL if CHECK_CALL in legal else FOLD

    def _decide(self, obs, adj) -> str:
        p = self.p
        legal = obs.legal_actions
        street = obs.street
        eq, pct = self.assess(obs)

        bet_t = p.bet_pct[street]
        pot_t = p.pot_pct[street]
        raise_t = p.raise_pct[street]
        margin = p.call_margin[street]
        give_up = p.give_up[street]
        value_only = False
        delay = False
        trap = False
        if adj is not None:
            bet_t = _clamp(bet_t + adj["bet_pct"])
            pot_t = _clamp(pot_t + adj["pot_pct"])
            raise_t = _clamp(raise_t + adj["raise_pct"])
            margin += adj["call_margin"]
            give_up = _clamp(give_up + adj["give_up"])
            value_only = adj["value_only"]
            delay = adj["delay_aggression"]
            trap = adj.get("trap", False)

        # Against someone who never folds there is no fold equity, so the only
        # reason to bet is value.  Published counter-strategies to never-fold
        # opponents also keep the pot small early and commit on the river,
        # which is worth substantially more than jamming from the flop.
        if delay and street != RIVER:
            pot_t = _clamp(pot_t + 0.10)

        def bet(big: bool) -> str:
            if big and RAISE_POT in legal:
                return RAISE_POT
            if RAISE_HALF in legal:
                return RAISE_HALF
            return RAISE_POT if RAISE_POT in legal else CHECK_CALL

        can_raise = RAISE_POT in legal or RAISE_HALF in legal

        # Nothing to call: bet the top of our range, check the rest.
        if obs.to_call == 0:
            # Against a bot that bets whenever it can, checking a strong hand
            # lets it put the money in for us -- and it never folds to a bet,
            # so there is nothing to win by betting first except on the river,
            # where it cannot bet after us.
            if trap and street != RIVER:
                return CHECK_CALL
            if can_raise and pct >= bet_t:
                return bet(pct >= pot_t)
            return CHECK_CALL

        # Facing a bet.
        price = obs.pot_odds()
        in_already = STARTING_STACK - obs.my_stack - obs.to_call

        if can_raise and pct >= raise_t and not (value_only and pct < bet_t):
            return bet(pct >= pot_t)

        # Already deep in the pot: a live hand calls rather than fold equity.
        if in_already >= p.committed_frac * STARTING_STACK and eq >= p.committed_eq:
            return CHECK_CALL

        if eq < give_up and price > 0.15:
            return FOLD if FOLD in legal else CHECK_CALL

        if eq >= price + margin:
            return CHECK_CALL
        return FOLD if FOLD in legal else CHECK_CALL
