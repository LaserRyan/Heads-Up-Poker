"""
reads.py - what Tulip learns about its opponent during one half.

Three reads, each a recency-weighted average (recent hands count more, so a
style switch shows up within ~100-200 hands) shrunk toward a typical-bot
prior until enough evidence exists:

  fold_rate   how often they fold when we bet or raise after the flop
  pf_fold     how often they fold when we raise preflop
  honesty     how strong their hand turns out to be at showdown after they
              bet/raised postflop (percentile vs random hands on that board;
              ~0.5 = bets anything, ~0.85+ = bets only real hands)
  pf_raise    how often they raise when they first act preflop

`confidence` ramps from 0 at hand 50 to 1 at hand 200, and every adjustment
in strategy.py is scaled by it. All state lives in `memory`, so it resets
each half like the tournament requires.
"""

import numpy as np

from game import Action, Street
from tulip.fast_eval import omaha_batch

DECAY = 0.993             # per hand: weight of old evidence (~140-hand memory)
RAMP_START, RAMP_FULL = 50, 200

# (prior value, prior weight in pseudo-events)
PRIORS = {
    "fold_rate": (0.40, 10.0),
    "pf_fold": (0.35, 10.0),
    "honesty": (0.72, 6.0),
    "pf_raise": (0.30, 10.0),
}

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


def new(memory: dict) -> None:
    memory["reads"] = {k: [0.0, 0.0] for k in PRIORS}   # [weighted sum, weight]
    memory["hands_seen"] = 0
    memory["hand"] = None


def value(memory: dict, name: str) -> float:
    """Current estimate, shrunk toward the prior."""
    s, w = memory["reads"][name]
    prior, k = PRIORS[name]
    return (s + prior * k) / (w + k)


def confidence(memory: dict) -> float:
    n = memory.get("hands_seen", 0)
    return float(min(1.0, max(0.0, (n - RAMP_START) / (RAMP_FULL - RAMP_START))))


def _add(memory: dict, name: str, x: float, weight: float = 1.0) -> None:
    r = memory["reads"][name]
    r[0] += x * weight
    r[1] += weight


def _hand_state(obs, memory: dict) -> dict:
    h = memory.get("hand")
    if h is None or h["idx"] != obs.hand_index:
        h = {"idx": obs.hand_index, "hole": tuple(obs.hole_cards), "hist": (),
             "pending_bet": None}
        memory["hand"] = h
    return h


def observe(obs, memory: dict) -> None:
    """Call at the start of every decision."""
    h = _hand_state(obs, memory)
    if h["pending_bet"] is not None:
        # We bet/raised last time and we're being asked again in the same hand,
        # so they called or raised - they did not fold.
        _add(memory, h["pending_bet"], 0.0)
        h["pending_bet"] = None
    h["hist"] = _hist(obs)


def _hist(obs) -> tuple:
    """This hand's actions as (street, is_me, action) tuples."""
    return tuple((a.street, a.is_me, a.action) for a in obs.action_history)


def note_action(obs, action, memory: dict) -> None:
    """Call with the action we are about to return."""
    h = _hand_state(obs, memory)
    if action in RAISES:
        h["pending_bet"] = "pf_fold" if obs.street == Street.PREFLOP else "fold_rate"
    else:
        h["pending_bet"] = None
    h["hist"] = _hist(obs) + ((obs.street, True, action),)


def hand_end(result, memory: dict, rng: np.random.Generator) -> None:
    """Call from on_hand_end."""
    for r in memory["reads"].values():
        r[0] *= DECAY
        r[1] *= DECAY
    memory["hands_seen"] = memory.get("hands_seen", 0) + 1

    h = memory.get("hand")
    if h is None or h["idx"] != result.hand_index:
        return  # we never got to act this hand (opponent folded first): nothing to learn

    if h["pending_bet"] is not None:
        folded = (not result.went_to_showdown) and result.winner == "me"
        _add(memory, h["pending_bet"], 1.0 if folded else 0.0)

    opp_actions = [(st, act) for st, is_me, act in h["hist"] if not is_me]

    pre = [act for st, act in opp_actions if st == Street.PREFLOP]
    if pre:
        _add(memory, "pf_raise", 1.0 if pre[0] in RAISES else 0.0)

    post_units = sum((1.0 if act == Action.RAISE_POT else 0.6) * (1.5 if st >= Street.TURN else 1.0)
                     for st, act in opp_actions if st != Street.PREFLOP and act in RAISES)
    if post_units > 0 and result.went_to_showdown and result.opp_hole is not None \
            and len(result.board) == 5:
        _add(memory, "honesty", _showdown_percentile(result, h["hole"], rng),
             weight=min(2.0, post_units))


def _showdown_percentile(result, my_hole, rng, n: int = 64) -> float:
    """Where the opponent's final hand ranks among random hands on that board."""
    board = np.asarray(result.board, dtype=np.int64)
    opp = np.asarray(result.opp_hole, dtype=np.int64)
    known = np.zeros(36, dtype=bool)
    known[board] = True
    known[opp] = True
    known[np.asarray(my_hole, dtype=np.int64)] = True
    deck = np.flatnonzero(~known)
    others = deck[np.argsort(rng.random((n, len(deck))), axis=1)[:, :5]]
    scores = omaha_batch(np.vstack([opp[None, :], others]),
                         np.broadcast_to(board, (n + 1, 5)))
    return float(((scores[0] > scores[1:]) + 0.5 * (scores[0] == scores[1:])).mean())
