"""
_styles.py - shared brain for the practice bots.

Every practice bot estimates hand strength the same realistic way (equity vs
a random hand, by simulation) and then plays it through a STYLE: how many
hands it plays, how often it bets/raises, how easily it folds, how often it
bluffs. That makes them stand-ins for the kinds of bots other teams submit,
rather than trivially beatable rule bots.

A style is a dict of the numbers below; see the practice-bots/*.py files.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.fast_eval import colex, omaha_batch  # noqa: E402
from game import Action, Street  # noqa: E402

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "bot", "data", "preflop_equity.npy")
PREFLOP = np.load(_DATA).astype(np.float32)
PREFLOP_SORTED = np.sort(PREFLOP)

DEFAULTS = {
    "pf_play": 0.20,       # preflop: fold hands below this percentile when facing a raise/open
    "pf_raise": 0.75,      # preflop: raise with hands above this percentile
    "pf_call_raise": 0.45, # preflop: call a raise with hands above this percentile
    "bet": 0.62,           # postflop: bet (when checked to) with equity above this
    "bet_pot": 0.78,       # ...pot-size bet above this
    "raise_eq": 0.75,        # postflop: raise a bet with equity above this
    "call_margin": 0.03,   # postflop: call if equity >= price + margin
    "bluff": 0.0,          # chance to bet/raise a weak hand when it could
    "bluff_raise": 0.0,    # chance to raise a bet with a weak hand
    "never_fold_below": 0.0,  # never fold when the price is below this (sticky)
    "samples": 64,
}


def _pct(hole) -> float:
    idx = int(colex(np.sort(np.asarray(hole, dtype=np.int64))[None, :])[0])
    return float(np.searchsorted(PREFLOP_SORTED, PREFLOP[idx])) / len(PREFLOP_SORTED)


def _equity(obs, n, rng) -> float:
    hole = np.asarray(obs.hole_cards, dtype=np.int64)
    board = np.asarray(obs.board, dtype=np.int64)
    known = np.zeros(36, dtype=bool)
    known[hole] = True
    known[board] = True
    deck = np.flatnonzero(~known)
    need = 5 - len(board)
    drawn = deck[np.argsort(rng.random((n, len(deck))), axis=1)[:, :5 + need]]
    full = np.concatenate([np.broadcast_to(board, (n, len(board))), drawn[:, 5:]], axis=1)
    me = omaha_batch(np.broadcast_to(hole, (n, 5)), full)
    op = omaha_batch(drawn[:, :5], full)
    return float(((me > op) + 0.5 * (me == op)).mean())


def _pick(obs, *prefs):
    for a in prefs:
        if a in obs.legal_actions:
            return a
    return Action.CHECK_CALL


def make_bot(style: dict, seed: int):
    """Return (new_half, choose_action, on_hand_end) for a style."""
    S = {**DEFAULTS, **style}

    def new_half(config, memory):
        memory["rng"] = np.random.default_rng(seed)

    def on_hand_end(result, memory):
        pass

    def choose_action(obs, memory):
        rng = memory["rng"]
        price = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
        if obs.street == Street.PREFLOP:
            p = _pct(obs.hole_cards)
            opp_raised = any(not a.is_me and a.action in (Action.RAISE_HALF, Action.RAISE_POT)
                             for a in obs.action_history)
            if p >= S["pf_raise"] or rng.random() < S["bluff"] * 0.5:
                if obs.raises_this_street < 3:
                    return _pick(obs, Action.RAISE_POT, Action.RAISE_HALF)
            if obs.to_call == 0:
                return Action.CHECK_CALL
            need = S["pf_call_raise"] if opp_raised else S["pf_play"]
            if p >= need or price < S["never_fold_below"]:
                return Action.CHECK_CALL
            return _pick(obs, Action.FOLD)

        e = _equity(obs, S["samples"], rng)
        if obs.to_call == 0:
            if e >= S["bet_pot"]:
                return _pick(obs, Action.RAISE_POT, Action.RAISE_HALF)
            if e >= S["bet"] or rng.random() < S["bluff"]:
                return _pick(obs, Action.RAISE_HALF)
            return Action.CHECK_CALL
        if e >= S["raise_eq"] and obs.raises_this_street < 3:
            return _pick(obs, Action.RAISE_POT, Action.RAISE_HALF)
        if e < 0.5 and obs.raises_this_street < 2 and rng.random() < S["bluff_raise"]:
            return _pick(obs, Action.RAISE_POT, Action.RAISE_HALF)
        if e >= price + S["call_margin"] or price < S["never_fold_below"]:
            return Action.CHECK_CALL
        return _pick(obs, Action.FOLD)

    return new_half, choose_action, on_hand_end


STYLES = {
    # Plays few hands, rarely raises, folds to pressure, never bluffs.
    "nit": dict(pf_play=0.55, pf_raise=0.92, pf_call_raise=0.75, bet=0.72, bet_pot=0.88,
                raise_eq=0.85, call_margin=0.12),
    # Plays most hands, raises a lot, bets medium hands, some bluffs.
    "lag": dict(pf_play=0.10, pf_raise=0.45, pf_call_raise=0.25, bet=0.52, bet_pot=0.66,
                raise_eq=0.64, call_margin=-0.03, bluff=0.15, bluff_raise=0.06),
    # Bets and raises constantly, mostly with air - but honest with strong hands too.
    "maniac": dict(pf_play=0.0, pf_raise=0.40, pf_call_raise=0.10, bet=0.60, bet_pot=0.70,
                   raise_eq=0.70, call_margin=-0.05, bluff=0.45, bluff_raise=0.25),
    # Tulip-like equity bot with no bluffs: its bets always mean strength.
    "honest": dict(pf_play=0.20, pf_raise=0.75, pf_call_raise=0.45, bet=0.62, bet_pot=0.76,
                   raise_eq=0.72, call_margin=0.03),
    # Calls almost anything, raises only monsters, rarely bets.
    "sticky": dict(pf_play=0.0, pf_raise=0.95, pf_call_raise=0.0, bet=0.80, bet_pot=0.90,
                   raise_eq=0.88, call_margin=-0.30, never_fold_below=0.40),
}
