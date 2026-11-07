"""
tulip_v1_strategy.py - FROZEN copy of Tulip v1 (commit f051f4b). Do not edit:

How a decision is made:
  1. Estimate EQUITY: how often our hand wins at showdown.
     - Preflop, unraised: look it up in a precomputed table (no simulation).
     - Otherwise: simulate random opponent hands and remaining board cards.
       If the opponent has bet or raised, their likely hands are weighted
       toward stronger ones (mildly) - a big bet usually means a real hand.
  2. Compare equity with the PRICE (pot odds) and fixed thresholds:
     strong -> bet/raise for value, decent and priced in -> call, else fold.
  3. Sometimes bet without a strong hand:
     - semi-bluffs: hands with real chances to improve, on flop/turn
     - rare pure bluffs on the river when checked to.

No opponent modelling across hands yet (that's a later step).
All tunable numbers live in PARAMS so the match runner can sweep them.
"""

import os
import time

import numpy as np

from game import Action, AgentConfig, HandResult, Observation, Street
from bot.fast_eval import colex, current_strength_batch, omaha_batch, scores_table

_HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot")
PREFLOP = np.load(os.path.join(_HERE, "data", "preflop_equity.npy")).astype(np.float32)
PREFLOP_SORTED = np.sort(PREFLOP)
scores_table()  # build the scoring table at import (server startup), not mid-hand

PARAMS = {
    # preflop, by percentile of starting hand (0 = worst, 1 = best)
    "pf_open_pot": 0.80,      # button's first action: raise pot
    "pf_open_half": 0.55,     # ...raise half
    "pf_open_call": 0.15,     # ...just call (complete the blind); below this fold
    "pf_bb_raise": 0.70,      # big blind after a button call: raise pot
    "pf_reraise_eq": 0.57,    # facing a raise: re-raise if equity vs their range >= this
    "pf_call_margin": 0.0,    # facing a raise: call if equity >= price + margin
    # postflop value thresholds (equity vs weighted range), per street
    "bet_half": {Street.FLOP: 0.60, Street.TURN: 0.62, Street.RIVER: 0.66},
    "bet_pot": {Street.FLOP: 0.72, Street.TURN: 0.75, Street.RIVER: 0.80},
    "raise_half": {Street.FLOP: 0.68, Street.TURN: 0.71, Street.RIVER: 0.76},
    "raise_pot": {Street.FLOP: 0.78, Street.TURN: 0.82, Street.RIVER: 0.88},
    "call_margin": {Street.FLOP: 0.04, Street.TURN: 0.03, Street.RIVER: 0.0},
    "raise_war_eq": 0.85,     # after 2+ raises this street, only re-raise with this much
    # bluffs
    "semi_lo": 0.42, "semi_hi": 0.60, "semi_freq": 0.30,
    "bluff_eq_max": 0.25, "bluff_freq": 0.07,
    # how strongly opponent aggression shifts their range (Q10: mild)
    "alpha_per_unit": 0.5, "alpha_cap": 1.5,
    # simulation size and time budget
    "samples": 128, "samples_mid": 64, "samples_low": 24,
    "budget_fraction": 0.50,  # aim to finish the half having used this share of the bank
    "emergency_fraction": 0.75,  # past this share: stop simulating entirely
    # The judge also charges web-server time we can't see (~0.2-1 ms per call
    # on a laptop). Add this much per call to our own clock to stay honest.
    "overhead_per_call": 0.0008,
}


# ----- helpers ----------------------------------------------------------------

def _pf_percentile(hole) -> tuple[float, float]:
    """(equity vs random, percentile among all starting hands)."""
    idx = int(colex(np.sort(np.asarray(hole, dtype=np.int64))[None, :])[0])
    eq = float(PREFLOP[idx])
    return eq, float(np.searchsorted(PREFLOP_SORTED, eq)) / len(PREFLOP_SORTED)


def _aggression_units(obs: Observation, street: Street) -> float:
    """How hard the opponent has pushed. Pot-size raise = 1, half = 0.6.
    Earlier streets count half."""
    units = 0.0
    for a in obs.action_history:
        if a.is_me or a.action not in (Action.RAISE_HALF, Action.RAISE_POT):
            continue
        u = 1.0 if a.action == Action.RAISE_POT else 0.6
        units += u if a.street == street else 0.5 * u
    return units


def _equity(obs: Observation, n: int, memory: dict,
            alpha_pf: float, alpha_now: float) -> float:
    """Simulated equity vs an opponent range weighted by pct**alpha, where pct
    is the opponent hand's strength percentile (preflop table and/or made hand
    on the current board).

    The simulation is cached per (hand, street): when the opponent re-raises
    we only redo the cheap reweighting, not the simulation."""
    key = (obs.hand_index, int(obs.street))
    cache = memory.get("sim")
    if cache is None or cache[0] != key:
        cache = (key,) + _simulate(obs, n, memory["rng"])
        memory["sim"] = cache
    _, res, pf_pct, now_pct = cache
    w = np.ones(len(res))
    if alpha_pf > 0:
        w *= pf_pct ** alpha_pf
    if alpha_now > 0 and now_pct is not None:
        w *= now_pct ** alpha_now
    return float((res * w).sum() / w.sum())


def _simulate(obs: Observation, n: int, rng: np.random.Generator):
    """Deal n random opponent hands + rest of board. Returns per sample:
    result (1 win, .5 tie, 0 loss), opponent preflop percentile, opponent
    made-hand percentile on the current board (None preflop)."""
    hole = np.asarray(obs.hole_cards, dtype=np.int64)
    board = np.asarray(obs.board, dtype=np.int64)
    known = np.zeros(36, dtype=bool)
    known[hole] = True
    known[board] = True
    deck = np.flatnonzero(~known)
    need_board = 5 - len(board)

    draws = np.argsort(rng.random((n, len(deck))), axis=1)[:, :5 + need_board]
    drawn = deck[draws]
    opp = drawn[:, :5]
    full_board = np.concatenate([np.broadcast_to(board, (n, len(board))), drawn[:, 5:]], axis=1)

    if need_board == 0:  # river: our score is the same in every sample
        me = omaha_batch(hole[None, :], board[None, :])[0]
    else:
        me = omaha_batch(np.broadcast_to(hole, (n, 5)), full_board)
    op = omaha_batch(opp, full_board)
    res = (me > op) + 0.5 * (me == op)

    pf_pct = _pct(PREFLOP[colex(np.sort(opp, axis=1))])
    now_pct = None
    if len(board) == 5:
        now_pct = _pct(op)
    elif len(board) >= 3:
        now_pct = _pct(current_strength_batch(opp, board))
    return res, pf_pct, now_pct


def _pct(values: np.ndarray) -> np.ndarray:
    """Rank of each value within the sample, mapped to (0, 1]."""
    order = values.argsort(kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks / len(values)


def _pick(obs: Observation, *prefs: Action) -> Action:
    """First of prefs that is legal; CHECK_CALL is always legal."""
    for a in prefs:
        if a in obs.legal_actions:
            return a
    return Action.CHECK_CALL


# ----- the three hooks --------------------------------------------------------

def new_half(config: AgentConfig, memory: dict) -> None:
    memory["rng"] = np.random.default_rng(20261107)
    memory["used"] = 0.0
    memory["calls"] = 0
    memory["bank"] = float(config.time_bank_seconds)
    memory["num_hands"] = int(config.num_hands)


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass


def choose_action(obs: Observation, memory: dict) -> Action:
    t0 = time.perf_counter()
    try:
        return _decide(obs, memory)
    finally:
        memory["used"] = memory.get("used", 0.0) + time.perf_counter() - t0
        memory["calls"] = memory.get("calls", 0) + 1


def estimated_judge_time(memory: dict) -> float:
    """Our best guess of what the judge has charged us so far this half."""
    return memory.get("used", 0.0) + memory.get("calls", 0) * PARAMS["overhead_per_call"]


def _sample_budget(obs: Observation, memory: dict) -> int:
    """Step the simulation size down smoothly when spending faster than
    planned; stop simulating only near a real emergency."""
    P = PARAMS
    bank = memory.get("bank", 8.0)
    est = estimated_judge_time(memory)
    progress = (obs.hand_index + 1) / max(1, memory.get("num_hands", 1000))
    planned = bank * P["budget_fraction"] * progress
    if est > bank * P["emergency_fraction"]:
        return 0
    if est > planned + 0.3:
        return P["samples_low"]
    if est > planned:
        return P["samples_mid"]
    return P["samples"]


def _decide(obs: Observation, memory: dict) -> Action:
    P = PARAMS
    rng = memory["rng"]
    street = obs.street
    opp_units = _aggression_units(obs, street)
    opp_pf_units = sum((1.0 if a.action == Action.RAISE_POT else 0.6)
                       for a in obs.action_history
                       if not a.is_me and a.street == Street.PREFLOP
                       and a.action in (Action.RAISE_HALF, Action.RAISE_POT))
    alpha_pf = min(P["alpha_cap"], P["alpha_per_unit"] * opp_pf_units)
    alpha_now = min(P["alpha_cap"], P["alpha_per_unit"] * opp_units)
    price = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    n = _sample_budget(obs, memory)

    # ----- preflop -----
    if street == Street.PREFLOP:
        eq, pct = _pf_percentile(obs.hole_cards)
        opp_raised = opp_pf_units > 0
        if not opp_raised:
            if obs.to_call > 0:  # button, first decision
                if pct >= P["pf_open_pot"]:
                    return _pick(obs, Action.RAISE_POT)
                if pct >= P["pf_open_half"]:
                    return _pick(obs, Action.RAISE_HALF)
                if pct >= P["pf_open_call"]:
                    return Action.CHECK_CALL
                return _pick(obs, Action.FOLD)
            # big blind, button just called
            if pct >= P["pf_bb_raise"]:
                return _pick(obs, Action.RAISE_POT)
            return Action.CHECK_CALL
        # facing a raise
        e = _equity(obs, n, memory, alpha_pf, 0.0) if n else eq
        if e >= P["pf_reraise_eq"] and obs.raises_this_street < 3:
            return _pick(obs, Action.RAISE_POT)
        if e >= price + P["pf_call_margin"]:
            return Action.CHECK_CALL
        return _pick(obs, Action.FOLD)

    # ----- emergency (time bank nearly gone): no simulation -----
    # Keep playing priced-in hands instead of folding to every bet.
    if n == 0:
        if obs.to_call == 0 or price <= 0.25:
            return Action.CHECK_CALL
        return _pick(obs, Action.FOLD)

    e = _equity(obs, n, memory, alpha_pf, alpha_now)

    # ----- nobody has bet: bet or check -----
    if obs.to_call == 0:
        if e >= P["bet_pot"][street]:
            return _pick(obs, Action.RAISE_POT, Action.RAISE_HALF)
        if e >= P["bet_half"][street]:
            return _pick(obs, Action.RAISE_HALF)
        if street != Street.RIVER and P["semi_lo"] <= e < P["semi_hi"] \
                and rng.random() < P["semi_freq"]:
            return _pick(obs, Action.RAISE_HALF)
        if street == Street.RIVER and e <= P["bluff_eq_max"] and rng.random() < P["bluff_freq"]:
            return _pick(obs, Action.RAISE_HALF)
        return Action.CHECK_CALL

    # ----- facing a bet -----
    war = obs.raises_this_street >= 2
    if not war or e >= P["raise_war_eq"]:
        if e >= P["raise_pot"][street]:
            return _pick(obs, Action.RAISE_POT, Action.RAISE_HALF)
        if e >= P["raise_half"][street]:
            return _pick(obs, Action.RAISE_HALF)
    if e >= price + P["call_margin"][street]:
        return Action.CHECK_CALL
    return _pick(obs, Action.FOLD)
