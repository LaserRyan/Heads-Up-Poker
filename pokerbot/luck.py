"""Luck-adjusted hand results (the all-in adjustment).

A hand's chip result is mostly card luck.  The biggest single source is the
runout after both players are all-in: from that moment nobody makes another
decision, so whatever the remaining board cards do is pure noise as far as
judging OUR play goes.  Replacing the realised result of such a hand by its
expectation over every possible runout -- our exact equity times the pot,
minus what we put in -- is unbiased (its expectation equals the realised
result's) and strictly less noisy.  It is the zero-strategy-knowledge special
case of AIVAT (Burch et al. 2018), and what poker trackers call "all-in EV".

Used only for decisions about OUR strategy (the safety valve), never for
anything the judge scores.
"""
from __future__ import annotations

import random
from itertools import combinations

import numpy as np

from .cards import COLEX, PAIR_ID, HOLE_PAIRS, BOARD_TRIPLES, load_pt

_TRI = np.array(BOARD_TRIPLES, dtype=np.int64)          # (10, 3)
MAX_RUNOUTS = 300


def _pair_ids(hole) -> np.ndarray:
    return np.array([PAIR_ID[hole[i], hole[j]] for i, j in HOLE_PAIRS], dtype=np.int64)


def _triple_ids(boards: np.ndarray) -> np.ndarray:
    """(R, 5) boards -> (R, 10) colex triple ids."""
    b = np.sort(boards, axis=1)
    t = b[:, _TRI]                                         # (R, 10, 3), sorted within
    return COLEX[t[..., 0], 0] + COLEX[t[..., 1], 1] + COLEX[t[..., 2], 2]


def equity(my_hole, opp_hole, board_known, seed: int = 0) -> float:
    """Our exact (or, preflop, sampled) share of the pot over all runouts."""
    used = set(my_hole) | set(opp_hole) | set(board_known)
    rest = [c for c in range(36) if c not in used]
    k = 5 - len(board_known)
    if k <= 0:
        runs = [()]
    elif k <= 2:
        runs = list(combinations(rest, k))
    else:
        rng = random.Random(seed)
        runs = [tuple(rng.sample(rest, k)) for _ in range(MAX_RUNOUTS)]
    boards = np.array([tuple(board_known) + r for r in runs], dtype=np.int64)
    tids = _triple_ids(boards)                              # (R, 10)
    pt = load_pt()
    R = len(runs)
    flat = tids.ravel()
    mine = pt[np.ix_(_pair_ids(my_hole), flat)].max(axis=0).reshape(R, 10).max(axis=1)
    theirs = pt[np.ix_(_pair_ids(opp_hole), flat)].max(axis=0).reshape(R, 10).max(axis=1)
    return float(((mine > theirs).sum() + 0.5 * (mine == theirs).sum()) / R)


def adjusted_net(result: dict, my_hole, allin_board_len: int | None) -> float:
    """The hand's result with the post-all-in runout averaged out.

    allin_board_len: how many board cards were visible when the last chip
    went in (None if the hand was not all-in).  Anything that is not an
    all-in showdown before the river is returned unchanged.
    """
    net = float(result["my_net"])
    if allin_board_len is None or allin_board_len >= 5:
        return net
    if not result.get("went_to_showdown"):
        return net
    opp = result.get("opp_hole")
    board = result.get("board") or []
    if not opp or len(board) != 5 or not my_hole:
        return net
    pot = float(result["pot"])
    eq = equity(tuple(my_hole), tuple(opp), tuple(board[:allin_board_len]),
                seed=int(result.get("hand_index", 0)))
    return eq * pot - pot / 2.0      # all-in: both put in half the pot
