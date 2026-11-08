"""The CFR blueprint: card abstraction + strategy lookup.

Abstraction (`bucket`)
----------------------
One function, used both to build the training deal bank and to play, so the two
can never disagree.

  preflop  48 buckets by the hand's exact percentile among all 376,992 hands.
  flop     32 strength buckets x 3 flush-draw classes = 96.
  turn     32 strength buckets x 3 flush-draw classes = 96.
  river    64 strength buckets.

Strength is `sample_equity`: the share of a fixed random pool of opponent hands
we beat on the visible board.  It is equal-population by construction, so equal
width bins are equal-mass bins.

Why a flush-draw dimension and not a straight-draw one: in this variant trips
beat a straight and 76% of hands reach the river with a straight or better, so
straight draws are close to worthless (a big wrap is ~7% against a set), while a
flush is rarer than a full house and beats it -- a made nut flush is ~91%
against a set on the flop.  The only draw worth a separate bucket is the
flush draw, and whether it draws to the NUT flush matters as much as whether it
exists at all, because the second-best flush is a stack-losing hand.

Strategy lookup (`Blueprint`)
-----------------------------
STRAT[decision_node, bucket, action] is the average CFR strategy.  At match time
the observed action history is walked through the betting tree (built from the
real engine, so the walk is exact) to find the node, and the row is read off.
"""
from __future__ import annotations

import os

import numpy as np

from . import tree as T

N_PRE, N_STR, N_DRAW, N_RIVER = 48, 32, 3, 64
N_BUCKETS = max(N_PRE, N_STR * N_DRAW, N_RIVER)  # 96
BUCKETS_PER_STREET = (N_PRE, N_STR * N_DRAW, N_STR * N_DRAW, N_RIVER)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "blueprint.npz")


def flush_draw_class(hole, board) -> int:
    """0 none, 1 non-nut flush draw, 2 nut flush draw (flop/turn only).

    A draw means: the board shows exactly two of a suit and we hold at least
    two of it (we must use exactly two hole cards, so one suited card in hand
    is never a draw).  It is the NUT draw if no higher card of that suit is
    still unseen -- i.e. we hold the best card of the suit not on the board.
    """
    bcount = [0, 0, 0, 0]
    for c in board:
        bcount[c & 3] += 1
    best = 0
    for s in range(4):
        if bcount[s] != 2:
            continue
        mine = [c >> 2 for c in hole if (c & 3) == s]
        if len(mine) < 2:
            continue
        on_board = {c >> 2 for c in board if (c & 3) == s}
        top_missing = 8
        while top_missing in on_board:
            top_missing -= 1
        cls = 2 if max(mine) >= top_missing else 1
        if cls > best:
            best = cls
    return best


def bucket(S, hole, board, street: int) -> int:
    """The abstraction bucket for this hand on this street."""
    if street == 0:
        _, pct = S.preflop_pair(hole)
        b = int(pct * N_PRE)
        return b if b < N_PRE else N_PRE - 1
    pb = S.pairbest(board)
    ours = S.made_score(hole, pb)
    pct = S.sample_equity(hole, board, pb, ours)
    if street == 3:
        b = int(pct * N_RIVER)
        return b if b < N_RIVER else N_RIVER - 1
    b = int(pct * N_STR)
    if b >= N_STR:
        b = N_STR - 1
    return flush_draw_class(hole, board) * N_STR + b


def bucket_from_pct(pct: float, hole, board, street: int) -> int:
    """Same as bucket() when the caller has already computed pct."""
    if street == 3:
        b = int(pct * N_RIVER)
        return b if b < N_RIVER else N_RIVER - 1
    b = int(pct * N_STR)
    if b >= N_STR:
        b = N_STR - 1
    return flush_draw_class(hole, board) * N_STR + b


class Blueprint:
    """Average-strategy table plus the tree needed to find our node."""

    def __init__(self, path: str = DATA):
        self.ok = False
        self.tree = T.load()
        if not os.path.exists(path):
            return
        z = np.load(path)
        self.strat = z["strat"]               # (n_decision_nodes, N_BUCKETS, 4) uint8
        self.dnode = z["dnode"]               # tree node -> decision index, -1 otherwise
        self.child = self.tree["child"]
        self.actor = self.tree["actor"]
        self.ok = True

    def node(self, history) -> int:
        child = self.child
        n = 0
        idx = T.ACTION_INDEX
        for rec in history:
            a = idx.get(rec["action"])
            if a is None:
                return -1
            n = int(child[n, a])
            if n < 0:
                return -1
        return n

    def probs(self, node: int, b: int):
        """Strategy row as four floats summing to 1, or None if unavailable."""
        d = int(self.dnode[node])
        if d < 0:
            return None
        row = self.strat[d, b]
        tot = int(row[0]) + int(row[1]) + int(row[2]) + int(row[3])
        if tot <= 0:
            return None
        return (row[0] / tot, row[1] / tot, row[2] / tot, row[3] / tot)
