"""Hand strength at match time.

Everything here runs inside a ~0.25 ms budget per decision, so there is no room
for a real equity rollout.  Two tiers:

  tier CHEAP   `q`, the fraction of currently-available hole PAIRS whose best
               hand on this board is worse than ours, mapped through a
               per-street calibration curve fitted offline.  ~45 us.
  tier SAMPLE  a Monte-Carlo estimate against a precomputed pool of opponent
               hands, which is exact-in-expectation on the river.  ~180 us.

`q` is exact and board-specific (it is computed on the real board, not an
approximation of it), but it measures pairs while the opponent holds a hand
whose score is the MAX over ten pairs, so the map from q to equity is steep and
board dependent.  Measured correlation with true equity: 0.89 river, 0.70 turn,
0.65 flop; a one-dimensional calibration leaves RMSE ~0.10.  That is good
enough to order hands, which is all the tier-CHEAP path is asked to do.
"""
from __future__ import annotations

import os
from itertools import combinations

import numpy as np

from . import cards

PREFLOP, FLOP, TURN, RIVER = 0, 1, 2, 3

# How many distinct board triples exist for a board of each length.
_TRIPLE_SETS = {3: tuple(combinations(range(3), 3)),
                4: tuple(combinations(range(4), 3)),
                5: tuple(combinations(range(5), 3))}


class Strength:
    """Per-process evaluator. Holds the big tables and scratch buffers."""

    def __init__(self, pool_size: int = 224, seed: int = 12345):
        self.pt = cards.load_pt()
        self.pair_id = cards.PAIR_ID
        self.pair_mask = cards.PAIR_MASK
        rng = np.random.default_rng(seed)
        # Fixed pool of opponent hands, as positions into the list of unseen
        # cards.  Drawing positions instead of cards means the pool is built
        # once at startup and simply re-indexed on every board.
        self.pool = np.array([rng.choice(26, 5, replace=False) for _ in range(pool_size)],
                             dtype=np.intp)
        self._cal = None  # (4, 101) calibration, filled by load_calibration()
        self.pf_eq = None
        self.pf_pct = None
        path = os.path.join(cards.DATA_DIR, "preflop_eq.npy")
        if os.path.exists(path):
            self.pf_eq = np.load(path)
        path = os.path.join(cards.DATA_DIR, "preflop_pct.npy")
        if os.path.exists(path):
            self.pf_pct = np.load(path)
        path = os.path.join(cards.DATA_DIR, "calibration.npy")
        if os.path.exists(path):
            self._cal = np.load(path).tolist()  # list-of-lists: no numpy in the hot path

    # -- preflop ----------------------------------------------------------

    def preflop_equity(self, hole) -> float:
        """Exact equity vs a uniformly random hand, from the offline table.

        Indexed by the colex rank of the five sorted hole cards, so the cost is
        four adds and one array read.  Built from 282M duels pooled over the
        19,998 suit-isomorphic hand classes (mean stderr 0.004).
        """
        if self.pf_eq is None:
            return 0.5
        cx = cards.COLEX_L
        return float(self.pf_eq[cx[hole[0]][0] + cx[hole[1]][1] + cx[hole[2]][2]
                                + cx[hole[3]][3] + cx[hole[4]][4]]) / 65535.0

    def preflop_index(self, hole) -> int:
        cx = cards.COLEX_L
        return (cx[hole[0]][0] + cx[hole[1]][1] + cx[hole[2]][2]
                + cx[hole[3]][3] + cx[hole[4]][4])

    def preflop_pair(self, hole):
        """(equity vs random, percentile rank among all 376,992 hands).

        Equity is the right number for a pot-odds comparison; the percentile is
        the right number for an aggression threshold, because preflop equity in
        this variant is squeezed into roughly [0.10, 0.69] and a threshold
        stated in equity units is almost meaningless.
        """
        i = self.preflop_index(hole)
        if self.pf_eq is None:
            return 0.5, 0.5
        e = float(self.pf_eq[i]) / 65535.0
        p = float(self.pf_pct[i]) / 65535.0 if self.pf_pct is not None else e
        return e, p

    # -- board level -------------------------------------------------------

    def pairbest(self, board) -> np.ndarray:
        """(630,) best score every hole pair makes with this (3/4/5 card) board."""
        b = sorted(board)
        cx = cards.COLEX_L
        tids = [cx[b[i]][0] + cx[b[j]][1] + cx[b[k]][2] for i, j, k in _TRIPLE_SETS[len(b)]]
        return self.pt[:, tids].max(axis=1)

    def valid_pairs(self, board) -> np.ndarray:
        """Boolean (630,) mask of pairs the opponent could actually hold."""
        mask = 0
        for c in board:
            mask |= (1 << int(c))
        return (self.pair_mask & mask) == 0

    # -- hand level --------------------------------------------------------

    def made_score(self, hole, pb: np.ndarray) -> int:
        pid = cards.PAIR_ID_L
        best = 0
        for i, j in cards.HOLE_PAIRS:
            s = pb[pid[hole[i]][hole[j]]]
            if s > best:
                best = s
        return int(best)

    def q_from_pairbest(self, board, pb: np.ndarray, ours: int) -> float:
        """Cheapest strength: share of live opponent PAIRS we beat (~25 us).

        Used only when the time bank is running hot.  Pairs are not hands, so
        this over-rates us a little, but it orders hands correctly.
        """
        live = pb[self.valid_pairs(board)]
        return float(np.count_nonzero(live < ours)) / live.size

    def q_and_score(self, hole, board):
        """(q, our_score, pairbest). q is the fraction of live pairs we beat."""
        pb = self.pairbest(board)
        ours = self.made_score(hole, pb)
        live = pb[self.valid_pairs(board)]
        q = float(np.count_nonzero(live < ours)) / live.size
        return q, ours, pb

    def sample_equity(self, hole, board, pb: np.ndarray, ours: int, n: int = 0) -> float:
        """Share of a random opponent hand pool we beat on THIS board.

        On the river this is an unbiased estimate of equity.  On earlier
        streets it measures current strength only, with no credit for draws.
        """
        seen = set(hole) | set(board)
        avail = np.array([c for c in range(36) if c not in seen], dtype=np.intp)
        pool = self.pool if n <= 0 else self.pool[:n]
        opp = avail[pool]                                  # (K, 5) actual cards
        pid = self.pair_id
        best = pb[pid[opp[:, 0], opp[:, 1]]]
        for i, j in cards.HOLE_PAIRS[1:]:
            np.maximum(best, pb[pid[opp[:, i], opp[:, j]]], out=best)
        return (float(np.count_nonzero(best < ours))
                + 0.5 * float(np.count_nonzero(best == ours))) / len(pool)

    # -- calibration -------------------------------------------------------

    def load_calibration(self, table: np.ndarray) -> None:
        self._cal = table

    def calibrated(self, street: int, q: float) -> float:
        """Map current board strength to equity, accounting for cards to come."""
        cal = self._cal
        if cal is None:
            return q
        row = cal[street]
        i = int(q * (len(row) - 1))
        return row[i if i < len(row) else len(row) - 1]
