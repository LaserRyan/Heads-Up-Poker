"""
fast_eval.py - vectorised (numpy) Omaha scoring for many hands at once.

The stock evaluator.omaha_score() scores one hand in ~25 us of pure Python.
Simulations need hundreds of scores per decision, so here every 5-card hand
gets a dense index (its "colex" rank among all C(36,5) hands) and the scores
live in one numpy array. Scoring N hands is then a handful of array ops.

Results are identical to evaluator.omaha_score (checked in tests).
"""

from itertools import combinations
from math import comb

import numpy as np

from evaluator import NUM_CARDS, score_five

N5 = comb(NUM_CARDS, 5)  # 376,992 five-card hands

# BINOM[c, k] = C(c, k); colex index of sorted cards c0<c1<..<c4 is
# sum_k C(c_k, k+1).
BINOM = np.array([[comb(c, k) for k in range(6)] for c in range(NUM_CARDS)], dtype=np.int64)
_COLEX_W = [BINOM[:, k + 1] for k in range(5)]

PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.int64)    # (10, 2)
TRIPLES = np.array(list(combinations(range(5), 3)), dtype=np.int64)  # (10, 3)
SUBSETS3_OF_4 = np.array(list(combinations(range(4), 3)), dtype=np.int64)  # (4, 3)

_SCORES: np.ndarray | None = None


def colex(sorted5: np.ndarray) -> np.ndarray:
    """Dense index of each row of an (..., 5) array of ascending cards."""
    s = sorted5
    return (_COLEX_W[0][s[..., 0]] + _COLEX_W[1][s[..., 1]] + _COLEX_W[2][s[..., 2]]
            + _COLEX_W[3][s[..., 3]] + _COLEX_W[4][s[..., 4]])


def scores_table() -> np.ndarray:
    """score_five for every 5-card hand, indexed by colex. Built once (~1-2 s)."""
    global _SCORES
    if _SCORES is None:
        table = np.empty(N5, dtype=np.int32)
        for combo in combinations(range(NUM_CARDS), 5):
            idx = sum(comb(c, k + 1) for k, c in enumerate(combo))
            table[idx] = score_five(combo)
        _SCORES = table
    return _SCORES


def _best(holes: np.ndarray, boards3: np.ndarray) -> np.ndarray:
    """holes: (N, 5) cards. boards3: (N, T, 3) board triples.
    Best score over 10 hole pairs x T triples -> (N,)."""
    n = holes.shape[0]
    hp = holes[:, PAIRS]                                   # (N, 10, 2)
    t = boards3.shape[1]
    hp = np.broadcast_to(hp[:, :, None, :], (n, 10, t, 2))
    bt = np.broadcast_to(boards3[:, None, :, :], (n, 10, t, 3))
    five = np.concatenate([hp, bt], axis=3)               # (N, 10, T, 5)
    five = np.sort(five, axis=3)
    return scores_table()[colex(five)].reshape(n, -1).max(axis=1)


def omaha_batch(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Omaha scores for N hands. holes (N,5), boards (N,5) -> (N,) int."""
    return _best(holes, boards[:, TRIPLES])


def current_strength_batch(holes: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Best made hand so far on a 3- or 4-card board (same for every row).
    holes (N,5), board (3,) or (4,) -> (N,)."""
    board = np.asarray(board, dtype=np.int64)
    if len(board) == 3:
        triples = board[None, None, :]
    else:
        triples = board[SUBSETS3_OF_4][None, :, :]
    triples = np.broadcast_to(triples, (holes.shape[0],) + triples.shape[1:])
    return _best(holes, triples)
