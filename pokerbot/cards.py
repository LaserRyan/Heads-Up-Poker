"""Fast short-deck (36-card) Omaha hand evaluation.

Card encoding matches evaluator.py: rank = c >> 2 (0='6' .. 8='A'), suit = c & 3.
Short-deck ordering: FLUSH beats FULL HOUSE, TRIPS beat a STRAIGHT, and the
wheel A-6-7-8-9 is the lowest straight.

The central idea
----------------
An Omaha hand is always exactly 2 hole cards + 3 board cards.  There are only
C(36,2) = 630 hole pairs and C(36,3) = 7140 board triples, so every possible
"2+3" hand fits in one 630 x 7140 uint32 table (18 MB):

    PT[pair_id, triple_id] = score of those five cards   (0 if cards overlap)

With it, scoring is pure array indexing:
  * one hand on one board -> PT[np.ix_(pair_ids, triple_ids)].max()
  * N hands on one board  -> pairbest = PT[:, tids].max(1), then
                             pairbest[hand_pairs].max(1)

The second form is what makes equity enumeration cheap: the per-board table is
built once and reused for every candidate opponent hand.
"""
from __future__ import annotations

import os
from itertools import combinations
from math import comb

import numpy as np

NUM_RANKS, NUM_SUITS, NUM_CARDS = 9, 4, 36
NUM_FIVE = comb(36, 5)      # 376992
NUM_PAIRS = comb(36, 2)     # 630
NUM_TRIPLES = comb(36, 3)   # 7140

HIGH_CARD, ONE_PAIR, TWO_PAIR, STRAIGHT = 0, 1, 2, 3
THREE_OF_A_KIND, FULL_HOUSE, FLUSH, FOUR_OF_A_KIND, STRAIGHT_FLUSH = 4, 5, 6, 7, 8
CATEGORY_SHIFT = 20
ACE, NINE = 8, 3

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------------------
# colex ranking: index of a sorted k-subset = sum_i C(card_i, i+1)
# ---------------------------------------------------------------------------
COLEX = np.array([[comb(c, k + 1) for k in range(5)] for c in range(NUM_CARDS)], dtype=np.int64)
COLEX_L = COLEX.tolist()

# PAIRS / TRIPLES are stored in COLEX order, so the array index IS the colex id.
# (itertools.combinations yields lexicographic order, which is a different
# permutation -- mixing the two silently corrupts every lookup.)
PAIRS = np.zeros((NUM_PAIRS, 2), dtype=np.int64)
for _a, _b in combinations(range(NUM_CARDS), 2):
    PAIRS[COLEX[_a, 0] + COLEX[_b, 1]] = (_a, _b)
TRIPLES = np.zeros((NUM_TRIPLES, 3), dtype=np.int64)
for _a, _b, _c in combinations(range(NUM_CARDS), 3):
    TRIPLES[COLEX[_a, 0] + COLEX[_b, 1] + COLEX[_c, 2]] = (_a, _b, _c)

PAIR_ID = np.full((NUM_CARDS, NUM_CARDS), -1, dtype=np.int32)
for _i in range(NUM_PAIRS):
    _a, _b = int(PAIRS[_i, 0]), int(PAIRS[_i, 1])
    PAIR_ID[_a, _b] = _i
    PAIR_ID[_b, _a] = _i
PAIR_ID_L = PAIR_ID.tolist()

PAIR_MASK = np.array([(1 << int(a)) | (1 << int(b)) for a, b in PAIRS], dtype=np.int64)
TRIPLE_MASK = np.array([(1 << int(a)) | (1 << int(b)) | (1 << int(c)) for a, b, c in TRIPLES],
                       dtype=np.int64)

# the 10 ways to pick 2 of 5 hole cards / 3 of 5 board cards
HOLE_PAIRS = tuple(combinations(range(5), 2))
BOARD_TRIPLES = tuple(combinations(range(5), 3))


def triple_id(a: int, b: int, c: int) -> int:
    """colex id of three cards given in any order."""
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return COLEX_L[a][0] + COLEX_L[b][1] + COLEX_L[c][2]


def board_triple_ids(board) -> list:
    """The 10 triple ids of a 5-card board."""
    b = sorted(board)
    cx = COLEX_L
    return [cx[b[i]][0] + cx[b[j]][1] + cx[b[k]][2] for i, j, k in BOARD_TRIPLES]


def hand_pair_ids(hole) -> list:
    """The 10 pair ids of a 5-card hand."""
    p = PAIR_ID_L
    return [p[hole[i]][hole[j]] for i, j in HOLE_PAIRS]


def hand_pair_ids_batch(holes: np.ndarray) -> np.ndarray:
    """(N,5) hole cards -> (N,10) pair ids."""
    h = np.asarray(holes, dtype=np.int64)
    return np.stack([PAIR_ID[h[:, i], h[:, j]] for i, j in HOLE_PAIRS], axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# 5-card scorer (identical ordering to evaluator._score_ranks)
# ---------------------------------------------------------------------------

def _pack(category: int, ranks) -> int:
    score = category << CATEGORY_SHIFT
    for slot, rank in enumerate(ranks):
        score |= rank << (16 - 4 * slot)
    return score


def score_five_ranks(ranks_desc, is_flush: bool) -> int:
    from collections import Counter
    counts = Counter(ranks_desc)
    if len(counts) == 5:
        if ranks_desc[0] - ranks_desc[4] == 4:
            top = ranks_desc[0]
        elif tuple(ranks_desc) == (ACE, NINE, 2, 1, 0):
            top = NINE
        else:
            top = None
        if top is not None:
            return _pack(STRAIGHT_FLUSH if is_flush else STRAIGHT, [top])
        return _pack(FLUSH if is_flush else HIGH_CARD, ranks_desc)
    groups = sorted(counts.items(), key=lambda rc: (rc[1], rc[0]), reverse=True)
    shape = tuple(c for _, c in groups)
    category = {(4, 1): FOUR_OF_A_KIND, (3, 2): FULL_HOUSE,
                (3, 1, 1): THREE_OF_A_KIND, (2, 2, 1): TWO_PAIR}.get(shape, ONE_PAIR)
    return _pack(category, [r for r, _ in groups])


def build_five_score() -> np.ndarray:
    """uint32 score of every 5-card hand, indexed by colex rank."""
    table = np.zeros(NUM_FIVE, dtype=np.uint32)
    cache = {}
    cx = COLEX_L
    for a, b, c, d, e in combinations(range(NUM_CARDS), 5):
        key = ((e >> 2, d >> 2, c >> 2, b >> 2, a >> 2),
               (a & 3) == (b & 3) == (c & 3) == (d & 3) == (e & 3))
        sc = cache.get(key)
        if sc is None:
            sc = score_five_ranks(key[0], key[1])
            cache[key] = sc
        table[cx[a][0] + cx[b][1] + cx[c][2] + cx[d][3] + cx[e][4]] = sc
    return table


def build_pt(five=None) -> np.ndarray:
    """The 630 x 7140 pair/triple score table. 0 where cards overlap."""
    if five is None:
        five = build_five_score()
    pt = np.zeros((NUM_PAIRS, NUM_TRIPLES), dtype=np.uint32)
    combo = np.empty((NUM_PAIRS, 5), dtype=np.int64)
    combo[:, :2] = PAIRS
    for t in range(NUM_TRIPLES):
        combo[:, 2:] = TRIPLES[t]
        # Overlapping cards make the colex index meaningless (and out of range),
        # so compute it only for the disjoint rows.
        ok = (PAIR_MASK & int(TRIPLE_MASK[t])) == 0
        s = np.sort(combo[ok], axis=1)
        idx = (COLEX[s[:, 0], 0] + COLEX[s[:, 1], 1] + COLEX[s[:, 2], 2]
               + COLEX[s[:, 3], 3] + COLEX[s[:, 4], 4])
        pt[ok, t] = five[idx]
    return pt


PT = None
_PT_PATH = os.path.join(DATA_DIR, "pt.npy")


def load_pt() -> np.ndarray:
    """Load (or build and cache) the pair/triple table."""
    global PT
    if PT is None:
        if os.path.exists(_PT_PATH):
            PT = np.load(_PT_PATH)
        else:
            PT = build_pt()
            os.makedirs(DATA_DIR, exist_ok=True)
            try:
                np.save(_PT_PATH, PT)
            except OSError:
                pass  # read-only filesystem at match time
    return PT


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def omaha_score_fast(hole, board) -> int:
    """Match-time scalar path. Needs 5 hole cards and a 5-card board."""
    pt = load_pt()
    return int(pt[np.ix_(hand_pair_ids(hole), board_triple_ids(board))].max())


def pairbest(board, pt=None) -> np.ndarray:
    """(630,) best score each hole PAIR can make with this 5-card board."""
    if pt is None:
        pt = load_pt()
    return pt[:, board_triple_ids(board)].max(axis=1)


def score_hands_on_board(hand_pairids: np.ndarray, board, pt=None) -> np.ndarray:
    """Score many hands (given as (N,10) pair ids) against one board."""
    return pairbest(board, pt)[hand_pairids].max(axis=1)
