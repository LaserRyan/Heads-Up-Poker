"""
build_preflop_table.py - precompute every starting hand's equity vs a random hand.

Run once from the repo root (takes a few minutes):
    python tools/build_preflop_table.py [samples_per_hand]

Writes tulip/data/preflop_equity.npy: float16 array of length C(36,5), indexed by
the hand's colex index (see tulip/fast_eval.py). Hands that are the same up to
relabelling suits share one simulated value.
"""

import os
import sys
import time
from itertools import combinations, permutations
from math import comb

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tulip.fast_eval import N5, colex, omaha_batch, scores_table  # noqa: E402


def all_hands() -> np.ndarray:
    """(N5, 5) array, row i = the hand whose colex index is i."""
    hands = np.empty((N5, 5), dtype=np.int64)
    for combo in combinations(range(36), 5):
        hands[sum(comb(c, k + 1) for k, c in enumerate(combo))] = combo
    return hands


def canonical_ids(hands: np.ndarray) -> np.ndarray:
    """Smallest colex index over all 24 suit relabellings of each hand."""
    ranks, suits = hands >> 2, hands & 3
    best = np.full(len(hands), np.iinfo(np.int64).max)
    for perm in permutations(range(4)):
        mapped = np.sort(ranks * 4 + np.array(perm)[suits], axis=1)
        best = np.minimum(best, colex(mapped))
    return best


def equity(hands: np.ndarray, samples: int, rng: np.random.Generator) -> np.ndarray:
    """Equity (win + tie/2) of each hand vs a random hand, random board."""
    b = len(hands)
    keys = rng.random((b, samples, 36))
    rows = np.arange(b)[:, None]
    keys[rows, :, hands] = 2.0  # never draw our own cards
    keys = keys.reshape(b, samples, 36)
    drawn = np.argsort(keys, axis=2)[:, :, :10]           # (b, S, 10)
    opp = drawn[:, :, :5].reshape(-1, 5)
    board = drawn[:, :, 5:].reshape(-1, 5)
    mine = np.repeat(hands, samples, axis=0)
    s_me = omaha_batch(mine, board)
    s_op = omaha_batch(opp, board)
    res = (s_me > s_op) + 0.5 * (s_me == s_op)
    return res.reshape(b, samples).mean(axis=1)


def main() -> None:
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    t0 = time.time()
    scores_table()
    hands = all_hands()
    canon = canonical_ids(hands)
    classes = np.unique(canon)
    print(f"{len(classes)} suit-distinct starting hands, {samples} samples each", flush=True)

    rng = np.random.default_rng(12345)
    eq_by_class = np.empty(len(classes))
    batch = 40
    for start in range(0, len(classes), batch):
        ids = classes[start:start + batch]
        eq_by_class[start:start + batch] = equity(hands[ids], samples, rng)
        if start % 2000 == 0:
            print(f"  {start}/{len(classes)}  {time.time() - t0:.0f}s", flush=True)

    lookup = dict(zip(classes.tolist(), range(len(classes))))
    pos = np.array([lookup[c] for c in canon.tolist()])
    table = eq_by_class[pos].astype(np.float16)
    out = os.path.join(ROOT, "tulip", "data", "preflop_equity.npy")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, table)
    print(f"wrote {out} in {time.time() - t0:.0f}s; "
          f"equity range {table.min():.3f}..{table.max():.3f}")


if __name__ == "__main__":
    main()
