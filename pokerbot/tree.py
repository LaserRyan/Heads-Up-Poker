"""The betting tree, enumerated from the real game engine.

Card-free: in this game the action sequence alone determines every chip amount
(stacks reset to 80 every hand, blinds are fixed, raises are pot fractions), so
the whole betting structure is one small tree shared by every deal.

It is built by driving game.Game itself through every legal action sequence,
which guarantees the tree's raise sizes, min-raise rule, all-in clamping and
4-raise cap are exactly the referee's, not a re-implementation of them.

Roles, not seats: role 0 is the button (small blind, acts first preflop and
last postflop), role 1 is the big blind.  Seats alternate every hand; roles
are what strategy depends on.

Arrays (one row per node):
    actor[n]      role to act, or -1 at a terminal
    street[n]     0..3
    child[n, a]   node reached by action a (FOLD, CHECK_CALL, RAISE_HALF,
                  RAISE_POT), or -1 if a is illegal here
    term[n]       0 = not terminal, 1 = fold, 2 = showdown
    folder[n]     role that folded (fold terminals only)
    contrib[n, r] chips role r has put in by this node
"""
from __future__ import annotations

import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tree.npz")

NOT_TERMINAL, FOLD_TERMINAL, SHOWDOWN_TERMINAL = 0, 1, 2
ACTION_NAMES = ("FOLD", "CHECK_CALL", "RAISE_HALF", "RAISE_POT")
ACTION_INDEX = {n: i for i, n in enumerate(ACTION_NAMES)}


def build():
    """Enumerate the tree with the real engine. Returns a dict of arrays."""
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from game import Action, Deal, Game

    # Any legal deal works: the betting tree does not depend on the cards.
    deal = Deal(hole_cards=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)), board=(10, 11, 12, 13, 14))

    actor, street, term, folder, contrib, child = [], [], [], [], [], []

    def add(g) -> int:
        n = len(actor)
        actor.append(-1)
        street.append(int(g.street))
        term.append(NOT_TERMINAL)
        folder.append(-1)
        # seat 0 is the button in this construction, so seat == role
        contrib.append((g.contributed[0], g.contributed[1]))
        child.append([-1, -1, -1, -1])
        return n

    def visit(g) -> int:
        n = add(g)
        if g.is_terminal:
            res = g.result()
            if res.went_to_showdown:
                term[n] = SHOWDOWN_TERMINAL
            else:
                term[n] = FOLD_TERMINAL
                # the winner is the non-folder
                folder[n] = 1 - res.winner_seat
            return n
        actor[n] = g.to_act
        for a in g.legal_actions():
            h = g.clone()
            h.apply(a)
            child[n][int(a)] = visit(h)
        return n

    g = Game(deal, button_seat=0)
    visit(g)
    return {
        "actor": np.array(actor, dtype=np.int8),
        "street": np.array(street, dtype=np.int8),
        "term": np.array(term, dtype=np.int8),
        "folder": np.array(folder, dtype=np.int8),
        "contrib": np.array(contrib, dtype=np.int16),
        "child": np.array(child, dtype=np.int32),
    }


_TREE = None


def load():
    global _TREE
    if _TREE is None:
        if os.path.exists(DATA):
            z = np.load(DATA)
            _TREE = {k: z[k] for k in z.files}
        else:
            _TREE = build()
            try:
                np.savez(DATA, **_TREE)
            except OSError:
                pass
    return _TREE


def node_for_history(history, tree=None) -> int:
    """Walk the observed action history from the root to the current node.

    The history from the judge lists every action of the hand (blinds
    excluded), and the tree was built from the same engine, so this walk is
    exact.  Returns -1 if the history leaves the tree (it cannot, unless the
    rules changed underneath us -- callers fall back to the heuristic).
    """
    t = tree if tree is not None else load()
    child = t["child"]
    n = 0
    for rec in history:
        a = ACTION_INDEX.get(rec["action"] if isinstance(rec, dict) else rec)
        if a is None:
            return -1
        n = int(child[n, a])
        if n < 0:
            return -1
    return n
