"""
agent_shortstack.py - ShortStackOmaha, shown to the judge as "Tulip v3" (our
team's inside joke - none of Tulip's code is in here). Heads-Up Short Deck
5-Card PLO. This is the bot agent.py runs.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

Layout
------
The strategy lives in the `pokerbot` package next to this file:

    pokerbot/cards.py     short-deck evaluation (630 x 7140 pair/triple table)
    pokerbot/strength.py  hand strength inside the per-decision time budget
    pokerbot/policy.py    the decision rules
    pokerbot/server.py    the ASGI plumbing that talks to the judge

The three functions the rules ask for (`new_half`, `choose_action`,
`on_hand_end`) are defined here and are the only entry points the judge uses.

Why the plumbing is not the stock FastAPI app
---------------------------------------------
The judge charges `wall_time - 1 ms` per request against an 8-second-per-half
bank.  Measured: stock FastAPI costs 2.11 ms charged per call (~10.6 s per
half, i.e. the bank runs out), a raw ASGI app costs 0.66 ms, and a server that
does nothing at all also costs 0.66 ms.  The HTTP contract is unchanged - same
routes, same status codes, same reply shape - so the judge, test_agent.py and
tools/smoke_test.py cannot tell the difference.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pokerbot import server
from pokerbot.blueprint import Blueprint
from pokerbot.bp_policy import BlueprintPolicy
from pokerbot.luck import adjusted_net
from pokerbot.obs import CHECK_CALL, FOLD, Obs
from pokerbot.opponent import OppModel
from pokerbot.policy import Budget, Policy
from pokerbot.strength import Strength

PARTICIPANT = "5-Points"
AGENT_NAME = "Tulip v3"   # really ShortStackOmaha; see agent.py for why

# This bot reads plain JSON data (dicts, action names) rather than the
# Observation objects game.py builds; tools/run_matches.py adapts when set.
TAKES_DICTS = True

# Slow setup belongs at import time, not in the first decision: the server has
# 15 s to answer its first request but only ~0.5 ms per decision afterwards.
STRENGTH = Strength()
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pokerbot", "data")

# Precomputed best responses to the card-independent bots, keyed by the label
# the consistency detector (pokerbot/fixedbot.py) assigns.  Missing files are
# simply skipped -- the bot then falls back to the frequency-based exploit.
BEST_RESPONSES = {}
for _kind in ("allin", "half", "station", "checkfold", "random"):
    _path = os.path.join(DATA, f"br_{_kind}.npz")
    if os.path.exists(_path):
        _bp = Blueprint(_path)
        if _bp.ok:
            BEST_RESPONSES[_kind] = _bp

# The CFR blueprint: the default against sound opponents, and the safety net.
# Measured with a trained best response against each strategy: the heuristic
# concedes +5.9 bb/deal to a perfect adversary, the blueprint -0.01 +- 0.49.
# The safety valve below switches to it whenever our own luck-adjusted result
# turns significantly negative.  See README_STRATEGY.md.
BLUEPRINT = Blueprint()
VALVE_MIN_HANDS = 50          # never judge on fewer (luck-adjusted) hands; dev/valve_test2.py
VALVE_Z = 2.0                 # switch when mean + Z*stderr < 0
LUCK_ADJUST = True            # judge the valve on all-in-EV, not raw chips

# Archetype confidence needed before the frequency-based exploit takes over.
EXPLOIT_SWITCH = 0.6

# Leak exploit against competent opponents (OppModel.leak_adjustments): bet
# more on streets where they over-fold, call tighter when they rarely bet.
LEAK_EXPLOIT = True

# Strategy for opponents that are neither a confirmed fixed bot nor a
# degenerate archetype: "heuristic" (+ leak exploit), "blueprint", or "hybrid"
# (blueprint unless OppModel.is_loose(); measured best, dev/data/ab_hybrid.txt).
DEFAULT = "hybrid"


# ===== YOUR CODE HERE =====

def new_half(config, memory):
    """Start of a half-match (1000 hands). `memory` is a fresh empty dict."""
    budget = Budget()
    try:
        get = config.get if isinstance(config, dict) else (lambda k: getattr(config, k, None))
        budget.num_hands = int(get("num_hands") or 1000)
        budget.total = float(get("time_bank_seconds") or 8.0)
    except Exception:
        pass
    heuristic = Policy(STRENGTH, budget=budget)
    memory["hands"] = 0
    memory["budget"] = budget
    memory["opp"] = OppModel(STRENGTH)
    memory["heuristic"] = heuristic
    memory["br"] = {k: BlueprintPolicy(STRENGTH, bp, heuristic, mode="purify")
                    for k, bp in BEST_RESPONSES.items()}
    memory["blueprint"] = (BlueprintPolicy(STRENGTH, BLUEPRINT, heuristic, mode="purify")
                           if BLUEPRINT.ok else heuristic)
    memory["safe_mode"] = False
    memory["net"] = [0, 0.0, 0.0]     # Welford: count, mean, M2 of chips per hand


def choose_action(obs, memory):
    """Return one of obs['legal_actions'] (the judge accepts the name).

    Layered, most specific first:
      1. the opponent is a confirmed card-independent bot -> its exact
         precomputed best response;
      2. the opponent's frequencies say degenerate (never folds / always
         folds) -> the heuristic with exploit adjustments;
      3. otherwise -> the default strategy, untouched.
    """
    t0 = time.perf_counter()
    legal = obs["legal_actions"]
    action = None
    try:
        o = Obs(obs)
        opp = memory["opp"]
        opp.observe(o)
        budget = memory["budget"]
        kind = opp.identified()
        if memory["safe_mode"]:
            # The valve outranks everything, best responses included: if we
            # are clearly losing, whatever we think we know is wrong.
            action = memory["blueprint"].act(o, None)
        elif kind is not None and kind in memory["br"] and budget.tier < 2:
            action = memory["br"][kind].act(o)
        else:
            sc = opp.scores()
            if max(sc.values()) >= EXPLOIT_SWITCH or budget.tier >= 2:
                action = memory["heuristic"].act(o, opp.adjustments())
            elif DEFAULT == "blueprint" or (DEFAULT == "hybrid" and not opp.is_loose()):
                action = memory["blueprint"].act(o, None)
            else:
                adj = opp.leak_adjustments(o.street) if LEAK_EXPLOIT else None
                action = memory["heuristic"].act(o, adj)
        if action not in legal:
            action = CHECK_CALL if CHECK_CALL in legal else FOLD
        opp.note_my_action(action, o)
        memory["last"] = (o.hand_index, o.hole_cards, len(o.board))
    except Exception:
        # Never let a strategy bug cost an error: 5 errors forfeits the half,
        # and a forfeit loses the whole match whatever the chip count says.
        action = CHECK_CALL if CHECK_CALL in legal else FOLD
    try:
        memory["budget"].update(time.perf_counter() - t0, obs["hand_index"])
    except Exception:
        pass
    return action


def on_hand_end(result, memory):
    """Called once per finished hand, before the next decision is asked for."""
    memory["hands"] += 1
    try:
        memory["opp"].hand_ended(result)
    except Exception:
        pass
    try:
        _safety_valve(_luck_adjusted(result, memory), memory)
    except Exception:
        pass


ALL_IN_POT = 160      # both 80-chip stacks in the middle


def _luck_adjusted(result, memory):
    """Our result for this hand with the all-in runout averaged out
    (pokerbot/luck.py).  Unbiased, and far less noisy than raw chips, so the
    valve can tell "unlucky" from "outplayed" sooner."""
    net = float(result["my_net"])
    if not LUCK_ADJUST or not result.get("went_to_showdown") or result.get("pot") != ALL_IN_POT:
        return net
    last = memory.get("last")
    if not last or last[0] != result.get("hand_index"):
        return net
    # The last chip always goes in on the street of our last decision: either
    # we shoved and were called, or we called their shove.
    return adjusted_net(result, last[1], last[2])


def _safety_valve(x, memory):
    """Switch to the unexploitable blueprint if we are clearly losing.

    The test is one-sided and conservative: our mean result per hand must be
    below zero by VALVE_Z standard errors after at least VALVE_MIN_HANDS hands.
    Against an opponent that is genuinely beating the heuristic (a trained best
    response wins ~3 bb/hand) that fires after roughly 100 hands; against a
    competent opponent we merely break even with, a false trip needs a 2-sigma
    run of bad cards -- and costs little, since the blueprint plays about as
    well as the heuristic against competent bots.
    """
    st = memory["net"]
    n, mean, m2 = st[0] + 1, st[1], st[2]
    d = x - mean
    mean += d / n
    m2 += d * (x - mean)
    st[0], st[1], st[2] = n, mean, m2
    if memory["safe_mode"] or n < VALVE_MIN_HANDS:
        return
    var = m2 / (n - 1)
    if mean + VALVE_Z * (var / n) ** 0.5 < 0:
        memory["safe_mode"] = True

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING =====
MEMORY = {}
STATUS = {"started": False}
TOKEN = os.environ.get("AGENT_TOKEN") or None

app = server.make_app(globals())

if __name__ == "__main__":
    # The tournament sets PORT as an environment variable.
    port = int(os.environ.get("PORT", "5008"))
    server.serve(app, port, host="0.0.0.0")
