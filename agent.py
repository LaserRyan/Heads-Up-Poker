"""
agent.py - the bot the tournament plays (it runs `python agent.py`).

This repo holds two bots; ACTIVE picks which one this file runs:
    "agent_shortstack"  ShortStackOmaha (pokerbot/), named "Tulip v3"   <- SUBMITTED
    "agent_tulip"       Tulip v2 (tulip/)

Run either bot directly too, e.g. `python agent_tulip.py`.

Why we submit ShortStackOmaha (decided 2026-11-08)
-------------------------------------------------
Same 3 fresh seeds x 1000 deals, tools/run_matches.py, zero errors for both:
  * Head-to-head it beats Tulip v2 by ~0.9 bb/deal (and beats Tulip v1;
    Tulip v2 only chops v1). Strong teams decide the standings.
  * Against the baselines it wins far bigger (+9 to +18 bb/deal vs Tulip's
    +5 to +12), which helps the tiebreak.
  * Against the realistic practice bots both win almost everything
    (Tulip 14/15, it 13/15); Tulip wins by more vs maniac, sticky station,
    loose-aggressive and honest bots - the place a future version could
    borrow Tulip's exploits.
  * It thinks ~0.5 s per half vs Tulip's ~2 s, on a faster server, so the
    8 s time bank is never at risk. It also has a safety valve that falls
    back to its game-theory blueprint if it is being outplayed.
  Open risk: its pinned orjson/httptools/uvloop versions are unverified for
  the tournament build (all have fallbacks). Switch back by setting
  ACTIVE = "agent_shortstack".
"""

import importlib
import os
import runpy
import sys

ACTIVE = "agent_shortstack"

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

if __name__ == "__main__":
    runpy.run_path(os.path.join(_HERE, ACTIVE + ".py"), run_name="__main__")
else:
    # `import agent` (tests, smoke test) gets the active bot's module itself;
    # loaders that keep this module object (tools/run_matches.py) get its
    # contents copied in.
    _bot = importlib.import_module(ACTIVE)
    globals().update({k: v for k, v in vars(_bot).items() if not k.startswith("__")})
    sys.modules[__name__] = _bot
