"""
agent.py - the bot the tournament plays (it runs `python agent.py`).

This repo holds two bots; ACTIVE picks which one this file runs:
    "agent_tulip"       Tulip (tulip/), team 5-Points
    "agent_shortstack"  ShortStackOmaha (pokerbot/)

Run either bot directly too, e.g. `python agent_shortstack.py`.
"""

import importlib
import os
import runpy
import sys

ACTIVE = "agent_tulip"

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
