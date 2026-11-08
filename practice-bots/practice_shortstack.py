"""
practice_shortstack.py - practice opponent: ShortStackOmaha (agent_shortstack.py).

That bot takes and returns plain JSON-style data (dicts and action names)
instead of Observation objects, so this file adapts it for
tools/run_matches.py. To play it through the real judge instead, run
    PORT=5009 python agent_shortstack.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Action  # noqa: E402
import agent_shortstack as _S  # noqa: E402


def new_half(config, memory):
    _S.new_half(config.to_dict(), memory)


def choose_action(obs, memory):
    a = _S.choose_action(obs.to_dict(), memory)
    return Action[a] if isinstance(a, str) else Action(a)


def on_hand_end(result, memory):
    _S.on_hand_end(result.to_dict(), memory)
