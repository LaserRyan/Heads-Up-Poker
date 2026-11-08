"""A cheap view over the judge's raw observation JSON.

The stock plumbing rebuilds an Observation dataclass (and an ObservedAction
dataclass per history entry) on every request.  At ~4000 requests a half and a
total budget of 8 seconds that is real money, so we read the dict directly and
only materialise what a decision actually touches.

Street and action names are kept as the judge's strings/ints; the bot never
needs the enums, and returning the action NAME is what the judge wants back.
"""
from __future__ import annotations

PREFLOP, FLOP, TURN, RIVER = 0, 1, 2, 3
STREET_NUM = {"PREFLOP": 0, "FLOP": 1, "TURN": 2, "RIVER": 3}

FOLD = "FOLD"
CHECK_CALL = "CHECK_CALL"
RAISE_HALF = "RAISE_HALF"
RAISE_POT = "RAISE_POT"


class Obs:
    """Thin wrapper; attribute names match the Observation in game.py."""

    __slots__ = ("d", "street", "hole_cards", "board", "pot", "to_call", "my_bet",
                 "opp_bet", "my_stack", "opp_stack", "is_button", "legal_actions",
                 "raise_to", "min_raise_to", "raises_this_street", "hand_index",
                 "big_blind")

    def __init__(self, d: dict):
        self.d = d
        self.street = STREET_NUM[d["street"]]
        self.hole_cards = d["hole_cards"]
        self.board = d["board"]
        self.pot = d["pot"]
        self.to_call = d["to_call"]
        self.my_bet = d["my_bet"]
        self.opp_bet = d["opp_bet"]
        self.my_stack = d["my_stack"]
        self.opp_stack = d["opp_stack"]
        self.is_button = d["is_button"]
        self.legal_actions = d["legal_actions"]
        self.raise_to = d["raise_to"]
        self.min_raise_to = d["min_raise_to"]
        self.raises_this_street = d["raises_this_street"]
        self.hand_index = d["hand_index"]
        self.big_blind = d["big_blind"]

    @property
    def action_history(self):
        return self.d["action_history"]

    def can_raise(self) -> bool:
        return RAISE_POT in self.legal_actions or RAISE_HALF in self.legal_actions

    def pot_after_call(self) -> int:
        return self.pot + self.to_call

    def pot_odds(self) -> float:
        """Fraction of the final pot we must put in to call (0 when checking)."""
        t = self.to_call
        return t / (self.pot + t) if t else 0.0
