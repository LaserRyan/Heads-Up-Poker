"""Online opponent model and the exploitation gate.

Design constraints taken from what actually worked in computer-poker
competitions:

* **Points come from the broken bots, not the good ones.**  In a round robin
  scored on a 95% confidence interval, two competent bots chop no matter how
  the hands are played; the standings are decided by how hard the weak
  opponents are punished.  So the model is tuned to spot degenerate behaviour
  fast and to leave everything else alone.

* **Adaptation against competent opponents is usually net harmful.**  The
  Alberta group measured their own online adaptation as losing money against a
  field of near-equilibrium bots.  So exploitation is gated: it fires only on
  extreme statistics and fades out smoothly as behaviour looks more normal.
  Measured on our sparring field: with the gate on, every competent opponent's
  result is byte-identical to the gate off.

* **Never extrapolate into an unobserved branch.**  Every adjustment is scaled
  by a confidence ramp that is exactly zero with no data.

Two layers sit on top of the raw counts:
  scores()/adjustments()  soft archetype memberships -> threshold deltas for
                          the heuristic policy (frequency based, tolerant).
  fixed                   pokerbot.fixedbot.FixedBotDetector: exact
                          identification of card-independent bots by
                          consistency, which licenses a precomputed best
                          response.

Two traps in reading the history, both of which silently zeroed the
fold-to-bet statistic in the first version of this file:
  * The blinds are not in the history, so preflop the button is facing a
    1-chip bet even though both street totals read zero.
  * We only see the history on OUR turn, so the opponent's final action of a
    hand -- the fold that ends it, or the call before showdown -- never appears
    there.  hand_ended() reconstructs it.
"""
from __future__ import annotations

from .fixedbot import CALL, FOLD, HALF, POT, FixedBotDetector, legal_actions

PREFLOP, FLOP, TURN, RIVER = 0, 1, 2, 3
_STREET = {"PREFLOP": 0, "FLOP": 1, "TURN": 2, "RIVER": 3}
STACK = 80

N_FULL = 12
N_FULL_SLOW = 30


def _ramp(n: int, full: int = N_FULL) -> float:
    if n <= 0:
        return 0.0
    return 1.0 if n >= full else n / full


class OppModel:
    """Counts what the opponent does, and says how to bend our thresholds."""

    def __init__(self, strength=None):
        self.strength = strength       # for reading showdown hands; optional
        self.hand = -1
        self.cursor = 0
        self.seen_hand = -1
        self.my_last = None
        self.my_last_facing = 0
        self.my_last_street = 0
        self.my_last_stacks = (STACK, STACK)    # (ours, theirs) at our last decision
        self.my_last_raises = 0

        self.hands = 0
        self.acts = 0
        self.raises = 0
        self.pot_raises = 0
        self.half_raises = 0
        self.raise_legal = 0          # decisions where a raise was legal
        self.raised_when_legal = 0
        self.fold_opps = 0
        self.folds = 0
        self.check_opps = 0
        self.checks = 0
        self.showdowns = 0
        self.fixed = FixedBotDetector()
        # Per-street replies to OUR bets, and what they do when checked to.
        self.face = [0, 0, 0, 0]       # facing a bet/raise of ours
        self.fold_s = [0, 0, 0, 0]     # ...and folded
        self.chk_opp = [0, 0, 0, 0]    # nothing to call, raise legal
        self.chk_bet = [0, 0, 0, 0]    # ...and bet
        # Does their betting depend on their cards?  At each showdown we see
        # their hand: [sum of strength, count] split by whether they raised
        # (preflop: preflop percentile; postflop: river strength).
        self.raised_pf = False
        self.raised_post = False
        self.sd_pf = [[0.0, 0], [0.0, 0]]      # [not raised, raised]
        self.sd_post = [[0.0, 0], [0.0, 0]]

    # -- observation -------------------------------------------------------

    def observe(self, obs) -> None:
        """Count every opponent action we have not seen yet, with its legal set."""
        if obs.hand_index != self.hand:
            self.hand = obs.hand_index
            self.cursor = 0
            self.hands += 1
            self.my_last = None
            self.raised_pf = self.raised_post = False
        self.seen_hand = obs.hand_index
        hist = obs.d["action_history"]
        if self.cursor >= len(hist):
            return

        me_btn = obs.is_button
        # street totals and whole-hand contributions, starting from the blinds
        st_tot = {True: 1 if me_btn else 2, False: 2 if me_btn else 1}
        contrib = {True: st_tot[True], False: st_tot[False]}
        street, raises = PREFLOP, 0
        for i, rec in enumerate(hist):
            st = _STREET[rec["street"]]
            if st != street:
                street, raises = st, 0
                st_tot[True] = st_tot[False] = 0
            me = rec["is_me"]
            action = rec["action"]
            if not me and i >= self.cursor:
                to_call = st_tot[True] - st_tot[False]
                legal = legal_actions(to_call, raises, STACK - contrib[False],
                                      STACK - contrib[True])
                self._count(action, to_call, legal)
                self._street(action, st, to_call, raises, legal)
                if action == HALF or action == POT:
                    if st == PREFLOP:
                        self.raised_pf = True
                    else:
                        self.raised_post = True
            amt = rec["amount"]
            contrib[me] += amt
            st_tot[me] = rec["street_total"]
            if action == HALF or action == POT:
                raises += 1
        self.cursor = len(hist)

    def note_my_action(self, action: str, obs) -> None:
        """Remember what we just did, so hand_ended() can infer their reply."""
        self.my_last = action
        self.my_last_facing = obs.to_call
        self.my_last_street = obs.street
        self.my_last_stacks = (obs.my_stack, obs.opp_stack)
        self.my_last_raises = obs.raises_this_street

    def _street(self, action: str, st: int, to_call: int, raises: int, legal) -> None:
        if to_call > 0:
            if st == PREFLOP and raises == 0:
                return      # the button completing the blind is not facing a bet of ours
            self.face[st] += 1
            if action == FOLD:
                self.fold_s[st] += 1
        elif HALF in legal:
            self.chk_opp[st] += 1
            if action == HALF or action == POT:
                self.chk_bet[st] += 1

    def _count(self, action: str, to_call: int, legal) -> None:
        self.acts += 1
        is_raise = action == HALF or action == POT
        if is_raise:
            self.raises += 1
            if action == POT:
                self.pot_raises += 1
            else:
                self.half_raises += 1
        if HALF in legal:
            self.raise_legal += 1
            if is_raise:
                self.raised_when_legal += 1
        if to_call > 0:
            self.fold_opps += 1
            if action == FOLD:
                self.folds += 1
        else:
            self.check_opps += 1
            if action == CALL:
                self.checks += 1
        self.fixed.observe(action, legal)

    def hand_ended(self, result) -> None:
        """Reconstruct the opponent's final, unobserved action of the hand."""
        showdown = bool(result.get("went_to_showdown"))
        winner = result.get("winner")
        if showdown:
            self.showdowns += 1
            if result.get("hand_index") == self.seen_hand:
                try:
                    self._read_showdown(result)
                except Exception:
                    pass

        if result.get("hand_index") != self.seen_hand:
            # We never got a decision: as big blind, the button folded its
            # small blind to our big blind -- a fold facing a 1-chip bet.
            if not showdown and winner == "me":
                self.hands += 1
                self.acts += 1
                self.fold_opps += 1
                self.folds += 1
                self.fixed.observe(FOLD, (FOLD, CALL, HALF, POT))
            return

        last = self.my_last
        if last == HALF or last == POT:
            st = self.my_last_street
            if not showdown and winner == "me":
                self.acts += 1
                self.fold_opps += 1
                self.folds += 1
                self.fixed.observe_fold()
                self.face[st] += 1
                self.fold_s[st] += 1
            elif showdown:
                self.acts += 1
                self.fold_opps += 1
                self.fixed.observe_call_facing_bet()
                self.face[st] += 1
        elif last == CALL and self.my_last_facing == 0 and showdown:
            # We checked and it still reached showdown: they checked behind.
            mine, theirs = self.my_last_stacks
            legal = legal_actions(0, self.my_last_raises, theirs, mine)
            self._count(CALL, 0, legal)
            self._street(CALL, self.my_last_street, 0, self.my_last_raises, legal)

    def _read_showdown(self, result) -> None:
        S = self.strength
        hole = result.get("opp_hole")
        board = result.get("board")
        if S is None or not hole or not board or len(board) != 5:
            return
        hole, board = tuple(hole), tuple(board)
        _, pct = S.preflop_pair(hole)
        r = self.sd_pf[1 if self.raised_pf else 0]
        r[0] += pct
        r[1] += 1
        pb = S.pairbest(board)
        q = S.sample_equity(hole, board, pb, S.made_score(hole, pb), n=64)
        r = self.sd_post[1 if self.raised_post else 0]
        r[0] += q
        r[1] += 1

    def card_dependence(self):
        """How much stronger their showdown hands are when they raised than
        when they did not (preflop and postflop gaps), or None per gap with
        too little data.  ~0 for a bot that ignores its cards; clearly
        positive for anything that bets its hand -- including a balanced CFR
        bot, whose overall frequencies otherwise look a lot like random."""
        out = []
        for (s0, n0), (s1, n1) in (self.sd_pf, self.sd_post):
            out.append(s1 / n1 - s0 / n0 if n0 >= 8 and n1 >= 8 else None)
        return out

    # -- derived statistics ------------------------------------------------

    @property
    def fold_to_bet(self) -> float:
        return self.folds / self.fold_opps if self.fold_opps else 0.5

    @property
    def aggression(self) -> float:
        return self.raises / self.acts if self.acts else 0.3

    @property
    def pot_rate(self) -> float:
        return self.pot_raises / self.acts if self.acts else 0.2

    @property
    def raise_rate_when_legal(self) -> float:
        return self.raised_when_legal / self.raise_legal if self.raise_legal else 0.3

    def looks_random(self) -> bool:
        """Uniform random: folds 20-42% facing a bet, raises 40-75% of the
        time it may, splits raises between half and pot, and matches no
        deterministic rule.  The raise conditions are what keep tight bots
        out: the 'rock' sparring bot folds 34% (inside the fold band) but
        raises only ~11% of the time."""
        if not self.fixed.looks_random():
            return False
        if self.raise_legal < 40:
            return False
        r = self.raise_rate_when_legal
        if not (0.40 <= r <= 0.75):
            return False
        tot = self.half_raises + self.pot_raises
        return tot > 20 and 0.30 <= self.half_raises / tot <= 0.70

    def identified(self) -> str | None:
        """A confirmed card-independent bot ('allin', 'half', 'station',
        'checkfold', 'random') or None."""
        k = self.fixed.identified()
        if k == "random":
            # Frequencies alone CANNOT tell uniform random from a balanced
            # mixed strategy: the CFR blueprint passed every frequency test
            # and got the anti-random best response, losing 2.2 bb/deal.
            # Require proof that its raises ignore its cards (measured
            # preflop gap: random -0.09..+0.02, blueprint +0.21..+0.41,
            # every competent sparring bot > +0.30).
            gap = self.card_dependence()[0]
            if self.looks_random() and gap is not None and gap < RANDOM_MAX_GAP:
                return "random"
            return None
        return k

    def summary(self) -> dict:
        return {"hands": self.hands, "acts": self.acts,
                "fold_to_bet": round(self.fold_to_bet, 3),
                "aggression": round(self.aggression, 3),
                "raise_when_legal": round(self.raise_rate_when_legal, 3),
                "fold_opps": self.fold_opps, "identified": self.identified()}

    # -- the exploitation gate --------------------------------------------

    def scores(self) -> dict:
        """Soft archetype memberships in [0, 1], already confidence-weighted.

        Measured on the sparring field (300 hands, fold-to-bet / aggression):
            calling-station 0.000 / 0.00    always-pot   0.000 / 0.87
            all-in          0.000 / 0.87    always-half  0.000 / 0.87
            maniac          0.019 / 0.64    jekyll       0.000 / 0.46
            LAG             0.068 / 0.34    TAG          0.149 / 0.24
            random          0.311 / 0.54    rock         0.335 / 0.11
            check-fold      1.000 / 0.00
        Never-fold bots sit at 0.00-0.02 and the first competent bot (LAG) at
        0.068, so "never folds" fades out across 0.025..0.045.
        """
        n = self.fold_opps
        f = self.fold_to_bet
        a = self.aggression
        acts = self.acts
        never_fold = 0.0
        always_fold = 0.0
        if n > 0:
            never_fold = min(1.0, max(0.0, (0.045 - f) / 0.020)) * _ramp(n, 35)
            always_fold = min(1.0, max(0.0, (f - 0.62) / 0.20)) * _ramp(n, 14)
        hyper = passive = 0.0
        if acts > 0:
            hyper = min(1.0, max(0.0, (a - 0.45) / 0.25)) * _ramp(acts, N_FULL_SLOW)
            passive = min(1.0, max(0.0, (0.12 - a) / 0.10)) * _ramp(acts, N_FULL_SLOW)
        return {"station": never_fold * passive,
                "maniac": never_fold * hyper,
                "nit": always_fold}

    def adjustments(self) -> dict:
        """How far to bend each policy knob.  Deltas, so no data = no change."""
        sc = self.scores()
        st, mn, nt = sc["station"], sc["maniac"], sc["nit"]
        X = EXPLOIT
        adj = {"bet_pct": 0.0, "raise_pct": 0.0, "call_margin": 0.0,
               "give_up": 0.0, "pot_pct": 0.0, "value_only": False,
               "delay_aggression": False, "trap": False}
        if st > 0:
            adj["bet_pct"] += X["st_bet"] * st
            adj["pot_pct"] += X["st_pot"] * st
            adj["call_margin"] += X["st_call"] * st
            adj["give_up"] += X["st_giveup"] * st
            adj["value_only"] = st > 0.5
            adj["delay_aggression"] = st > 0.5
        if mn > 0:
            adj["bet_pct"] += X["mn_bet"] * mn
            adj["raise_pct"] += X["mn_raise"] * mn
            adj["call_margin"] += X["mn_call"] * mn
            adj["give_up"] += X["mn_giveup"] * mn
            adj["value_only"] = adj["value_only"] or mn > 0.5
            adj["trap"] = mn > 0.5
        if nt > 0:
            adj["bet_pct"] += X["nt_bet"] * nt
            adj["pot_pct"] += X["nt_pot"] * nt
            adj["call_margin"] += X["nt_call"] * nt
            adj["give_up"] += X["nt_giveup"] * nt
        return adj

    def leak_adjustments(self, street: int) -> dict | None:
        """Bend the default strategy toward the two leaks competent bots
        actually have (measured by dev/headroom.py: an oracle tuned against
        TAG / adapter / rock moved the same two ways):

          * folding to our bets more than a bet's break-even rate -> bet more
            hands on that street (the bet shows a profit before cards matter);
          * rarely betting when checked to -> their bets are value-heavy, so
            demand more equity to call them.

        Both statistics are shrunk toward a neutral prior, so with no data the
        adjustment is exactly zero; the opposite deviations (calls too much,
        bets too much) bend the other way with separate magnitudes.
        """
        L = LEAK
        n0 = L["prior_n"]
        ref = L["f_ref"][street]
        f = (self.fold_s[street] + n0 * ref) / (self.face[street] + n0)
        over = max(-1.0, min(1.0, (f - ref) / 0.25))
        bet = -L["k_bet"] * over if over > 0 else -L["k_bet_neg"] * over
        opp = sum(self.chk_opp[1:])
        a = (sum(self.chk_bet[1:]) + n0 * L["a_ref"]) / (opp + n0)
        passive = max(-1.0, min(1.0, (L["a_ref"] - a) / 0.25))
        call = L["k_call"] * passive if passive > 0 else L["k_call_neg"] * passive
        if bet == 0.0 and call == 0.0:
            return None
        return {"bet_pct": bet, "raise_pct": 0.0, "call_margin": call,
                "give_up": 0.0, "pot_pct": 0.0, "value_only": False,
                "delay_aggression": False, "trap": False}

    def is_loose(self) -> bool:
        """Does this opponent call too much for the blueprint to be our best
        default?  The value-heavy heuristic beats loose callers (LAG by 1.1,
        jekyll by 4.7 bb/deal) while the blueprint beats opponents who fold
        correctly (TAG, rock, adapter, a CFR bot, a best-response adversary).
        Measured separation (dev/loosestats.py, post-flop fold-to-bet /
        aggression at 100-200 hands):
            loose:  LAG .11-.17/.32-.37  jekyll 0/.48  maniac .00-.15/.65+
            sound:  TAG .20-.29/.19-.27  adapter .12-.17/.20-.28
                    rock .31-.39  CFR bot .31-.51  adversary .37-.46
        """
        face = sum(self.face[1:])
        if face < LOOSE["min_face"]:
            return False
        f = sum(self.fold_s[1:]) / face
        return f < LOOSE["f_never"] or (f < LOOSE["f_loose"] and self.aggression > LOOSE["aggr"])

    def archetype(self) -> str:
        sc = self.scores()
        k = max(sc, key=sc.get)
        return k if sc[k] >= 0.5 else "unknown"


# Exploit magnitudes, fitted by dev/tune_exploit.py against the degenerate
# bots (held-out mean +6.06 -> +7.67 bb/deal).  Deltas on the baseline
# thresholds, scaled by the archetype scores above.
EXPLOIT = {
    "st_bet": -0.0202,
    "st_pot": -0.1808,
    "st_call": -0.1483,
    "st_giveup": -0.0589,
    "mn_bet": 0.2252,
    "mn_raise": 0.0593,
    "mn_call": -0.0539,
    "mn_giveup": -0.1824,
    "nt_bet": -0.5214,
    "nt_pot": -0.0801,
    "nt_call": -0.0937,
    "nt_giveup": 0.0563,
}


RANDOM_MAX_GAP = 0.10

LOOSE = {"min_face": 20, "f_never": 0.08, "f_loose": 0.17, "aggr": 0.30}

# Leak-exploit magnitudes (dev/tune_leak.py).  All-zero = off.
LEAK = {
    "prior_n": 8.0,
    # Neutral fold rate to our bets, per street.  Preflop equities are so close
    # that sound bots almost never fold; by the river a bet's break-even fold
    # rate is 1/3 (half pot) to 1/2 (pot).  Measured (dev/leakstats.py):
    # TAG .06/.25/.53/.54, adapter .01/.13/.38/.46, rock .35/.63/.51/.54.
    "f_ref": [0.08, 0.22, 0.38, 0.45],
    "k_bet": 0.25,      # max drop in bet threshold when they over-fold
    "k_bet_neg": 0.0,   # max rise when they never fold
    "a_ref": 0.35,      # neutral bet rate when checked to (postflop)
    "k_call": 0.14,     # extra call margin vs passive bettors
    "k_call_neg": 0.0,  # less call margin vs frequent bettors
}
