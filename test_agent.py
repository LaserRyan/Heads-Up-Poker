"""
Tests for YOUR bot (agent.py). Run them from the bundle folder:

    python -m pytest -q

They talk to your bot the same way the judge does (through its web routes,
but without starting a real server), so a passing run means the judge can
play with your bot. Add your own tests at the bottom!
"""

import importlib.util
import os
import random
import sys

import pytest
from fastapi.testclient import TestClient

# Works whether this file sits next to agent.py or inside a tests/ folder.
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = HERE if os.path.isfile(os.path.join(HERE, "game.py")) else os.path.dirname(HERE)
sys.path.insert(0, BUNDLE)  # so agent.py can import game.py and evaluator.py

from game import Action, AgentConfig, Game, make_deal  # noqa: E402
from judge_engine import coerce_action  # noqa: E402  (reads answers exactly like the judge)

TOKEN = "test-token"
AUTH = {"X-Agent-Token": TOKEN}


def load_agent():
    """Load a fresh copy of agent.py (its own MEMORY and globals)."""
    spec = importlib.util.spec_from_file_location("agent_under_test", os.path.join(BUNDLE, "agent.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bot():
    """(agent module, test client) with the tournament's token check switched on."""
    module = load_agent()
    module.TOKEN = TOKEN
    return module, TestClient(module.app)


def start_half(client, half=1, seat=0):
    """POST /new-half the way the judge does."""
    body = {"half": half, "seat": seat, "config": AgentConfig().to_dict()}
    return client.post("/new-half", json=body, headers=AUTH)


def play_hands(client, num_hands, seed="test", raise_rate=0.3):
    """Play hands where YOUR bot is seat 0 and a random opponent is seat 1.

    Every decision goes through /act exactly like in a real match, and
    finished hands are delivered with the next /act (like the judge does).
    Returns the list of every action your bot chose.
    """
    rng = random.Random(seed)
    pending = []      # finished hands waiting to be delivered
    chosen = []
    for hand in range(num_hands):
        game = Game(make_deal(seed, hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == 0:
                r = client.post("/act", json={"observation": obs.to_dict(), "hand_results": pending},
                                headers=AUTH)
                pending = []
                assert r.status_code == 200, f"/act failed: {r.json()}"
                reply = r.json()
                assert reply.get("hand_end_errors", []) == [], \
                    f"on_hand_end crashed: {reply['hand_end_errors'][0]['traceback']}"
                action = coerce_action(reply["action"])
                assert action in obs.legal_actions, \
                    f"choose_action returned {reply['action']!r}; legal were {[a.name for a in obs.legal_actions]}"
                chosen.append(action)
            else:
                # Random opponent that raises sometimes, so your bot faces bets.
                raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
                action = rng.choice(raises) if raises and rng.random() < raise_rate else Action.CHECK_CALL
            game.apply(action)
        pending.append(game.hand_result(0).to_dict())
    r = client.post("/hand-end", json={"results": pending}, headers=AUTH)  # end of half
    assert r.status_code == 200 and r.json()["hand_end_errors"] == []
    return chosen


# ---------------------------------------------------------------------------
# Your bot plays correctly
# ---------------------------------------------------------------------------

def test_names_are_set(bot):
    _, client = bot
    info = client.get("/").json()
    assert isinstance(info["participant"], str) and info["participant"]
    assert isinstance(info["agent_name"], str) and info["agent_name"]


def test_new_half_works(bot):
    _, client = bot
    r = start_half(client)
    assert r.status_code == 200, f"new_half crashed: {r.json()}"


def test_every_decision_is_legal(bot):
    # 200 hands with a random opponent: every choose_action answer must be legal.
    _, client = bot
    start_half(client)
    play_hands(client, 200)


def test_works_in_both_seats_and_both_halves(bot):
    _, client = bot
    for half, seat in ((1, 0), (2, 1)):
        assert start_half(client, half, seat).status_code == 200
        play_hands(client, 30, seed=f"half{half}")


def test_handles_aggressive_opponent(bot):
    # An opponent that raises whenever it can: your bot must cope with
    # big bets and all-ins.
    _, client = bot
    start_half(client)
    play_hands(client, 100, seed="aggressive", raise_rate=1.0)


def test_memory_is_fresh_each_half(bot):
    module, client = bot
    start_half(client)
    play_hands(client, 10)
    start_half(client, half=2, seat=1)
    after_reset = dict(module.MEMORY)
    fresh = load_agent()
    fresh.new_half(AgentConfig(), fresh.MEMORY)
    assert after_reset.keys() == fresh.MEMORY.keys(), "memory should only hold what new_half() puts there"


# ---------------------------------------------------------------------------
# The server plumbing still works (only fails if you edited it)
# ---------------------------------------------------------------------------

def test_token_is_checked(bot):
    _, client = bot
    body = {"half": 1, "seat": 0, "config": {}}
    assert client.post("/new-half", json=body).status_code == 401
    assert client.post("/new-half", json=body, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_bad_token_is_refused_before_the_body_is_read(bot):
    # Without the right token the bot must answer 401 without parsing the body,
    # so junk sent by anyone else can't slow your bot down.
    _, client = bot
    junk = b"this is not JSON" * 1000
    assert client.post("/act", content=junk).status_code == 401
    assert client.post("/act", content=junk, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_malformed_body_is_rejected(bot):
    _, client = bot
    start_half(client)
    assert client.post("/act", content=b"not json", headers=AUTH).status_code == 400
    assert client.post("/act", json=[1, 2, 3], headers=AUTH).status_code == 400


def test_act_before_new_half_is_refused(bot):
    _, client = bot
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    assert client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH).status_code == 409


def test_crashes_are_reported_to_the_judge(bot):
    # If choose_action raises, the judge must get a 500 with your traceback.
    module, client = bot

    def broken(obs, memory):
        raise ValueError("oops")

    module.choose_action = broken
    start_half(client)
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    r = client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH)
    assert r.status_code == 500
    assert "ValueError: oops" in r.json()["error"] and "Traceback" in r.json()["traceback"]


def test_reads_port_from_environment():
    source = open(os.path.join(BUNDLE, "agent.py"), encoding="utf-8").read()
    assert 'os.environ.get("PORT"' in source, "agent.py must read its port from $PORT"
    assert '0.0.0.0' in source, "agent.py must listen on 0.0.0.0 so the judge can reach it"


# ---------------------------------------------------------------------------
# Add your own tests below, e.g. "with two aces preflop my bot raises".
# ---------------------------------------------------------------------------


# ===== Our own tests ======================================================

def _random_game_observations(n_hands, seed, raise_rate=0.5):
    """Observations from real games with a random opponent, both seats."""
    rng = random.Random(seed)
    for hand in range(n_hands):
        game = Game(make_deal(f"fuzz{seed}", hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            yield obs, game
            raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
            game.apply(rng.choice(raises) if raises and rng.random() < raise_rate
                       else rng.choice(obs.legal_actions))


def test_fuzz_every_spot_is_legal(bot):
    """3000 hands of random play from both seats: every answer legal, no crash."""
    _, client = bot
    start_half(client)
    seen = 0
    for obs, _ in _random_game_observations(3000, seed=11):
        r = client.post("/act", json={"observation": obs.to_dict(), "hand_results": []},
                        headers=AUTH)
        assert r.status_code == 200, r.text
        assert coerce_action(r.json()["action"]) in obs.legal_actions
        seen += 1
    assert seen > 3000


def test_survives_garbage_in_on_hand_end(bot):
    """A malformed hand result must never be reported as an error."""
    module, client = bot
    start_half(client)
    weird = [{"hand_index": 5}, {"went_to_showdown": True}, {}, {"winner": "me", "hand_index": -3}]
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    r = client.post("/act", json={"observation": obs, "hand_results": weird}, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["hand_end_errors"] == []


def test_strategy_survives_unknown_history(bot):
    """An action history that matches no tree node must fall back, not crash."""
    module, _ = bot
    module.new_half(AgentConfig(), module.MEMORY)
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    obs["action_history"] = [{"street": "FLOP", "is_me": False, "action": "RAISE_POT",
                              "amount": 999, "street_total": 999}] * 9
    assert module.choose_action(obs, module.MEMORY) in obs["legal_actions"]


def test_decisions_are_fast():
    """Mean decision time must leave room in the 8 s bank (~4000 calls/half)."""
    import time
    module = load_agent()
    module.new_half(AgentConfig(), module.MEMORY)
    obs_list = [o.to_dict() for o, _ in _random_game_observations(300, seed=5)]
    for o in obs_list[:50]:
        module.choose_action(o, module.MEMORY)          # warm up
    t = time.perf_counter()
    for o in obs_list:
        module.choose_action(o, module.MEMORY)
    per = (time.perf_counter() - t) / len(obs_list)
    assert per < 0.0008, f"{per*1e3:.2f} ms per decision is too slow"


def test_never_folds_when_checking_is_free(bot):
    _, client = bot
    start_half(client)
    for obs, _ in _random_game_observations(400, seed=3):
        if obs.to_call == 0:
            r = client.post("/act", json={"observation": obs.to_dict(), "hand_results": []},
                            headers=AUTH)
            assert r.json()["action"] != "FOLD"


def test_luck_adjusted_equity_matches_reference_evaluator():
    """The all-in adjustment's exact runout equity must agree with evaluator.py."""
    from itertools import combinations

    import evaluator
    from pokerbot import luck

    for i in range(6):
        deal = make_deal("luck", i)
        a, b = deal.hole_cards
        for k in (3, 4):
            known = tuple(deal.board[:k])
            rest = [c for c in range(36) if c not in set(a) | set(b) | set(known)]
            w = n = 0
            for r in combinations(rest, 5 - k):
                sa, sb = evaluator.omaha_score(a, known + r), evaluator.omaha_score(b, known + r)
                w += 1 if sa > sb else (0.5 if sa == sb else 0)
                n += 1
            assert abs(luck.equity(a, b, known) - w / n) < 1e-12


def test_luck_adjustment_never_breaks_on_hand_end():
    """All-in showdown results (and odd ones) go through the valve without error."""
    module = load_agent()
    mem = {}
    module.new_half(AgentConfig(), mem)
    calls = 0
    for o, _ in _random_game_observations(200, seed=21):
        module.choose_action(o.to_dict(), mem)
        deal = make_deal(1, o.hand_index)
        module.on_hand_end({"hand_index": o.hand_index, "my_net": 80, "pot": 160,
                            "went_to_showdown": True, "board": list(deal.board),
                            "opp_hole": list(deal.hole_cards[1]), "winner": "me"}, mem)
        calls += 1
    assert mem["net"][0] == calls      # every result reached the valve
