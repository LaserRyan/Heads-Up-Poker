"""
run_matches.py - play your bot against several opponents in-process, fast.

No web servers needed: it imports each bot file and calls its three functions
directly, using the real game.py engine and the judge's duplicate scoring
(both halves, seats swapped, 95% interval on bb/deal).

From the repo root:
    python tools/run_matches.py                          # agent.py vs baselines + practice bots, 3 seeds
    python tools/run_matches.py --seeds 5 --deals 500
    python tools/run_matches.py --opponents baseline-bots/baseline_random.py
    python tools/run_matches.py --bot some_other_agent.py --jobs 2

Thinking time is measured like the judge's time bank (per bot, per half), so
"max time/half" should stay well under 8 s.
"""

import argparse
import glob
import importlib.util
import math
import os
import statistics
import sys
import time
import traceback
import types
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _stub_web_modules() -> None:
    """Bot files import fastapi/uvicorn for their server. We never start it,
    so if those packages are missing, stand in with harmless fakes."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        return
    except ImportError:
        pass

    class _App:
        def __getattr__(self, _name):
            return lambda *a, **k: (lambda f: f)

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = lambda *a, **k: _App()
    fastapi.Request = object
    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = object
    fastapi.responses = responses
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = lambda *a, **k: None
    sys.modules.update({"fastapi": fastapi, "fastapi.responses": responses, "uvicorn": uvicorn})


_stub_web_modules()

from game import (BIG_BLIND, ERROR_LIMIT, TIME_BANK_SECONDS, AgentConfig, Game,  # noqa: E402
                  button_for_deal, default_action, make_deal)

_loaded = 0


def load_bot(path: str):
    """Import a bot file as a fresh module."""
    global _loaded
    _loaded += 1
    spec = importlib.util.spec_from_file_location(f"_bot{_loaded}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    if getattr(mod, "TAKES_DICTS", False):
        return _dict_api(mod)
    return mod


def _dict_api(mod):
    """Wrap a bot that takes JSON-style dicts (like the judge sends) and
    returns action names, so it plays through the same Seat interface."""
    from game import Action

    def choose_action(obs, memory):
        a = mod.choose_action(obs.to_dict(), memory)
        return Action[a] if isinstance(a, str) else Action(a)

    return types.SimpleNamespace(
        new_half=lambda config, memory: mod.new_half(config.to_dict(), memory),
        choose_action=choose_action,
        on_hand_end=lambda result, memory: mod.on_hand_end(result.to_dict(), memory),
    )


class Seat:
    """One bot for one half: its memory, time used and errors."""

    def __init__(self, mod, name: str):
        self.mod, self.name = mod, name
        self.memory: dict = {}
        self.time = 0.0
        self.errors = 0
        self.first_error = ""
        self.forfeited = False

    def _call(self, fn, *args):
        t = time.perf_counter()
        try:
            return fn(*args), True
        except Exception:
            self.errors += 1
            if not self.first_error:
                self.first_error = traceback.format_exc()
            return None, False
        finally:
            self.time += time.perf_counter() - t

    def _check_forfeit(self):
        if self.errors >= ERROR_LIMIT:
            self.forfeited = True

    def new_half(self):
        self._call(self.mod.new_half, AgentConfig(), self.memory)
        self._check_forfeit()

    def act(self, obs):
        if self.forfeited or self.time >= TIME_BANK_SECONDS:
            return default_action(obs.to_call)
        action, ok = self._call(self.mod.choose_action, obs, self.memory)
        if ok and action not in obs.legal_actions:
            ok = False
            self.errors += 1
            if not self.first_error:
                self.first_error = f"illegal action {action!r}; legal {obs.legal_actions}"
        self._check_forfeit()
        return action if ok else default_action(obs.to_call)

    def hand_end(self, result):
        if not self.forfeited:
            self._call(self.mod.on_hand_end, result, self.memory)
            self._check_forfeit()


def play_half(bot_a, bot_b, seed, deals: int, half: int):
    """bot_a sits in seat 0 in half 1 and seat 1 in half 2. Returns
    (bot_a's chip net per deal, bot_a Seat, bot_b Seat)."""
    a, b = Seat(bot_a, "A"), Seat(bot_b, "B")
    seats = (a, b) if half == 1 else (b, a)
    a_seat = 0 if half == 1 else 1
    for s in seats:
        s.new_half()
    nets = []
    for d in range(deals):
        game = Game(make_deal(seed, d), button_for_deal(d), hand_index=d)
        while not game.is_terminal:
            seat = game.to_act
            game.apply(seats[seat].act(game.observation(seat)))
        for i, s in enumerate(seats):
            s.hand_end(game.hand_result(i))
        nets.append(game.result().nets[a_seat])
    return nets, a, b


def play_match(args):
    """One full duplicate match. Runs in a worker process."""
    bot_path, opp_path, seed, deals = args
    bot, opp = load_bot(bot_path), load_bot(opp_path)
    n1, a1, b1 = play_half(bot, opp, seed, deals, 1)
    bot, opp = load_bot(bot_path), load_bot(opp_path)  # fresh process per half in the tournament
    n2, a2, b2 = play_half(bot, opp, seed, deals, 2)
    paired = [(x + y) / BIG_BLIND for x, y in zip(n1, n2)]
    mean = statistics.fmean(paired)
    se = statistics.stdev(paired) / math.sqrt(len(paired))
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    forfeit_a, forfeit_b = a1.forfeited or a2.forfeited, b1.forfeited or b2.forfeited
    if forfeit_a:
        verdict = "LOSS (forfeit)"
    elif forfeit_b:
        verdict = "WIN (opp forfeit)"
    else:
        verdict = "WIN" if lo > 0 else "LOSS" if hi < 0 else "CHOP"
    return {
        "opp": os.path.basename(opp_path), "seed": seed, "mean": mean, "lo": lo, "hi": hi,
        "verdict": verdict, "time": max(a1.time, a2.time), "errors": a1.errors + a2.errors,
        "first_error": a1.first_error or a2.first_error,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bot", default=os.path.join(ROOT, "agent.py"))
    p.add_argument("--opponents", nargs="*",
                   default=sorted(glob.glob(os.path.join(ROOT, "baseline-bots", "baseline_*.py")))
                   + sorted(glob.glob(os.path.join(ROOT, "practice-bots", "practice_*.py"))))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--deals", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = p.parse_args()

    tasks = [(args.bot, opp, f"seed{s}", args.deals)
             for opp in args.opponents
             for s in range(args.seed_offset, args.seed_offset + args.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(play_match, tasks))

    print(f"\n{os.path.basename(args.bot)}: {len(tasks)} matches x {args.deals} deals "
          f"({time.time() - t0:.0f}s)\n")
    print(f"{'opponent':34} {'seed':7} {'bb/deal':>8}  {'95% interval':>18}  {'verdict':18} "
          f"{'max s/half':>10} {'errors':>6}")
    for r in results:
        print(f"{r['opp']:34} {r['seed']:7} {r['mean']:+8.3f}  [{r['lo']:+7.3f},{r['hi']:+7.3f}]  "
              f"{r['verdict']:18} {r['time']:10.2f} {r['errors']:6d}")
    by_opp: dict[str, list[float]] = {}
    for r in results:
        by_opp.setdefault(r["opp"], []).append(r["mean"])
    print(f"\n{'opponent':34} {'avg bb/deal':>11}")
    for opp, means in by_opp.items():
        print(f"{opp:34} {statistics.fmean(means):+11.3f}")
    print(f"{'FIELD AVERAGE':34} {statistics.fmean(r['mean'] for r in results):+11.3f}")
    wins = sum(r["verdict"].startswith("WIN") for r in results)
    print(f"\nwins {wins}/{len(results)}, worst time {max(r['time'] for r in results):.2f}s/half, "
          f"total errors {sum(r['errors'] for r in results)}")
    for r in results:
        if r["first_error"]:
            print(f"\nfirst error vs {r['opp']} {r['seed']}:\n{r['first_error']}")
            break


if __name__ == "__main__":
    main()
