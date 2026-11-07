"""
show_reads.py - what Tulip believes about each opponent after a half.

    python tools/show_reads.py [--deals 1000]

Sanity check for bot/reads.py: a nit should show a high fold rate, a calling
station a low one, a maniac low honesty, and so on.
"""

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import run_matches as rm  # noqa: E402
from bot import reads  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--deals", type=int, default=1000)
    args = p.parse_args()
    opps = sorted(glob.glob(os.path.join(ROOT, "baseline-bots", "baseline_*.py"))) \
        + sorted(glob.glob(os.path.join(ROOT, "practice-bots", "practice_*.py")))
    print(f"{'opponent':30} {'fold_rate':>9} {'honesty':>8} {'pf_raise':>8} {'pf_fold':>7} {'conf':>5}")
    for opp in opps:
        _, a, _ = rm.play_half(rm.load_bot(os.path.join(ROOT, "agent.py")), rm.load_bot(opp),
                               "reads", args.deals, 1)
        m = a.memory
        print(f"{os.path.basename(opp):30} {reads.value(m, 'fold_rate'):9.2f} "
              f"{reads.value(m, 'honesty'):8.2f} {reads.value(m, 'pf_raise'):8.2f} {reads.value(m, 'pf_fold'):7.2f} "
              f"{reads.confidence(m):5.2f}")


if __name__ == "__main__":
    main()
