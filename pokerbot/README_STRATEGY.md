# How the bot plays

Heads-Up Short Deck 5-card PLO. Everything below was measured, not assumed; the
scripts that measured it are in `dev/` (not shipped in the Docker image).

## Decision layers (`agent.py: choose_action`)

1. **Safety valve → CFR blueprint.** If our own *luck-adjusted* result is
   significantly negative (mean + 2·stderr < 0 after ≥ 50 hands), we are being
   outplayed and whatever we think we know is wrong: play the blueprint for the
   rest of the half. Outranks every other layer.
2. **Confirmed fixed bot → exact best response.** `pokerbot/fixedbot.py`
   reconstructs which actions were legal at every opponent decision and tests
   each against always-pot (the "all-in bot"), always-half, calling station and
   check-fold; one deviation eliminates a rule for good. A *timed-out* opponent
   (the judge plays check-or-fold for it) is detected from a recent window.
   Uniform random play needs frequency evidence **and** proof that its raises
   ignore its cards (showdown hands no stronger when it raised) — frequencies
   alone mistook the CFR blueprint for random and lost 2.2 bb/deal.
3. **Degenerate frequencies → heuristic with tuned exploit deltas** (never folds /
   always folds / bets everything).
4. **Everyone else → hybrid default.** The CFR blueprint, unless the opponent is
   shown to be a *loose caller* (rarely folds post-flop, or folds little and bets
   a lot), in which case the value-heavy heuristic plus the **leak exploit**
   (bet more on streets where they over-fold, call tighter vs passive bettors).

Why hybrid: the blueprint beats opponents who fold correctly (TAG, rock, an
adaptive bot, a CFR bot, a trained adversary, our own previous version by
+2.6 bb/deal); the heuristic beats loose callers (LAG by 1.1, jekyll by 4.3).

## The pieces

| piece | what it is |
|---|---|
| `pokerbot/cards.py` | 630 × 7140 pair×triple score table. Exact vs the reference evaluator on all 376,992 hands. |
| preflop table | Equity vs a random hand for all 376,992 hands, 282M duels over 19,998 suit-isomorphic classes. |
| postflop strength | Share of a fixed opponent pool we beat now, mapped to equity by a fitted curve. |
| heuristic | Equity vs pot odds for calls, percentile for aggression; thresholds by cross-entropy search. |
| blueprint | External-sampling MCCFR, linear discounting, 150M iterations, exact betting tree × 96 buckets. |
| `pokerbot/luck.py` | All-in adjustment (zero-knowledge AIVAT): an all-in hand is scored by exact equity × pot, not by the runout. Unbiased; 35–65% less variance vs competent bots. Verified exact against `evaluator.py`. |
| `pokerbot/opponent.py` | Frequencies, per-street fold-to-our-bet, showdown card dependence, archetype scores, leak exploit, loose-caller test. |
| plumbing | Raw ASGI + orjson + httptools/uvloop. Same HTTP contract as the stock server. |

## Measured results (paired arena, 4 × 1000-deal duplicate matches per opponent)

| | before this round (v1) | now |
|---|---|---|
| points vs 18-bot field | 15.0 | **17.5** |
| mean bb/deal | +5.36 | +5.40 |
| vs CFR blueprint | −3.29 (LOSS) | −0.02 (chop) |
| vs trained adversary (best response to our heuristic) | +0.27 (chop) | +1.19 (WIN) |
| vs TAG / rock / adapter | chop / chop / win | win / win / win |
| real HTTP judge vs v1, 1000 deals | | **+2.84, CI [+1.65, +4.02], WIN** |
| real HTTP judge vs bundled random | | +17.85, CI [+16.0, +19.7], WIN |
| time bank, Linux container, 1000 hands | | 0.09–0.23 s of 8 s |

Field: the 5 baselines, TAG, LAG, maniac, rock, all-in, jekyll (half maniac / half
station), times-out, and the adaptive bots in `dev/adaptive.py` (adapter learns
our fold rates and bet strength; shifter and turncoat change style mid-half),
the frozen previous version (`dev/snap`), the CFR blueprint, and a trained
best response to our heuristic.

Tried and kept out: playing the blueprint everywhere (17.5 points but −0.4 bb/deal
vs hybrid: it gives back 4 bb/deal to loose bots); a looser valve (z = 0.5) —
helps vs strong bots, costs vs LAG/maniac/turncoat, dominated by the hybrid; a
bandit over strategies — with ~8 bb/hand of noise it would need ~1000 hands to
tell a 1 bb/deal difference, so a measured prior (the loose-caller test) beats it.

## Rebuilding / re-measuring

```
python dev/build_preflop.py          # preflop equity (~90 s on 11 workers)
python dev/build_calibration.py      # postflop strength -> equity curve
python dev/build_dealbank.py         # 1.5M deals with buckets (~6 min)
python dev/cfr.py --iters 150000000  # blueprint (~20 min on 22 threads)
python dev/cfr_br.py --opp allin     # best responses: allin half station checkfold random
python dev/ab.py                     # paired A/B against the field
python dev/valve_test2.py            # safety-valve variants
python dev/tune_leak.py              # leak-exploit magnitudes (held-out check)
python dev/run_match.py agent.py dev/snap/agent_v1.py --deals 1000 --restart
```
