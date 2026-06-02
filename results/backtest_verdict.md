# BTC/USDT 1h — Backtest Verdict (Negative Finding)

**Date:** 2026-06-02
**Asset / timeframe:** BTC/USDT, 1h, spot, long-only (Flat/Long, no short)
**Data:** 12,959 real Binance 1h bars, ~18 months (2024-12-09 → 2026-06-02 09:00 UTC),
read-only public OHLCV.
**Pipeline:** causal features → Flat/Long labels (N=1, hurdle 0.24% = 20 bps
round-trip + 2 bps/leg slippage) → walk-forward XGBoost (5 splits, embargo =
label horizon) → event-driven spot backtester (10 bps/side taker + 2 bps/side
slippage, next-bar-open execution, 20% fixed-fractional sizing, minNotional
gate, no pyramiding).

No strategy, model, or label changes were made across any of the four runs
below. Runs (2)–(4) are pure post-hoc measurement/threshold analysis on the
**same** OOS prediction vector produced by run (1).

---

## (1) Raw baseline — signal = P(Long) ≥ 0.50, net of costs

Out-of-sample: 10,755 bars, 1,496 round trips, turnover 595×.

| Metric | Strategy (net) | Buy & Hold |
|---|---:|---:|
| Total return | **−47.18%** | −11.64% |
| Sharpe (annualized) | −8.03 | −0.03 |
| Max drawdown | −47.76% | −50.08% |
| Win rate | 32.0% | — |

**No clear edge.** The strategy loses heavily and underperforms holding BTC on
both net return and Sharpe. Accuracy (63.6%) is *below* the always-Flat rate
(77.6%), confirming accuracy is meaningless here — judged by net-of-cost return.

---

## (2) Gross-vs-net decomposition (same fills, costs added back)

Gross equity reconstructed by adding cumulative costs back along the identical
executed path (same fills, sizing, timing) — an exact decomposition, not a
re-simulation.

| Metric | Gross (no cost) | Net (real) |
|---|---:|---:|
| Total return | **+6.23%** | −47.18% |
| Sharpe (annualized) | **+1.13** | −8.03 |
| Final equity ($10k start) | $10,623.20 | $5,282.29 |

**Cost reconciliation — clean, no discrepancy:**
- Total costs paid: $5,340.90 = **53.41%** of starting equity.
- Gross − Net total-return gap: **53.41%**.
- Check A (total costs vs traded_notional × per-side rate): diff 0.0000%.
- Check B (return gap vs turnover × per-side rate × avgEq/startEq): diff **~0 pp**.
- Check C (return gap vs total_costs / start equity): diff **~0 pp**.
- No cost double-counted; no mistimed exit (sells = buys − 1; one position open
  at window end, expected).

**Finding:** there *is* a slim gross edge (+6.2%), but transaction costs
(53.4% drag at 595× turnover) overwhelm it entirely.

---

## (3) Pre-registered experiment v1 — ILL-POSED, null by construction

Rule attempted: fit isotonic calibrator on a held-out fold, then pick the
calibrated-probability threshold that cuts round-trip turnover **another 10×**
from the calibrated baseline.

**Why it was ill-posed:** isotonic calibration *already* collapsed turnover
~70× on its own (calibrated P(Long) rarely exceeds 0.5, because the raw model
is systematically over-confident). The "another 10×" target (≤ 2 round trips)
was statistical noise; even the most extreme threshold (p ≥ 0.99) only reached
10 round trips.

**Locked-slice result: 0 trades, 0.00% return, Sharpe 0.00** — the frozen
p ≥ 0.99 config never fired on the locked data. **Null by construction**, not
evidence of edge or its absence. Discarded; rule replaced in run (4).

---

## (4) Pre-registered experiment v2 — CORRECTED, locked slice read once

Same time split as v1: 10,755 OOS bars → tuning = first 75% (8,066) |
**LOCKED = last 25% (2,689)**. Time barrier: tuning ends @ ts 1770696000000,
locked begins @ 1770699600000.

Corrected rule (replaces the calibrated-baseline rule): on the **tuning fold
only**, pick the model-score threshold whose round-trip count is closest in
absolute terms to (raw-0.50 baseline ÷ 10) — testing whether cutting the
cost-bleed mechanism ~10× recovers the gross edge. Ranking by raw vs isotonic
score is identical under monotonic calibration, so the raw score was used.

- Raw p ≥ 0.50 baseline (tuning): 1,086 round trips → target 108.6.
- **Chosen threshold: p ≥ 0.755** → 105 round trips on tuning (10.3× reduction).
  Frozen before the locked slice was read.

**Tune-eval (tuning fold, net of costs):** 105 round trips, turnover 41.75×,
net **−4.88%** / Sharpe **−2.30** vs B&H −12.19% / −0.13; win rate 36.2%,
avg PnL/trade −4.71, maxDD −6.07% (B&H −50.08%).

**LOCKED FINAL SLICE (single read, same frozen threshold, net of costs):**

| Metric | Strategy (net) | Buy & Hold |
|---|---:|---:|
| Round trips | 27 (54 fills) | — |
| Turnover | 10.83× | — |
| Net return | **−1.29%** | +1.18% |
| Net Sharpe | **−4.71** | +0.30 |
| Win rate | 33.3% | — |
| Avg PnL/trade | **−4.71** (−4.76 on locked) | — |
| Max drawdown | −1.29% | −14.85% |

Lockbox discipline held: threshold chosen on tuning only, locked slice read
exactly once, no re-tuning after the lock was opened.

---

## Verdict

**No tradeable edge at this horizon / feature set.** The gross edge is real but
far too small to clear costs. Cutting turnover ~10× (run 4) did not recover it:
per-trade economics stay negative (avg PnL/trade ≈ −4.7), so keeping only the
highest-confidence longs still loses, and on the never-seen locked slice the
strategy underperforms buy-and-hold (−1.29% vs +1.18%). The only favorable
dimension is shallower drawdown — a mechanical consequence of being in the
market less, not skill (equivalent to holding cash).

## Scope caveat

This conclusion is specific to: **this configuration** (current causal feature
set, N=1 forward-return Flat/Long labels, XGBoost params as specified, 0.755
threshold), **BTC/USDT only**, a **~18-month, largely down/chop OOS window**,
and a **long-only spot** account with the stated cost model. It is **not** a
claim about the strategy family, other assets, other horizons, other feature
sets, or other market regimes. A different horizon, richer features, or a
trending regime could change the result; none of those were tested here.
