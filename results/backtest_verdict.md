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

---
---

# BTC/USDT 4h + Funding Rate — Pre-registered Negative (Fourth)

**Date:** 2026-06-04
**Asset / timeframe:** BTC/USDT, 4h, spot, long-only (Flat/Long, no short, no
leverage). Funding from the BTC/USDT:USDT perpetual (information source only —
the account stays spot long-only).
**Hypothesis (fresh information test, NOT a tweak to prior models):** derivatives
positioning (funding rate + open interest) carries predictive information for BTC
spot direction. Fresh, never-touched locked holdout.

## Amendment to the pre-registration (made BEFORE any modeling)

Open-interest **history is capped at ~30 days on every free ccxt source**
(Binance & OKX ~30d; Bybit returns recent-only regardless of `since`). The
intersection of OHLCV + funding + OI is therefore only ~30 days (~180 4h bars) —
far too little for a feature warmup + walk-forward + 25% locked holdout. So the
combined **funding+OI hypothesis is UNTESTABLE on available data and is deferred
(untested, NOT rejected).** Funding-rate history, by contrast, goes back to 2019
and is clean. The narrowed, pre-registered hypothesis actually tested here:
**funding-rate positioning alone has predictive edge for BTC spot direction.**
Acceptance rule (unchanged): positive net Sharpe **AND** beats buy-and-hold on
the locked slice (read once), else terminate.

## Data (two sources, point-in-time aligned)

- **OHLCV 4h:** 14,808 real Binance bars, 2019-09-01 → 2026-06-04, deduped on
  PK, forming candle dropped. 0 duplicates; **1 single-bar gap** (2020-02-19
  12:00, known Binance outage — in the tuning region, immaterial to causality).
- **Funding:** 7,377 settlements, 2019-09-10 → 2026-06-04, deduped, rates in
  [−0.30%, +0.30%], **14.5% negative**. Settlement cadence varies over time
  (8h → shorter); the point-in-time forward-fill is cadence-agnostic.
- **Experiment domain = intersection** where both exist: ~14,752 4h candles.

## Point-in-time settled-funding alignment (the critical rule)

For a 4h candle closing at `t`, the funding feature is the **most recently
SETTLED** funding rate with settlement timestamp `≤ t` (`searchsorted(side=
"right") − 1`), forward-filled between settlements. A rate that settles **after**
`t` is never used; values are never interpolated using a future settlement.

**Leakage test (REQUIRED, new) — PASSES.** `tests/test_funding_pit_leakage.py`
proves the derivatives analogue of the existing OHLCV no-overlap test: perturbing
**any** funding value that settles after candle T's close leaves T's feature row
**byte-identical** (and the whole prefix ≤ T unchanged); later rows *do* move
(perturbation is real); bumping the in-force settled value *does* change T's row
(anti-vacuity); per-candle boundary math verified. Runs green alongside the
OHLCV causality test (full suite: 97 passed).

## Features + labels

6 causal OHLCV features + **2 funding features**: `funding_rate` (settled rate in
force at the close) and `funding_z` (trailing z-score over the last **42 candles
= 7 days**, `min_periods=window`, past-only — defines "extreme funding vs the
recent regime"). Labels unchanged: Flat/Long, N=1 (next-4h forward return), 24bps
round-trip hurdle. **Class balance: Long 36.75% / Flat 63.25%** — both ≫15%, no
imbalance warning. 13,851 rows aligned with labels.

## Time split

Aligned rows 13,851 → tuning = first 75% (10,388), **LOCKED = last 25% (3,462)**,
1-bar embargo at the seam. Barrier: tuning ends @ ts 1729108800000
(2024-10-16 20:00), embargo bar 2024-10-17 00:00 dropped, locked begins @
1729137600000 (2024-10-17 04:00). Locked untouched until the final read.

## Tuning (walk-forward, threshold 0.50, net of 24bps) — round trips: **1,543**

OOS 2020-08-20 → 2024-10-16, 8,655 bars, turnover 631×.

| Metric | Strategy (net) | Buy & Hold BTC |
|---|---:|---:|
| Total return | −38.17% | +478.87% |
| Net Sharpe | −1.32 | 1.01 |
| Sortino | −1.10 | 1.01 |
| Max drawdown | −45.59% | −77.04% |
| Win rate | 47.8% | — |
| Avg PnL/trade | −$2.47 | — |

**Funding-sign breakdown (round trips by funding regime AT ENTRY):**

| Regime | Trades | Total PnL | Avg PnL | Win% |
|---|---:|---:|---:|---:|
| **Negative** funding (short-squeeze / long-tradeable leg) | 193 | **+$102.61** | +$0.53 | 52.3% |
| **Positive** funding (crowded-long / avoid leg) | 1,350 | **−$3,915.61** | −$2.90 | 47.1% |

## LOCKED FINAL SLICE (single read, frozen config) — round trips: **578**

Full coverage: all 3,462 locked rows predicted leakage-free (each scored by a
model trained only on strictly-earlier, embargoed bars). OOS 2024-10-17 →
2026-06-03, turnover 232×.

| Metric | Strategy (net) | Buy & Hold BTC |
|---|---:|---:|
| Total return | **−18.26%** | −4.70% |
| Net Sharpe | **−2.13** | +0.17 |
| Sortino | −1.65 | +0.17 |
| Max drawdown | −20.38% | −49.84% |
| Win rate | 43.9% | — |
| Avg PnL/trade | −$3.03 | — |

**Funding-sign breakdown (locked):**

| Regime | Trades | Total PnL | Avg PnL | Win% |
|---|---:|---:|---:|---:|
| **Negative** funding | 106 | −$115.66 | −$1.09 | 48.1% |
| **Positive** funding | 472 | −$1,636.04 | −$3.47 | 43.0% |

## Verdict — REJECT (terminate)

Locked slice: **net Sharpe −2.13** and **−18.26% vs −4.70% B&H** — fails **both**
acceptance conditions. **Funding-alone hypothesis rejected.** Trade counts are
healthy (1,543 tuning / 578 locked), so this is a real out-of-sample negative,
not a small-sample artifact.

**Key finding (the reason the holdout matters):** the funding-sign asymmetry the
hypothesis predicted *appeared in tuning exactly as expected* — the
**negative-funding (short-squeeze) leg was the ONLY profitable bucket in tuning**
(+$102.61, 52.3% win), while the positive-funding (crowded-long) leg caused the
entire loss (−$3,915.61). But it **did NOT survive out-of-sample**: on the locked
slice even the negative-funding leg went **negative** (−$115.66, 48.1% win). The
apparent edge in the long-tradeable leg was **tuning-period noise**, caught by the
locked holdout. A **negative-funding-only variant was deliberately NOT built** —
selecting that leg after seeing it win in tuning, then testing it on the holdout,
would be fishing the holdout and would invalidate the lockbox. The honest outcome
stands: funding-alone rejected out-of-sample; funding+OI deferred as untestable.

The strategy's only favorable dimension is, again, shallower drawdown (−20% vs
−50% B&H) — a mechanical consequence of being in cash much of the time at high
turnover, not skill.

## Scope caveat

Specific to: funding-rate + OHLCV features as defined, N=1 Flat/Long labels,
XGBoost params as specified, threshold 0.50, BTC/USDT 4h, long-only spot, the
stated cost model, and a tuning window dominated by 2020–24 / locked window of
2024-10 → 2026-06. **Open interest was not tested** (deferred — no deep free
history). Not a claim about the derivatives-information family, OI specifically,
other assets/horizons/regimes, or a negative-funding-conditioned strategy (which
was intentionally left untested to preserve the holdout).

---
---

# Volatility-Targeted Sizing — Pre-registered Negative (Fifth & Final)

**Date:** 2026-06-04
**Asset / timeframe:** BTC/USDT, 4h, spot, long-only, no leverage. Same model,
features, labels (N=1, 24bps), XGB params, walk-forward, and time split as the
**fourth** experiment (the funding-feature model) — the SAME locked slice held
out. This is a **sizing layer on the identical signal**, not a re-tuned model.
**Hypothesis (tests ONE thing):** does volatility-scaled position sizing convert
the model's signal into risk-adjusted edge that fixed-fractional sizing was
hiding — or is any Sharpe improvement just variance compression that a sized
benchmark gets for free?

## Method (no engine change; rebalancing costs charged)

The existing backtester takes a single scalar `position_fraction` and never
rebalances a held position, so true vol-targeting cannot be expressed through it
by a parameter alone. Per the pre-registered decision, the FIXED curves come from
the **unchanged** `run_backtest`, while the SIZED model and SIZED buy-and-hold
curves are built by a **separate vol-targeting layer** that rebalances per bar
and charges every rebalance's traded notional through the **unchanged**
`CostModel.fill_cost`. The SIZED model and SIZED B&H use the SAME layer, SAME
cost model, and SAME frozen params, so the A/B and the dual benchmark are
apples-to-apples. The rebalancing drag is included by construction (and confirmed
charged: e.g. locked SIZED B&H added 1,376 rebalances, $50.9 cost).

**Point-in-time vol (leakage check PASSES).** Vol at bar T = trailing 30-bar
(5-day) RMS of 4h log returns, past-only, **no centering** (`sqrt(mean(r²))`).
`tests/test_realized_vol_pit_leakage.py` proves perturbing any bar after T leaves
T's vol byte-identical (and the whole prefix), a recent bar does move it
(anti-vacuity), and the estimator is uncentered — green alongside the funding and
OHLCV no-look-ahead tests.

**Sizing rule:** `position_fraction(T) = clip(target_vol / realized_vol(T), 0,
max_fraction)`, applied only when the model says Long. `max_fraction = 0.20` (the
existing per-trade / exposure cap — scales DOWN freely, never exceeds 20%, no
leverage). `target_vol` and `max_fraction` were chosen on the **TUNING region
only** and frozen before the locked slice: `target_vol = 0.002669 / 4h (~12.5%
ann.)`, anchored so median-vol Long bars hit the cap and high-vol bars scale down
(genuine redistribution — fraction p5/p50/p95 = 0.10 / 0.20 / 0.20). An
average-exposure-matched target_vol was rejected: with a 0.20 cap it forces
fraction ≡ 0.20 everywhere (SIZED ≡ FIXED), which would be no test at all.

## Round trips (prominent)

Tuning: FIXED model **1,543** / SIZED model **1,538**. Locked: FIXED model
**579** / SIZED model **575**. B&H = 1 entry each (plus per-bar rebalances on the
sized variants). Healthy counts — not a small-sample artifact.

## LOCKED FINAL SLICE — read once (OOS 2024-10-17 → 2026-06-04, 3,463 bars, full coverage, net 24bps)

| Curve | Total return | Net Sharpe | Sortino | Max drawdown | Turnover |
|---|---:|---:|---:|---:|---:|
| FIXED model | −18.27% | **−2.13** | −1.65 | −20.4% | 232× |
| SIZED model | −19.80% | **−2.52** | −1.90 | −21.0% | 224× |
| FIXED buy-and-hold | −4.37% | **+0.17** | +0.18 | −49.8% | 1× |
| **SIZED buy-and-hold** | −0.57% | **0.00** | 0.00 | **−12.3%** | 4× |

(Tuning showed the same ordering: FIXED model Sharpe −1.32, SIZED model −1.63,
FIXED B&H +1.01, **SIZED B&H +0.91** with maxDD compressed −77% → −22.9%.)

## Verdict — sizing is risk-shaping, NOT edge creation (three independent reads)

1. **Sizing made the model WORSE, not better — fails the A/B.** SIZED model Sharpe
   is *below* FIXED model on both tuning (−1.63 vs −1.32) and locked (−2.52 vs
   −2.13). There was no hidden edge for vol-targeting to surface; scaling a
   zero-signal position just adds rebalancing cost to a loser.
2. **Sized B&H beat the sized model — fails the dual benchmark.** SIZED B&H
   (Sharpe 0.00 locked / +0.91 tuning) dominates SIZED model (−2.52 / −1.63). If
   sizing conferred edge the sized *model* would beat the sized *benchmark*; it
   does the opposite.
3. **Vol-targeting's only real effect — drawdown compression — is free on naked
   long BTC.** SIZED B&H cut maxDD −50% → −12% (locked) and −77% → −23% (tuning)
   with **no model at all**. That is exactly the "variance compression a sized
   benchmark gets for free" the experiment was built to expose. And it is not
   even a free lunch on B&H: vol-targeting did not improve B&H Sharpe either
   (1.01 → 0.91 tuning; 0.17 → 0.00 locked) — de-risking into high vol (which for
   BTC often preceded recoveries) plus rebalancing drag costs return/Sharpe. It
   is a risk-preference trade, not alpha.

**Conclusion:** volatility-targeted sizing is a **risk-shaping tool (shallower
drawdowns), not an edge-creation tool.** It cannot rescue a model with no
directional signal, and any Sharpe-flattering it appears to produce is obtainable
on a pure long-BTC position without the model.

## Project termination (per the pre-registered commitment)

This is the **fifth and final** pre-registered experiment. Across all five —
(1) BTC 1h baseline, (2) the turnover-reduction threshold experiment, (3) SOL 1d
fresh asset/horizon, (4) BTC 4h funding-rate information, and (5) this
vol-targeted sizing layer — no configuration produced a tradeable edge on a
never-seen locked slice, on either the **directional** axis (signal) or the
**risk-adjusted** axis (sizing). Per the commitment registered before this run,
that result **terminates the project**: the premise — *a solo developer
extracting durable directional or risk-adjusted edge from OHLCV (+ derivatives
features) with an XGBoost Flat/Long classifier and this cost model* — is
**rejected**. The pipeline is sound and the negatives are honest (passing
leakage/causality tests, lockbox discipline held every time); the edge simply is
not there at this scope.

## Scope caveat

Specific to: this sizing rule (30-bar RMS vol, target_vol 0.002669/4h,
max_fraction 0.20), the fourth experiment's model/features/labels, BTC/USDT 4h,
long-only spot, the stated cost model, and the 2024-10 → 2026-06 locked window.
Not a claim that vol-targeting is useless in general (it demonstrably shapes
drawdown), nor about leveraged/multi-asset risk-parity sizing, other models, or
other regimes — only that, here, it creates no edge a sized benchmark lacks.

---
---

# BTC/USDT 1d + On-chain AdrActCnt — Pre-registered Negative (Sixth)

**Date:** 2026-06-10
**Asset / timeframe:** BTC/USDT, 1d, spot, long-only (Flat/Long, no short, no
leverage). On-chain data: CoinMetrics Community API `AdrActCnt` (daily active
addresses), information source only.
**Hypothesis (fresh information test):** on-chain activity carries predictive
information for BTC spot direction beyond OHLCV. Project reopened for this one
pre-registered experiment after the five-experiment termination.

## Pre-registered design (frozen before any results were seen)

- **Tuning window:** 2018-01-01 → 2023-01-01. **Locked holdout:** 2023-01-01 →
  **2025-06-01 (HARD CAP** — stored data extends to 2026-06 but the runner
  refuses to read past the cap, enforced by assertion).
- **Exactly three new features** appended after the unchanged 6 OHLCV features:
  `adr_zscore_28d` (28d rolling z of AdrActCnt, window ending D-1),
  `adr_mom_7d` (`AdrActCnt[D-1]/AdrActCnt[D-8] − 1`),
  `adr_price_diverge_28d` (`adr_zscore_28d` minus 28d trailing close z-score).
- **Point-in-time rule:** a feature row for trading day D uses only AdrActCnt
  dated **D-1 or earlier** (CoinMetrics publishes day D at end of day D UTC).
  Proven by `tests/test_onchain_pit_leakage.py`: perturbation invariance on a
  100-row sample, literal D-1 recompute, anti-vacuity, and a deliberate
  injected-leak check that the test catches. Full suite 105 passed.
- Everything else unchanged: labels (N=1, 24bps hurdle), XGB params,
  walk-forward (5 splits, embargo = label horizon), costs (10bps taker +
  2bps slippage per side), threshold 0.50, 20% fixed-fractional, long-only.
  Nothing was tuned on tuning results (no threshold/param search was run).
- **Acceptance (ALL four required):** holdout net Sharpe > 0; holdout net
  Sharpe > B&H Sharpe same period; avg PnL/trade > 0; trades ≥ 30. Any single
  failure = REJECT, no reruns.

## Tuning window (walk-forward OOS 2018-12-12 → 2022-12-30, 1,480 bars, net of 24bps)

| Metric | Strategy (net) | Buy & Hold BTC |
|---|---:|---:|
| Net Sharpe | **+0.21** (gross +0.58) | **+0.91** |
| Total return | +6.61% (gross +22.00%) | +382.07% |
| Max drawdown | −16.27% | −76.63% |
| Round trips | 285 (win rate 58.2%) | — |
| Avg PnL/trade | **+$2.32** | — |
| Turnover | 114.4× | — |

First positive-economics tuning run in project history (positive net Sharpe,
positive avg PnL/trade, >50% win rate) — noted, not acted on.

## LOCKED HOLDOUT (single read, frozen config; walk-forward within
## 2023-01-01 → 2025-06-01; OOS 2023-07-07 → 2025-05-26, 690 bars)

| Metric | Strategy (net) | Buy & Hold BTC |
|---|---:|---:|
| Net Sharpe | **+1.07** (gross +1.61) | **+1.63** |
| Total return | +15.10% (gross +22.80%) | +260.64% |
| Max drawdown | −5.27% | −28.10% |
| Round trips | 154 (win rate 53.9%) | — |
| Avg PnL/trade | **+$9.81** | — |
| Turnover | 61.5× | — |

## Verdict — REJECT (criterion 2 failed)

| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | Holdout net Sharpe > 0.0 | +1.07 | PASS |
| 2 | Holdout net Sharpe > B&H Sharpe (same period) | 1.07 vs **1.63** | **FAIL** |
| 3 | Holdout avg PnL/trade > 0 | +$9.81 | PASS |
| 4 | Holdout trade count ≥ 30 | 154 | PASS |

Per the locked rule, one failure = **REJECT. No reruns, no adjustments.**

**Key finding:** this is the strongest result the pipeline has ever produced —
the first configuration with genuinely positive out-of-sample economics
(net Sharpe +1.07, +$9.81/trade, costs NOT fatal: gross 1.61 → net 1.07).
On-chain activity does appear to carry usable information. But the bar is
beating the asset itself, and 2023-01 → 2025-06 was a strong BTC bull window:
buy-and-hold posted Sharpe 1.63 / +261%. A long-only filter that is in the
market only part-time could not keep up on the risk-adjusted axis, despite a
5× shallower drawdown (−5.3% vs −28.1%). The drawdown advantage is real but —
as in every prior experiment — partially mechanical (being in cash most bars).
A "beat Sharpe OR beat return with ≤ half the drawdown" criterion would have
read differently; that criterion was NOT pre-registered, so it does not count.
Honest negative under the registered rule.

## Scope caveat

Specific to: AdrActCnt alone (no other on-chain metrics), these three feature
definitions, N=1 Flat/Long labels at 24bps, XGBoost as specified, threshold
0.50, BTC/USDT 1d, long-only spot with the stated cost model, and a holdout
dominated by a bull regime. Not a claim about on-chain data generally, other
metrics (e.g. fees, transfer value, SOPR), other horizons, or regime-dependent
deployment — none of those were tested.

---
---

# ETH/USDT 1d + On-chain AdrActCnt — Pre-registered Negative (Seventh)

**Date:** 2026-06-10
**Asset / timeframe:** ETH/USDT, 1d, spot, long-only (Flat/Long, no short, no
leverage). On-chain data: CoinMetrics Community API `AdrActCnt` for **eth**.
**Hypothesis (pre-registered replication of the sixth experiment):** the
on-chain activity signal that produced positive OOS economics on BTC replicates
on a second asset. Identical features, costs, labels, walk-forward, threshold,
windows, and acceptance criteria — only the asset pair changed.

## Design notes

- Fetcher parameterized by asset (`--asset eth`), per-asset parquet caches; the
  BTC cache verified byte-identical (SHA256) before/after. The four leakage
  proofs re-run against the REAL cached ETH AdrActCnt series (100-row rule
  check, literal D-1 recompute, anti-vacuity, deliberate injection) — all
  green alongside the unchanged BTC tests (suite: 109 passed).
- Windows shared with the sixth experiment: tuning 2018-01-01 → 2023-01-01;
  locked holdout 2023-01-01 → **2025-06-01 hard cap** (data extends to 2026-06;
  the runner refuses to read past the cap). Nothing tuned on tuning results.

## Tuning window (walk-forward OOS 2018-12-12 → 2022-12-30, 1,480 bars, net of 24bps)

| Metric | Strategy (net) | Buy & Hold ETH |
|---|---:|---:|
| Net Sharpe | **+1.00** (gross +1.29) | **+1.16** |
| Total return | +65.30% (gross +86.67%) | +1,242.12% |
| Max drawdown | −16.09% | −79.30% |
| Round trips | 296 (win rate 58.8%) | — |
| Avg PnL/trade | **+$22.06** | — |
| Turnover | 120.7× | — |

The signal replicated on ETH in tuning, stronger than on BTC (net Sharpe 1.00
vs 0.21; avg PnL +$22.06 vs +$2.32).

## LOCKED HOLDOUT (single read, frozen config; walk-forward within
## 2023-01-01 → 2025-06-01; OOS 2023-07-07 → 2025-05-26, 690 bars)

| Metric | Strategy (net) | Buy & Hold ETH |
|---|---:|---:|
| Net Sharpe | **+0.25** (gross +0.68) | **+0.58** |
| Total return | +3.54% (gross +11.09%) | +37.03% |
| Max drawdown | −9.97% | −63.75% |
| Round trips | 154 (win rate 55.8%) | — |
| Avg PnL/trade | **+$2.30** | — |
| Turnover | 61.8× | — |

## Verdict — REJECT (criterion 2 failed)

| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | Holdout net Sharpe > 0.0 | +0.25 | PASS |
| 2 | Holdout net Sharpe > B&H Sharpe (same period) | 0.25 vs **0.58** | **FAIL** |
| 3 | Holdout avg PnL/trade > 0 | +$2.30 | PASS |
| 4 | Holdout trade count ≥ 30 | 154 | PASS |

Per the locked rule, one failure = **REJECT. No reruns, no adjustments.**

**Key finding:** the replication PARTIALLY held. For the second asset in a row,
the on-chain configuration posted positive out-of-sample economics on a locked
holdout (net Sharpe +0.25, +$2.30/trade, 55.8% win rate) — criteria 1, 3, 4
passed on both BTC and ETH. That consistency (positive net Sharpe, positive
per-trade PnL, >50% win rate, 2 assets × 2 windows = 4 independent OOS reads)
suggests AdrActCnt carries genuine, if modest, information. But the same
structural failure repeated: a long-only, part-time-invested filter could not
beat holding the asset on the risk-adjusted axis in windows where the asset
itself rallied (ETH holdout B&H Sharpe 0.58 vs 0.25; tuning-vs-holdout decay
1.00 → 0.25 also notable). Both experiments delivered far shallower drawdowns
(−10% vs −64% here), but drawdown is not a registered criterion. Honest
negative under the registered rule, both times.

## Scope caveat

Specific to: eth AdrActCnt alone, these three feature definitions, N=1
Flat/Long labels at 24bps, XGBoost as specified, threshold 0.50, ETH/USDT 1d,
long-only spot with the stated cost model, and the shared 2023-01 → 2025-06
holdout. The repeated criterion-2 failure pattern (signal real but below the
asset's own bull-window Sharpe) is an observation, not a tested hypothesis
about regime-conditioned deployment — that would require a new pre-registered
experiment.
