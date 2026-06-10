# Quant Trading Agent

A production-grade quantitative trading research pipeline built 
backtester-first, with a hard performance gate before any live 
infrastructure is built. The pipeline was used to run eight 
pre-registered experiments testing whether a solo developer can 
extract directional or risk-adjusted edge from OHLCV, 
derivatives, on-chain, and cross-exchange data using XGBoost. 
The answer, across all eight, was no ACCEPT — and that is the 
result, not a failure. Experiments 6–7 (on-chain active-address 
features, BTC then an ETH replication) produced the project's 
first genuinely positive out-of-sample economics on locked 
holdouts, and were still rejected under the pre-registered rule: 
a long-only filter never beat the asset's own Sharpe in a bull 
window. Experiment 8 (Coinbase-premium features at 1h) was 
rejected at the tuning gate — a real gross edge (+1.68 Sharpe) 
destroyed by costs at 765× hourly turnover — with its holdout 
deliberately left unread.

## The Core Principle

> The backtester is a gate, not a step.

Phases 0–4 (data → features → labels → train → backtest) produce 
one verdict: does this beat buy-and-hold net of costs, 
out-of-sample? Phases 5–8 (paper engine, risk gate, dashboard, 
live) are only worth building if that verdict is yes.

The gate returned no across eight pre-registered experiments. The 
live execution path (Phase 8) was never enabled. The paper engine, 
risk gate, and dashboard were built as a pure engineering exercise 
with the live path hard-gated behind `LIVE_TRADING_CONFIRMED=true`.

## Architecture
core/          position manager, exchange wrapper, trading daemon,
risk engine
ml/            features, labels, walk-forward training
backtest/      event-driven backtester, cost model, metrics
src/onchain/   CoinMetrics + dual-exchange fetchers, PIT feature
builders (on-chain D-1 lag; Coinbase premium T-inclusive)
web/           read-only FastAPI dashboard
config/        pydantic-settings, all secrets from env vars
tests/         114 tests, all passing
scripts/       fetch, backtest, paper trading, halt reset
results/       backtest_verdict.md — the full experiment record

**Stack:** Python 3.13, XGBoost, CCXT, FastAPI, SQLite (WAL).  
**Execution:** spot only, long-only, no shorting, no leverage 
anywhere in the codebase.  
**Single chokepoint:** every order routes through 
`risk_check(order)` before reaching any executor — enforced by 
tests that assert the executor is never called when `risk_check` 
rejects.

## Leakage Guarantees

The pipeline's most important property is that it cannot lie to 
itself. Five independent guarantees are enforced and tested:

**OHLCV no-overlap test:** perturbing any bar after time T leaves 
T's feature row byte-identical. Perturbing any bar before T leaves 
T's label unchanged. Proven by `tests/test_features_labels.py`.

**Funding rate point-in-time test:** for each 4h candle closing at 
T, the funding feature uses only the most recently *settled* rate 
at or before T. Perturbing any settlement after T's close leaves 
T's feature row byte-identical. Proven by 
`tests/test_funding_pit_leakage.py`.

**Realized volatility point-in-time test:** the vol estimate at T 
uses only data through T's close, uncentered. Perturbing any bar 
after T leaves T's vol estimate byte-identical. Proven by 
`tests/test_realized_vol_pit_leakage.py`.

**On-chain D-1 point-in-time test:** CoinMetrics publishes day 
D's active-address count at end of day D UTC, so a feature row 
for trading day D may use only values dated D-1 or earlier. 
Proven four ways on a 100-row sample (for both BTC and real ETH 
data): perturbing any value dated D or later leaves D's feature 
row byte-identical; a literal recompute from D-1-or-earlier 
values matches; bumping the D-1 value moves the row 
(anti-vacuity); and a deliberately injected AdrActCnt[D] leak is 
caught. Proven by `tests/test_onchain_pit_leakage.py`.

**Coinbase-premium T-inclusive test:** the premium at T is 
computed from both exchanges' closed bars at T, so T itself is 
usable; the proof is that the 168h z-score and 24h momentum 
windows never reach past T. Same four proofs (100-row 
perturbation invariance for values dated strictly after T, 
literal recompute, anti-vacuity on all three features, deliberate 
T+1 injection caught). Proven by 
`tests/test_coinbase_premium_pit_leakage.py`.

A pipeline with lookahead leakage produces beautiful backtests. 
These tests prove the negative results are real.

## The Eight Experiments

All eight were pre-registered (hypothesis and acceptance rule stated 
before any data was touched). Seven were evaluated on a locked 
holdout read exactly once; the eighth was rejected at the tuning 
gate with its holdout deliberately left unread. Every verdict was 
accepted as written.

| # | Config | Tuning | Locked | Verdict |
|---|--------|--------|--------|---------|
| 1 | BTC/USDT 1h, raw signal | −38% / Sharpe −1.32 | −47% / −8.03 | REJECT |
| 2 | BTC/USDT 1h, isotonic calibration + turnover-anchored threshold | +0.02% / 0.04 | 0 trades (null by construction) | REJECT |
| 3 | SOL/USDT 1d | +25% / 0.50 | −5.71% / −0.31 | REJECT |
| 4 | BTC/USDT 4h + funding rate feature (PIT-aligned, leakage-tested) | −38% / −1.32 | −18% / −2.13 | REJECT |
| 5 | BTC/USDT 4h, vol-targeted sizing (4-way A/B vs fixed + sized B&H) | FIXED −38%/−1.32, SIZED −38%/−1.63, FIXED B&H +479%/1.01, SIZED B&H +44%/0.91 | FIXED −18%/−2.13, SIZED −20%/−2.52, FIXED B&H −4%/0.17, SIZED B&H −0.6%/0.00 | REJECT |
| 6 | BTC/USDT 1d + on-chain AdrActCnt, 3 features (D-1 PIT, leakage-tested) | +6.6% / 0.21 | +15.1% / **1.07** vs B&H 1.63 | REJECT |
| 7 | ETH/USDT 1d + on-chain AdrActCnt (pre-registered replication of 6) | +65.3% / 1.00 | +3.5% / **0.25** vs B&H 0.58 | REJECT |
| 8 | BTC/USDT 1h + Coinbase-premium, 3 features (T-inclusive PIT) | gross +26.9%/**+1.68**, net −41.3%/−2.54, 765× turnover | **not run** — rejected at tuning gate | REJECT |

**Experiment 5 dual-benchmark finding:** vol-targeted sizing made 
the model *worse* than fixed sizing on both tuning and locked 
(−2.52 vs −2.13 Sharpe). The sized buy-and-hold benchmark beat the 
sized model on every metric, confirming that any Sharpe improvement 
from vol-targeting is free variance compression, not model edge.

**Experiment 4 funding-sign finding:** the negative-funding 
(short-squeeze) leg was the only profitable bucket in tuning 
(+$103, 52% win) — exactly as hypothesized. On the locked slice it 
went negative (−$116, 48% win). The apparent edge was 
tuning-period noise, caught by the holdout. A 
negative-funding-only variant was deliberately not built, as that 
would have been fishing the holdout.

**Experiment 8 cost-structure finding:** the cleanest gross/net 
separation of the project: the Coinbase premium has real gross 
edge at 1h (gross Sharpe +1.68, matching buy-and-hold's), but 765× 
turnover and the ~68pp gross-to-net gap annihilate it (net −2.54, 
avg PnL/trade −$2.20 across 1,876 OOS trades). Rejected at the 
tuning gate without reading the holdout — with per-trade economics 
that negative, criterion 3 cannot pass, and spending the locked 
slice would have bought nothing. Across eight experiments the 
pattern is consistent: this cost model admits daily-horizon 
signals and annihilates hourly ones.

**Experiments 6–7 on-chain finding:** the first configuration with 
genuinely positive out-of-sample economics — positive net Sharpe, 
positive avg PnL/trade (+$9.81 BTC, +$2.30 ETH), and >50% win 
rates on every read (2 assets × tuning + holdout = 4 independent 
OOS reads), with costs no longer fatal (BTC holdout gross 1.61 → 
net 1.07). Three of four acceptance criteria passed on both 
holdouts. Both still REJECT on the same criterion: a long-only, 
part-time-invested filter never beat the asset's own Sharpe in 
holdout windows that were bull runs (BTC B&H +261%, ETH +37%). 
The 5× shallower drawdowns (−5% vs −28%; −10% vs −64%) were not a 
registered criterion and do not count. A regime-conditioned 
variant was not built — both holdouts are spent, and testing one 
against them would be fishing.

Full details, cost reconciliation, and regime caveats for all 
eight experiments are in `results/backtest_verdict.md`.

## Why Negative Results Are the Credential

A pipeline with lookahead leakage or data snooping produces 
positive results almost automatically. The tests above prove these 
results are clean. The consistent finding across three assets, 
three timeframes, two orthogonal information sources (derivatives 
positioning and on-chain activity), and a risk-adjusted sizing 
layer is itself a result: a tree model on OHLCV and 
publicly-available data does not produce an edge that beats 
holding the asset, net of costs, at these horizons, for a solo 
developer without co-location or proprietary data — even when the 
signal itself is real, as the on-chain experiments showed.

The more important demonstration is methodological: every 
experiment was pre-registered, the acceptance rule was committed 
before results were seen, the locked holdout was read exactly once, 
and the verdict was accepted — including when tuning data showed 
apparent edge that died out-of-sample.

## Risk Engine

`core/risk.py` implements a real circuit-breaker:
- Trips `SYSTEM_HALT` on rolling 24h drawdown ≤ −3%, N 
  consecutive exchange failures, or stale heartbeat
- When halted: blocks BUYs, allows SELLs (flatten to cash routes 
  through the same `risk_check` chokepoint — no bypass path)
- HALT never self-clears; manual reset only via 
  `scripts/reset_halt.py --confirm`
- Tested: a −3% drawdown seeds the halt from the real evaluation 
  path, then asserts a 0.99-confidence long is refused and the 
  position is flattened

## Dashboard

Read-only FastAPI dashboard at `localhost:8000`. Opens SQLite in 
`?mode=ro` (engine-level write rejection as defense-in-depth). 
Displays mode, position, last 20 executions with fee/slippage, 
rolling 24h PnL and drawdown, HALT state and reason, heartbeat. 
The only state-changing endpoint is `POST /reset-halt`, which 
requires explicit `confirm` and routes through the same guarded 
`clear_halt` path as the manual script.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # fill in your exchange keys
pytest                           # 114 tests, all should pass
```

To run the paper daemon (no live execution without 
`LIVE_TRADING_CONFIRMED=true`):
```bash
python scripts/run_paper.py
```

To view the dashboard:
```bash
uvicorn web.app:app --host 127.0.0.1 --port 8000
```

## What Is Not Built

`scripts/run_live.py` exists as a scaffold but live execution is 
not implemented. It requires `LIVE_TRADING_CONFIRMED=true` and 
`--i-understand-the-risks`. It will not be enabled until a 
different model clears the Phase 3 gate on a clean holdout.
