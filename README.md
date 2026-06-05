# Quant Trading Agent

A production-grade quantitative trading research pipeline built 
backtester-first, with a hard performance gate before any live 
infrastructure is built. The pipeline was used to run five 
pre-registered experiments testing whether a solo developer can 
extract directional or risk-adjusted edge from OHLCV and 
derivatives data using XGBoost. The answer, across all five, 
was no — and that is the result, not a failure.

## The Core Principle

> The backtester is a gate, not a step.

Phases 0–4 (data → features → labels → train → backtest) produce 
one verdict: does this beat buy-and-hold net of costs, 
out-of-sample? Phases 5–8 (paper engine, risk gate, dashboard, 
live) are only worth building if that verdict is yes.

The gate returned no across five pre-registered experiments. The 
live execution path (Phase 8) was never enabled. The paper engine, 
risk gate, and dashboard were built as a pure engineering exercise 
with the live path hard-gated behind `LIVE_TRADING_CONFIRMED=true`.

## Architecture
core/          position manager, exchange wrapper, trading daemon,
risk engine
ml/            features, labels, walk-forward training
backtest/      event-driven backtester, cost model, metrics
web/           read-only FastAPI dashboard
config/        pydantic-settings, all secrets from env vars
tests/         100 tests, all passing
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
itself. Three independent guarantees are enforced and tested:

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

A pipeline with lookahead leakage produces beautiful backtests. 
These tests prove the negative results are real.

## The Five Experiments

All five were pre-registered (hypothesis and acceptance rule stated 
before any data was touched), evaluated on a locked holdout read 
exactly once, and accepted as written.

| # | Config | Tuning | Locked | Verdict |
|---|--------|--------|--------|---------|
| 1 | BTC/USDT 1h, raw signal | −38% / Sharpe −1.32 | −47% / −8.03 | REJECT |
| 2 | BTC/USDT 1h, isotonic calibration + turnover-anchored threshold | +0.02% / 0.04 | 0 trades (null by construction) | REJECT |
| 3 | SOL/USDT 1d | +25% / 0.50 | −5.71% / −0.31 | REJECT |
| 4 | BTC/USDT 4h + funding rate feature (PIT-aligned, leakage-tested) | −38% / −1.32 | −18% / −2.13 | REJECT |
| 5 | BTC/USDT 4h, vol-targeted sizing (4-way A/B vs fixed + sized B&H) | FIXED −38%/−1.32, SIZED −38%/−1.63, FIXED B&H +479%/1.01, SIZED B&H +44%/0.91 | FIXED −18%/−2.13, SIZED −20%/−2.52, FIXED B&H −4%/0.17, SIZED B&H −0.6%/0.00 | REJECT |

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

Full details, cost reconciliation, and regime caveats for all five 
experiments are in `results/backtest_verdict.md`.

## Why Negative Results Are the Credential

A pipeline with lookahead leakage or data snooping produces 
positive results almost automatically. The tests above prove these 
results are clean. The consistent finding across two assets, two 
timeframes, an orthogonal information source, and a risk-adjusted 
sizing layer is itself a result: a tree model on OHLCV and 
publicly-available derivatives data does not have a durable edge 
net of costs, at these horizons, for a solo developer without 
co-location or proprietary data.

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
pytest                           # 100 tests, all should pass
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
