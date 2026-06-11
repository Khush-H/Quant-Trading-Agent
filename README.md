# Quant Trading Agent

A machine learning trading pipeline built to answer one question honestly:
can a gradient-boosted classifier extract executable alpha from crypto spot
markets, net of realistic transaction costs?

**The answer, across nine pre-registered experiments, is no.**

That is not a failure of engineering. It is the engineering working correctly.

---

## What This Project Actually Is

Most student trading projects optimise until the backtest looks good, then
stop. This one was built differently. The core constraint was established
before a single experiment ran: **the backtester is a gate, not a step.**
No live infrastructure would be built until the gate returned a positive
verdict on a locked, unread holdout. It never did. The infrastructure was
built anyway — as a pure engineering exercise — and the gate result was
accepted.

The result is a codebase that demonstrates something more useful than a
profitable backtest: a system explicitly designed to resist being gamed,
including by its own creator.

---

## Architecture

```
Data Layer          → CCXT (Binance/Coinbase), CoinMetrics on-chain API,
                      SQLite WAL, atomic parquet writes

Feature Pipeline    → Causal-only OHLCV features, on-chain active address
                      metrics, cross-exchange Coinbase Premium; all with
                      explicit point-in-time leakage tests per data source

Labels              → 2-class (Flat/Long), 24bps round-trip hurdle,
                      no-overlap leakage proof

Backtester          → Event-driven walk-forward, fills at t+1 open,
                      embargo gap = label horizon, cost model: 10bps taker
                      + 2bps slippage per side, gross/net Sharpe decomposition

Risk Engine         → 24h rolling drawdown HALT (≤ −3%), consecutive
                      exchange-failure HALT, stale-heartbeat HALT;
                      never self-clears; manual reset only

Paper Daemon        → Hourly execution loop, drops forming candle, scores
                      only closed bars, fills via same CostModel as
                      backtester (paper/backtest agree by construction)

Dashboard           → Read-only FastAPI; SQLite opened in mode=ro at engine
                      level; SELECT-only queries proven by connection-spy
                      test; single state-changing endpoint requires explicit
                      confirm flag and routes through the same guarded
                      clear_halt as the manual reset script

Live Gate           → Phase 8 hard-gated behind LIVE_TRADING_CONFIRMED=true
                      and --i-understand-the-risks; never unlocked
```

**120 tests passing.** Every invariant has a test. Every leakage proof is
adversarial — it injects a deliberately broken builder and asserts the test
catches it.

---

## The Single Chokepoint Invariant

Every order in the system — paper, backtest, and live — routes through a
single `risk_check()` function before reaching any executor. There is no
bypass path. This is proven by test: the executor is unreachable without
passing through `risk_check()`. The HALT state blocks BUYs and allows SELLs
(flatten-to-cash), and HALT never self-clears.

This is the same design principle as a security-critical system with a single
authentication boundary. The trading risk engine and an access control gateway
are solving the same structural problem.

---

## The Nine Experiments

All pre-registered. All run on locked holdouts read exactly once. All accepted
as written.

| # | Asset / TF | Information Class | Net Sharpe | Verdict |
|---|-----------|-------------------|-----------|---------|
| 1 | BTC 1h | OHLCV raw | −8.03 | REJECT — cost drag |
| 2 | BTC 1h | OHLCV + calibration | −4.71 | REJECT — cost drag |
| 3 | SOL 1d | OHLCV | −0.31 | REJECT — sign flip |
| 4 | BTC 4h | OHLCV + funding rate | −2.13 | REJECT — OOS collapse |
| 5 | BTC 4h | Vol-targeted sizing | −2.52 | REJECT — sizing adds no edge |
| 6 | BTC 1d | On-chain (AdrActCnt) | +1.07 | REJECT — below B&H (1.63) |
| 7 | ETH 1d | On-chain (AdrActCnt) | +0.25 | REJECT — below B&H (0.58) |
| 8 | BTC 1h | Coinbase Premium | −2.54 | REJECT at tuning gate |
| 9 | LINK 1d | Coinbase Premium | −0.33 | REJECT at tuning gate |

**The most important number in this table is not the Sharpe.** It is that
Experiments 6 and 7 — the on-chain experiments — posted positive net Sharpe,
positive avg PnL/trade, and above-50% win rates across four independent
out-of-sample reads on two assets. The signal exists. The strategy cannot beat
a passive benchmark in a sustained bull market, which is the correct and honest
finding.

**The pre-registration protocol** required a stated hypothesis and acceptance
rule before results were seen. One attempt was made during the project to
change the acceptance benchmark after seeing a near-miss result. It was caught,
interrogated, and rejected. The original criteria stood. The record of that
exchange is in the commit history.

---

## Security-Relevant Design Decisions

For readers coming from a cybersecurity background, the engineering decisions
that matter most are not the ML components:

**Tamper-evident audit trail.** Every experiment result is committed to
`results/backtest_verdict.md` before any subsequent work begins. The git
history is the pre-registration record. Rewriting it would be detectable.

**Leakage as an adversarial problem.** Each new data source required a
dedicated leakage test with four components: a rule check, a literal
recompute, an anti-vacuity proof (confirming the feature actually consumes
the data), and a deliberate injection test (confirming the test catches the
exact failure it guards against). A test that only checks the happy path is
not a test.

**Read-only enforcement at the engine level.** The dashboard database
connection opens SQLite in `?mode=ro`. This is not application-level
enforcement — it is enforced by the database engine itself and proven by a
connection-spy test that attempts a write and asserts it raises.

**No self-clearing failure states.** The HALT condition requires manual reset
via a script that demands an explicit `--confirm` flag. Automatic recovery
from a failure state is a security anti-pattern. The system fails closed.

---

## Stack

Python 3.13 · XGBoost · scikit-learn · CCXT · FastAPI · SQLite (WAL) ·
isotonic calibration · pandas · pytest (120 tests)

---

## What Was Not Built

Phase 8 (live execution) was scaffolded and hard-gated. The gate never
opened. The live infrastructure exists as an engineering exercise. No real
capital was deployed. No API keys with withdrawal permissions were ever used.

---

## Setup

```bash
git clone https://github.com/Khush-H/Quant-Trading-Agent.git
cd Quant-Trading-Agent

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env   # no API keys required for backtesting

pytest                 # 120 tests, all should pass
```

No exchange API keys are required to run the backtester or fetch
historical data. Binance public OHLCV and CoinMetrics Community API
endpoints are unauthenticated. The Coinbase Exchange candle endpoint
used for premium features is also public.

Live execution (Phase 8) requires exchange API keys and will not run
without `LIVE_TRADING_CONFIRMED=true` and `--i-understand-the-risks`
passed explicitly. The gate was never unlocked during this project.

---

## Repository Structure

```
config/
  settings.py   — pydantic settings; MODE constraint + live-gate validation
core/
  engine.py     — the single order gateway; every order passes risk_check()
  risk.py       — RiskEngine: drawdown/failure/heartbeat HALTs, never self-clears
  exchange.py   — ccxt connectivity and the per-mode order executors
  position.py   — position tracking, realized/unrealized PnL
  database.py   — SQLite (WAL) persistence: candles, features, state
ml/
  features.py   — causal OHLCV features (+ on-chain/premium columns appended)
  labels.py     — Flat/Long labels, 24bps round-trip hurdle
  train.py      — walk-forward XGBoost, strict OOS evaluation
  models/       — trained model artifacts (gitignored)
backtest/
  engine.py     — event-driven spot backtester, fills at t+1 open
  costs.py      — CostModel: 10bps taker + 2bps slippage per side
  metrics.py    — Sharpe, drawdown, turnover, benchmark comparison
src/
  onchain/      — CoinMetrics AdrActCnt fetcher, Coinbase Premium fetcher
                  (BTC 1h / LINK 1d), PIT feature modules
web/            — read-only FastAPI dashboard (app, queries, templates)
scripts/
  fetch_data.py — OHLCV ingest (drops the forming candle)
  run_backtest.py
  run_onchain_backtest.py        — Experiments 6–7
  run_premium_backtest.py        — Experiment 8
  run_link_premium_backtest.py   — Experiment 9
  run_paper.py / run_live.py     — paper loop / hard-gated live entry
  reset_halt.py — manual HALT reset, requires --confirm
tests/          — 120 tests, all passing
results/
  backtest_verdict.md  — full experimental record
data/           — runtime artifacts: SQLite DB, parquet caches (gitignored)
```
