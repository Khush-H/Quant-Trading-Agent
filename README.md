# Quant Trading System

A local quantitative trading system for crypto, built on a strict
**backtest → paper → live** promotion path. The defining design choice is that
**every order — simulated, paper, or live — flows through a single gateway**, so
the risk layer can never be bypassed.

> ⚠️ This repo is currently a **scaffold**. Config and guardrails are real and
> tested; trading logic is intentionally not implemented yet (most functions
> raise `NotImplementedError`). Build it out in the order below.

---

## Safety model (read this first)

Three guardrails are enforced in code, not by convention:

1. **One mode, paper by default.** `MODE` is `backtest | paper | live` and
   defaults to `paper` (`config/settings.py`). There is no code path that
   defaults into live.
2. **Live is gated.** `MODE=live` refuses to even construct settings unless
   `LIVE_TRADING_CONFIRMED=true` *and* exchange credentials are present.
   `scripts/run_live.py` re-checks both before doing anything.
3. **One order gateway.** All orders go through `core.engine.submit_order`,
   which runs the risk layer *before* dispatching to the mode-appropriate
   executor (`SimulatedExecutor` / `PaperExecutor` / `LiveExecutor`). Strategy
   code must never call an executor directly. The backtest uses the *same*
   gateway, so a strategy that passes risk in backtest behaves identically in
   paper and live.

Secrets (API keys, DB URL) come **only** from environment variables. `.env` is
git-ignored; copy `.env.example` to `.env` for local development.

---

## Project layout

```
config/      settings.py — pydantic-settings; MODE + live gate + all secrets
core/        database.py  — persistence (OHLCV, orders, fills, positions)
             engine.py    — Order types + submit_order (THE order gateway)
             risk.py      — RiskEngine.check(); the only approver/resizer
             position.py  — position & PnL tracking
             exchange.py  — ccxt client + the 3 per-mode executors
ml/          features.py  — causal feature engineering
             labels.py    — supervised targets (no look-ahead)
             train.py     — XGBoost training (walk-forward)
             models/      — trained artifacts (git-ignored)
backtest/    engine.py    — replay loop (routes orders via submit_order)
             costs.py     — fees + slippage model
             metrics.py   — Sharpe, drawdown, hit rate, ...
web/         app.py       — FastAPI read-only dashboard (mode banner)
             templates/   — Jinja2 templates
scripts/     fetch_data.py, run_backtest.py, run_paper.py, run_live.py
tests/       guardrail + gateway tests
```

---

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env      # PowerShell: Copy-Item .env.example .env
pytest                      # guardrail tests should pass on the scaffold
```

Pinned versions (do not loosen casually): `ccxt==4.2.14`, `xgboost==2.0.3`,
`pandas==2.2.0`, `numpy==1.26.4`, `scikit-learn==1.4.0`, `fastapi==0.109.2`,
`uvicorn==0.27.1`, `pydantic==2.6.1`.

---

## Build order

Build strictly in this sequence. Each step depends on the one before it, and
the ordering is deliberate — costs and risk are introduced *before* anything
touches real money.

1. **Data** — `scripts/fetch_data.py`, `core/database.py`.
   Fetch and persist clean OHLCV. Everything downstream needs trustworthy data
   first. Read-only against the exchange.

2. **Features / Labels** — `ml/features.py`, `ml/labels.py`.
   Build a strictly *causal* feature matrix and aligned targets. Get this wrong
   (look-ahead) and every later result is fiction.

3. **Backtest** — `backtest/engine.py`, `backtest/costs.py`,
   `backtest/metrics.py`.
   Replay history through `submit_order`, *with realistic costs*. Establish the
   metrics you trust before you trust any model. Build the backtest **before**
   training so the model is judged against costs, not on paper accuracy.

4. **Train** — `ml/train.py`.
   Train the XGBoost model with walk-forward / purged validation. Evaluate it
   through the step-3 backtest, not in isolation.

5. **Paper** — `scripts/run_paper.py`, `core/exchange.py` (`PaperExecutor`).
   Run on **live prices with simulated fills**. Same gateway, same risk layer —
   this is the dress rehearsal for live.

6. **Risk** — `core/risk.py`, `core/position.py`.
   Harden the risk layer: notional/leverage caps, open-position limits, daily
   loss limits, drawdown stops. (The interface exists from day one; this is
   where the real rules and stateful checks land, validated against paper.)

7. **Dashboard** — `web/app.py`.
   Wire the read-only monitoring surface to live state: positions, orders,
   equity curve. Display the active mode prominently. The dashboard never
   places orders.

8. **Live** — `scripts/run_live.py`, `core/exchange.py` (`LiveExecutor`).
   Final step, only after paper + hardened risk are proven. Requires
   `MODE=live` **and** `LIVE_TRADING_CONFIRMED=true` **and** credentials. Start
   with the smallest possible size.

---

## Running

```bash
# Fetch data (mode-agnostic, read-only)
python -m scripts.fetch_data --symbol BTC/USDT --timeframe 1h

# Backtest (forces MODE=backtest)
python -m scripts.run_backtest --symbol BTC/USDT --timeframe 1h

# Paper trading (forces MODE=paper) — the default, safe loop
python -m scripts.run_paper

# Dashboard
uvicorn web.app:app --host 127.0.0.1 --port 8000

# LIVE — real money. You must opt in explicitly; nothing defaults here:
#   MODE=live LIVE_TRADING_CONFIRMED=true python -m scripts.run_live
```

---

## Conventions

- Never read secrets from `os.environ` directly — import `get_settings()` from
  `config`.
- Never call an executor's `execute()` directly — always go through
  `core.engine.submit_order`.
- Keep features causal and backtests costed; treat both as correctness issues,
  not nice-to-haves.
