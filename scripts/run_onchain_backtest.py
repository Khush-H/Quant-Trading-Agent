"""Experiment 6 runner: BTC/USDT 1d walk-forward with on-chain AdrActCnt features.

Pre-registered windows (hard-coded; not flags, so they cannot drift):

* TUNING : 2018-01-01 <= candle open ts < 2023-01-01   (the default)
* HOLDOUT: 2023-01-01 <= candle open ts < 2025-06-01   (LOCKED)

The holdout end is a HARD CAP at 2025-06-01 regardless of how far the stored
candles or the on-chain parquet extend. The holdout window refuses to run
without ``--confirm-holdout``, which must only ever be passed after the
explicit "RUN HOLDOUT" instruction. Nothing in the tuning path reads a single
bar or on-chain value dated at/after 2023-01-01.

Everything else is the unchanged pipeline: causal features (+3 on-chain
columns under the proven D-1 lag), Flat/Long labels (N=1, 24bps hurdle),
walk-forward XGBoost (5 splits, embargo = label horizon), event-driven spot
backtest (10bps taker + 2bps slippage per side, next-bar-open execution, 20%
fixed-fractional, long-only). Threshold fixed at the 0.50 default — nothing is
tuned on results.

Usage:
    python -m scripts.run_onchain_backtest                  # tuning window
    python -m scripts.run_onchain_backtest --window holdout --confirm-holdout
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

SYMBOL = "BTC/USDT"
TIMEFRAME = "1d"
THRESHOLD = 0.50  # pipeline default; pre-registered, not tuned
N_SPLITS = 5      # pipeline default

# Pre-registered boundaries, epoch ms UTC. End bounds are EXCLUSIVE on the
# candle OPEN ts.
TUNING_START_MS = 1514764800000   # 2018-01-01
TUNING_END_MS = 1672531200000     # 2023-01-01
HOLDOUT_END_HARDCAP_MS = 1748736000000  # 2025-06-01 — hard cap, never exceeded

_WINDOWS = {
    "tuning": (TUNING_START_MS, TUNING_END_MS),
    "holdout": (TUNING_END_MS, HOLDOUT_END_HARDCAP_MS),
}


def _load_capped_ohlcv(db, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = db.load_candles(SYMBOL, TIMEFRAME)
    if not rows:
        raise SystemExit(
            f"No candles for {SYMBOL} {TIMEFRAME}. Run scripts.fetch_data first."
        )
    ohlcv = pd.DataFrame([dict(r) for r in rows]).set_index("ts").sort_index()
    ohlcv = ohlcv.loc[(ohlcv.index >= start_ms) & (ohlcv.index < end_ms)]
    # Belt-and-braces: prove the cap held before anything downstream runs.
    assert int(ohlcv.index.max()) < end_ms, "window cap violated"
    assert int(ohlcv.index.max()) < HOLDOUT_END_HARDCAP_MS, "2025-06-01 hard cap violated"
    return ohlcv


def _load_capped_onchain(end_ms: int) -> pd.DataFrame:
    from src.onchain.coinmetrics_fetcher import load_adr_act_cnt

    adr = load_adr_act_cnt()  # parquet cache; no network on reruns
    # The D-1 lag inside the feature builder is the real guarantee (proven by
    # tests/test_onchain_pit_leakage.py); capping the frame at the window end
    # is an extra fence so out-of-window data is not even present in memory.
    cutoff = pd.Timestamp(end_ms, unit="ms")
    return adr.loc[adr["date"] < cutoff].reset_index(drop=True)


def _gross_curve(equity: pd.Series, trades: pd.DataFrame) -> pd.Series:
    """Net equity with every fill's cost added back along the executed path.

    Exact decomposition (same fills, sizing, timing) — not a re-simulation:
    gross_equity[t] = net_equity[t] + cumulative costs paid through t.
    """
    if trades.empty:
        return equity.copy()
    costs_by_ts = trades.groupby("ts")["cost"].sum()
    cum_costs = costs_by_ts.reindex(equity.index).fillna(0.0).cumsum()
    return equity + cum_costs


def run(window: str) -> dict:
    from backtest.engine import run_backtest
    from backtest.metrics import sharpe, total_return
    from config import get_settings
    from core.database import FEATURE_COLUMNS, Database
    from ml.features import build_features
    from ml.labels import build_labels
    from ml.train import walk_forward
    from src.onchain.features import ONCHAIN_FEATURE_COLUMNS

    settings = get_settings()
    assert abs(settings.label_hurdle - 0.0024) < 1e-12, (
        f"expected the pre-registered 24bps hurdle, got {settings.label_hurdle}"
    )
    db = Database(settings)
    start_ms, end_ms = _WINDOWS[window]

    ohlcv = _load_capped_ohlcv(db, start_ms, end_ms)
    adr = _load_capped_onchain(end_ms)

    features = build_features(ohlcv, onchain=adr)
    expected_cols = list(FEATURE_COLUMNS) + list(ONCHAIN_FEATURE_COLUMNS)
    assert list(features.columns) == expected_cols, "feature set drifted"
    labels = build_labels(ohlcv, settings=settings)

    wf = walk_forward(features, labels, settings=settings, n_splits=N_SPLITS)
    signal = wf.oos_signal(threshold=THRESHOLD)

    oos_index = ohlcv.index.intersection(signal.index)
    oos_ohlcv = ohlcv.loc[oos_index].sort_index()
    result = run_backtest(
        oos_ohlcv, signal.reindex(oos_ohlcv.index),
        settings=settings, timeframe=TIMEFRAME,
    )

    m, b = result.metrics, result.benchmark
    gross_eq = _gross_curve(result.equity_curve, result.trades)
    gross_sharpe = sharpe(gross_eq.pct_change().dropna(), TIMEFRAME)
    gross_return = total_return(gross_eq)

    # B&H over the FULL window (entry at window start), for reference next to
    # the engine's B&H over the OOS evaluation window (the apples-to-apples one).
    full_bh_eq = ohlcv["close"] / ohlcv["close"].iloc[0]
    full_bh_sharpe = sharpe(full_bh_eq.pct_change().dropna(), TIMEFRAME)

    def _d(ts_ms: int) -> str:
        return str(pd.Timestamp(int(ts_ms), unit="ms").date())

    print("\n" + "=" * 68)
    print(f"EXPERIMENT 6 — {SYMBOL} {TIMEFRAME} + on-chain AdrActCnt "
          f"[{window.upper()} WINDOW]")
    print("=" * 68)
    print(f"Window (candle open) : {_d(ohlcv.index.min())} -> {_d(ohlcv.index.max())}"
          f"  ({len(ohlcv)} bars, cap < {_d(end_ms)})")
    print(f"Feature rows         : {len(features)}  ({len(expected_cols)} features)")
    print(f"OOS bars backtested  : {len(result.equity_curve)}"
          f"  ({_d(oos_index.min())} -> {_d(oos_index.max())})")
    print(f"Walk-forward folds   : {len(wf.folds)} "
          f"(embargo {settings.label_horizon} bar, threshold {THRESHOLD:.2f})")
    print("-" * 68)
    print(f"Net Sharpe (ann.)    : {m['sharpe']:>10.2f}")
    print(f"Gross Sharpe (ann.)  : {gross_sharpe:>10.2f}   (costs added back, same fills)")
    print(f"Net return           : {m['total_return']:>10.2%}   (gross: {gross_return:+.2%})")
    print(f"Trade count          : {m['num_trades']:>10d}   (round trips; win rate {m['win_rate']:.1%})")
    print(f"Avg PnL per trade    : {m['avg_trade_pnl']:>+10.2f}")
    print(f"Turnover             : {m['turnover']:>9.2f}x")
    print(f"Max drawdown         : {m['max_drawdown']:>10.2%}")
    print("-" * 68)
    print(f"Buy & hold Sharpe    : {b['sharpe']:>10.2f}   (same OOS evaluation window)")
    print(f"  .. B&H return      : {b['total_return']:>10.2%}   maxDD {b['max_drawdown']:.2%}")
    print(f"Buy & hold Sharpe    : {full_bh_sharpe:>10.2f}   (full {window} window, reference)")
    print("=" * 68 + "\n")

    return {
        "window": window,
        "net_sharpe": m["sharpe"],
        "gross_sharpe": gross_sharpe,
        "net_return": m["total_return"],
        "gross_return": gross_return,
        "num_trades": m["num_trades"],
        "avg_trade_pnl": m["avg_trade_pnl"],
        "turnover": m["turnover"],
        "max_drawdown": m["max_drawdown"],
        "bh_sharpe_oos": b["sharpe"],
        "bh_sharpe_full_window": full_bh_sharpe,
        "folds": wf.folds,
    }


def main() -> None:
    # Pin mode and the pre-registered cost model BEFORE settings construct:
    # 10bps taker (CostModel default) + 2bps slippage per side -> 24bps hurdle.
    os.environ["MODE"] = "backtest"
    os.environ["SLIPPAGE_BPS"] = "2"

    parser = argparse.ArgumentParser(description="Experiment 6 walk-forward run.")
    parser.add_argument("--window", choices=list(_WINDOWS), default="tuning")
    parser.add_argument(
        "--confirm-holdout", action="store_true",
        help="Required for --window holdout. Only after the explicit "
             "'RUN HOLDOUT' instruction.",
    )
    args = parser.parse_args()

    if args.window == "holdout" and not args.confirm_holdout:
        raise SystemExit(
            "REFUSED: the holdout is locked. Pass --confirm-holdout only "
            "after the explicit 'RUN HOLDOUT' instruction."
        )

    run(args.window)


if __name__ == "__main__":
    main()
