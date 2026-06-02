"""Run the walk-forward OOS strategy through the backtester and print a verdict.

Forces MODE=backtest for the process, then end-to-end:

  1. loads stored candles for a symbol/timeframe,
  2. builds the causal feature matrix and the Flat/Long labels,
  3. produces strictly out-of-sample predictions via walk-forward (embargoed),
  4. backtests the OOS signal candle-by-candle with realistic costs, and
  5. prints a clear verdict comparing the NET-OF-COST strategy return AND Sharpe
     against simply buying and holding BTC over the same window.

Accuracy is reported too, but flagged explicitly as SECONDARY and MISLEADING on
imbalanced data: a model that always predicts the majority class (Flat) can post
high accuracy while making zero money. The backtest verdict, not accuracy, is
what decides whether the strategy has an edge.

Usage:
    python -m scripts.run_backtest --symbol BTC/USDT --timeframe 1h \
        --threshold 0.5 --splits 5
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd


def _load_ohlcv(db, symbol: str, timeframe: str) -> pd.DataFrame:
    rows = db.load_candles(symbol, timeframe)
    if not rows:
        raise SystemExit(
            f"No candles for {symbol} {timeframe}. Run scripts.fetch_data first."
        )
    return pd.DataFrame([dict(r) for r in rows]).set_index("ts").sort_index()


def _accuracy(signal: pd.Series, labels: pd.Series) -> float:
    common = signal.index.intersection(labels.index)
    if len(common) == 0:
        return 0.0
    return float((signal.loc[common] == labels.loc[common]).mean())


def run(
    symbol: str,
    timeframe: str,
    threshold: float = 0.5,
    n_splits: int = 5,
    settings=None,
    db=None,
) -> dict:
    # Imported here so the MODE pin in main() lands before settings construct.
    from backtest.engine import run_backtest
    from config import get_settings
    from core.database import Database
    from ml.features import build_features
    from ml.labels import build_labels
    from ml.train import walk_forward

    settings = settings or get_settings()
    db = db or Database(settings)

    ohlcv = _load_ohlcv(db, symbol, timeframe)
    features = build_features(ohlcv)
    labels = build_labels(ohlcv, settings=settings)

    wf = walk_forward(features, labels, settings=settings, n_splits=n_splits)
    signal = wf.oos_signal(threshold=threshold)

    # Backtest over exactly the OOS window the predictions cover.
    oos_index = ohlcv.index.intersection(signal.index)
    oos_ohlcv = ohlcv.loc[oos_index].sort_index()
    result = run_backtest(
        oos_ohlcv, signal.reindex(oos_ohlcv.index),
        settings=settings, timeframe=timeframe,
    )

    accuracy = _accuracy(signal, wf.oos_labels)
    _print_verdict(symbol, timeframe, threshold, result, accuracy, wf)
    return {
        "metrics": result.metrics,
        "benchmark": result.benchmark,
        "accuracy": accuracy,
    }


def _print_verdict(symbol, timeframe, threshold, result, accuracy, wf) -> None:
    m = result.metrics
    b = result.benchmark
    beat_return = m["total_return"] > b["total_return"]
    beat_sharpe = m["sharpe"] > b["sharpe"]

    print("\n" + "=" * 64)
    print(f"BACKTEST VERDICT — {symbol} {timeframe}  (OOS, net of costs)")
    print("=" * 64)
    print(f"Out-of-sample bars : {len(result.equity_curve)}")
    print(f"Signal threshold   : P(Long) >= {threshold:.2f}")
    print(f"Trades             : {m['num_trades']}  "
          f"(win rate {m['win_rate']:.1%}, avg PnL/trade {m['avg_trade_pnl']:+.2f})")
    print(f"Turnover           : {m['turnover']:.2f}x")
    print("-" * 64)
    print(f"{'':22}{'STRATEGY':>14}{'BUY & HOLD':>16}")
    print(f"{'Total return':22}{m['total_return']:>13.2%}{b['total_return']:>16.2%}")
    print(f"{'Sharpe (annualized)':22}{m['sharpe']:>14.2f}{b['sharpe']:>16.2f}")
    print(f"{'Sortino':22}{m['sortino']:>14.2f}{b['sortino']:>16.2f}")
    print(f"{'Max drawdown':22}{m['max_drawdown']:>13.2%}{b['max_drawdown']:>16.2%}")
    print("-" * 64)
    verdict = (
        "EDGE: strategy beats buy-and-hold on BOTH net return and Sharpe."
        if (beat_return and beat_sharpe) else
        "NO CLEAR EDGE: strategy does not beat buy-and-hold on both "
        "net return and Sharpe."
    )
    print(verdict)
    print("-" * 64)
    print(f"[secondary] OOS accuracy: {accuracy:.1%}  "
          "<-- MISLEADING on imbalanced labels; a majority-class predictor can "
          "score high while making no money. Judge by the net-of-cost verdict "
          "above, not this.")
    if wf.calibration is not None and len(wf.calibration):
        print("-" * 64)
        print("Calibration (predicted P(Long) vs observed Long frequency):")
        print(wf.calibration.to_string(index=False,
                                       float_format=lambda x: f"{x:.3f}"))
    print("=" * 64 + "\n")


def main() -> None:
    # Pin the mode for this process BEFORE settings are first constructed.
    os.environ["MODE"] = "backtest"

    from config import Mode, get_settings

    parser = argparse.ArgumentParser(
        description="Walk-forward OOS backtest + verdict vs buy-and-hold."
    )
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="P(Long) cutoff for going long.")
    parser.add_argument("--splits", type=int, default=5,
                        help="Number of walk-forward test blocks.")
    args = parser.parse_args()

    settings = get_settings()
    if settings.mode is not Mode.BACKTEST:  # defensive
        raise SystemExit(f"Expected backtest mode, got {settings.mode.value!r}.")

    run(args.symbol, args.timeframe, threshold=args.threshold,
        n_splits=args.splits, settings=settings)


if __name__ == "__main__":
    main()
