"""Run a backtest.

Forces MODE=backtest for the process, then drives the backtest engine (which
routes simulated orders through the shared risk gateway). Implemented in the
backtest phase.

Usage:
    python -m scripts.run_backtest --symbol BTC/USDT --timeframe 1h
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    # Pin the mode for this process BEFORE settings are first constructed.
    os.environ["MODE"] = "backtest"

    from config import Mode, get_settings

    parser = argparse.ArgumentParser(description="Run a backtest.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()

    settings = get_settings()
    if settings.mode is not Mode.BACKTEST:  # defensive
        sys.exit(f"Expected backtest mode, got {settings.mode.value!r}.")

    print(f"[run_backtest] mode={settings.mode.value} symbol={args.symbol}")
    raise NotImplementedError("Implemented in the backtest phase.")


if __name__ == "__main__":
    main()
