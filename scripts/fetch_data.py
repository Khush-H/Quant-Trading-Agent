"""Fetch & store historical OHLCV data (data phase).

First step of the build: get clean data into the database so everything
downstream (features, labels, backtest) has something to run on. Read-only
against the exchange; never places orders, so it is mode-agnostic.

The most recent bar returned by the exchange is the *currently forming* one and
is incomplete until its interval closes. We drop it on every fetch so the store
only ever contains finalized candles — otherwise the last row would change
underneath the backtester and features computed on it would be wrong.

Usage:
    python -m scripts.fetch_data --symbol BTC/USDT --timeframe 1h --limit 1000
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from config import Settings, get_settings
from core.database import Database

# Bar open-interval in milliseconds, for the timeframes we support.
_TF_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def _build_readonly_exchange(settings: Settings):
    """Construct a ccxt client for market-data reads only.

    Credentials are optional for public OHLCV endpoints; we still pass any
    configured keys (sourced from env via settings) and honor sandbox mode.
    ccxt is imported lazily so non-fetch entry points don't pay for it.
    """
    import ccxt  # deferred: heavy import, only needed here

    creds = settings.exchange_credentials()
    klass = getattr(ccxt, settings.exchange_id)
    exchange = klass(
        {
            "apiKey": creds["apiKey"],
            "secret": creds["secret"],
            "password": creds["password"],
            "enableRateLimit": True,
        }
    )
    if settings.exchange_sandbox and exchange.has.get("sandbox", False):
        exchange.set_sandbox_mode(True)
    return exchange


def _drop_incomplete_last(
    candles: List[list], timeframe: str, now_ms: int
) -> List[list]:
    """Drop the trailing bar if its interval has not closed yet.

    A bar with open time ``ts`` covers ``[ts, ts + interval)``. It is complete
    only once ``now >= ts + interval``. We always inspect the last row because
    the exchange routinely returns the in-progress bar as the final element.
    """
    if not candles:
        return candles
    interval = _TF_MS.get(timeframe)
    if interval is None:
        # Unknown timeframe: be conservative and drop the last bar outright,
        # since we can't prove it closed.
        return candles[:-1]
    last_ts = int(candles[-1][0])
    if now_ms < last_ts + interval:
        return candles[:-1]
    return candles


def fetch_and_store(
    symbol: str,
    timeframe: str,
    *,
    since: Optional[int] = None,
    limit: int = 1000,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
) -> int:
    """Fetch OHLCV from the exchange and persist finalized bars. Returns count.

    Resumes from the last stored bar when ``since`` is not given, so repeated
    runs extend the history rather than refetching it.
    """
    settings = settings or get_settings()
    db = db or Database(settings)
    db.init_schema()

    if since is None:
        last = db.latest_candle_ts(symbol, timeframe)
        # Refetch the last stored bar too (UPSERT makes it harmless) so we never
        # leave a gap if the previous run stopped on a boundary.
        since = last if last is not None else None

    exchange = _build_readonly_exchange(settings)
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    now_ms = exchange.milliseconds()

    finalized = _drop_incomplete_last(raw, timeframe, now_ms)
    written = db.upsert_candles(symbol, timeframe, finalized)

    if finalized:
        db.set_state(
            f"ingest:{symbol}:{timeframe}:last_ts", str(int(finalized[-1][0]))
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument(
        "--since", default=None,
        help="Epoch ms to start from; defaults to resuming the stored history.",
    )
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    settings = get_settings()
    print(f"[fetch_data] mode={settings.mode.value} "
          f"exchange={settings.exchange_id} sandbox={settings.exchange_sandbox}")
    print(f"[fetch_data] symbol={args.symbol} timeframe={args.timeframe}")

    since = int(args.since) if args.since is not None else None
    written = fetch_and_store(
        args.symbol, args.timeframe, since=since, limit=args.limit, settings=settings
    )
    print(f"[fetch_data] stored {written} finalized candle(s) "
          f"(incomplete trailing bar dropped).")


if __name__ == "__main__":
    main()
