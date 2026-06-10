"""Dual-exchange Coinbase-premium fetcher (Experiment 8).

Fetches BTC/USD 1h OHLCV from Coinbase Advanced Trade and BTC/USDT 1h OHLCV
from Binance over the experiment range, aligns the two series on UTC hour
timestamps, and caches the aligned premium series as parquet.

Alignment rule (pre-registered): an hour enters the cache ONLY if BOTH
exchanges have a closed bar for it. Hours where either side is missing are
DROPPED — never forward-filled, never interpolated — and the drop counts are
recorded in a sidecar meta JSON next to the cache. If hours present on exactly
one exchange exceed 1% of the expected hourly grid, the fetch raises instead
of caching (the experiment's misalignment stop-rule).

``coinbase_premium = (coinbase_close - binance_close) / binance_close * 100``
— both closes from the SAME closed hour bar, so the premium at T is fully
known at T's close by construction. The rolling-window features built on top
of it are PIT-proven separately by ``tests/test_coinbase_premium_pit_leakage.py``.

Read-only public market data; no credentials, no orders, mode-agnostic.

Usage:
    python -m src.onchain.coinbase_premium_fetcher
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

START_TIME = "2019-01-01"   # inclusive, UTC
END_TIME = "2023-01-01"     # exclusive, UTC
HOUR_MS = 60 * 60 * 1000

COINBASE_SYMBOL = "BTC/USD"
BINANCE_SYMBOL = "BTC/USDT"
COINBASE_PAGE_LIMIT = 300   # Coinbase Advanced Trade hard cap per request
BINANCE_PAGE_LIMIT = 1000

# Stop-rule: hours present on exactly one exchange, as a share of the
# expected hourly grid, must not exceed this.
MAX_MISALIGNED_FRACTION = 0.01

_REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = _REPO_ROOT / "data" / "onchain" / "btc_coinbase_premium.parquet"
META_PATH = CACHE_PATH.with_suffix(".meta.json")

_MAX_RETRIES = 5
_RETRY_BACKOFF_S = 2.0


def _build_exchange(kind: str):
    """Public-data ccxt client. ``kind`` is 'coinbase' or 'binance'."""
    import ccxt  # deferred: heavy import

    if kind == "coinbase":
        # Prefer the Advanced Trade client; fall back for older ccxt builds.
        for name in ("coinbaseadvanced", "coinbase"):
            if hasattr(ccxt, name):
                return getattr(ccxt, name)({"enableRateLimit": True})
        raise RuntimeError("No Coinbase exchange class available in ccxt.")
    return ccxt.binance({"enableRateLimit": True})


def _fetch_with_retries(exchange, symbol: str, since_ms: int, limit: int):
    """One paginated fetch_ohlcv call with backoff on transient errors."""
    import ccxt

    backoff = _RETRY_BACKOFF_S
    last_err: Optional[Exception] = None
    for _ in range(_MAX_RETRIES):
        try:
            return exchange.fetch_ohlcv(
                symbol, timeframe="1h", since=since_ms, limit=limit
            )
        except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
            last_err = exc
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(
        f"{type(exchange).__name__} fetch_ohlcv failed after {_MAX_RETRIES} tries"
    ) from last_err


def _fetch_closes(kind: str, symbol: str, start_ms: int, end_ms: int,
                  limit: int) -> pd.Series:
    """All 1h closes for ``symbol`` with open ts in [start_ms, end_ms).

    Paginates FORWARD from ``start_ms``: each request resumes at the bar after
    the last one received; an empty page (exchange outage / delisting gap)
    advances the cursor by one full page of hours rather than stalling. Bars
    are deduped on ts (keep last) and clipped to the range.
    """
    exchange = _build_exchange(kind)
    closes: dict[int, float] = {}
    since = start_ms
    while since < end_ms:
        batch = _fetch_with_retries(exchange, symbol, since, limit)
        if not batch:
            since += limit * HOUR_MS  # gap: skip forward, do not stall
            continue
        for row in batch:
            ts = int(row[0])
            if start_ms <= ts < end_ms:
                closes[ts] = float(row[4])
        last_ts = int(batch[-1][0])
        next_since = last_ts + HOUR_MS
        if next_since <= since:  # defensive: never loop on the same page
            next_since = since + limit * HOUR_MS
        since = next_since
    return pd.Series(closes, name=f"{kind}_close").sort_index()


def fetch_premium(start: str = START_TIME, end: str = END_TIME) -> tuple[pd.DataFrame, dict]:
    """Fetch both exchanges, align on UTC hours, compute the premium.

    Returns ``(df, meta)`` where df has columns [timestamp_utc, coinbase_close,
    binance_close, coinbase_premium] and meta records the alignment audit.
    Raises RuntimeError if misaligned hours exceed the 1% stop-rule.
    """
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    cb = _fetch_closes("coinbase", COINBASE_SYMBOL, start_ms, end_ms,
                       COINBASE_PAGE_LIMIT)
    bn = _fetch_closes("binance", BINANCE_SYMBOL, start_ms, end_ms,
                       BINANCE_PAGE_LIMIT)

    cb_idx, bn_idx = cb.index, bn.index
    common = cb_idx.intersection(bn_idx)
    cb_only = len(cb_idx.difference(bn_idx))
    bn_only = len(bn_idx.difference(cb_idx))
    misaligned = cb_only + bn_only
    expected_grid = (end_ms - start_ms) // HOUR_MS
    missing_both = int(expected_grid - len(cb_idx.union(bn_idx)))
    misaligned_frac = misaligned / expected_grid

    meta = {
        "start": start, "end": end,
        "expected_grid_hours": int(expected_grid),
        "coinbase_bars": int(len(cb_idx)),
        "binance_bars": int(len(bn_idx)),
        "aligned_rows": int(len(common)),
        "dropped_coinbase_only": int(cb_only),
        "dropped_binance_only": int(bn_only),
        "dropped_misaligned_total": int(misaligned),
        "missing_on_both": missing_both,
        "misaligned_fraction": misaligned_frac,
        "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    if misaligned_frac > MAX_MISALIGNED_FRACTION:
        raise RuntimeError(
            f"STOP-RULE: {misaligned} misaligned hours = "
            f"{misaligned_frac:.2%} of the {expected_grid}-hour grid "
            f"(> {MAX_MISALIGNED_FRACTION:.0%}). Not caching. Meta: {meta}"
        )

    cb_al = cb.loc[common].sort_index()
    bn_al = bn.loc[common].sort_index()
    premium = (cb_al - bn_al) / bn_al * 100.0
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(common.sort_values(), unit="ms"),
            "coinbase_close": cb_al.to_numpy(),
            "binance_close": bn_al.to_numpy(),
            "coinbase_premium": premium.to_numpy(),
        }
    ).reset_index(drop=True)
    return df, meta


def load_premium(refresh: bool = False) -> pd.DataFrame:
    """Cached aligned premium series; fetches and caches on first use.

    Atomic write via temp file, same pattern as the AdrActCnt cache: a failed
    fetch can never corrupt an existing cache.
    """
    if CACHE_PATH.exists() and not refresh:
        return pd.read_parquet(CACHE_PATH)

    df, meta = fetch_premium()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(CACHE_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df


def main() -> None:
    df = load_premium()
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}

    print(f"[coinbase_premium_fetcher] cached at: {CACHE_PATH}")
    print(f"rows: {len(df)}  range: {df['timestamp_utc'].iloc[0]} -> "
          f"{df['timestamp_utc'].iloc[-1]} UTC")
    if meta:
        print(f"expected hourly grid : {meta['expected_grid_hours']}")
        print(f"coinbase bars        : {meta['coinbase_bars']}")
        print(f"binance bars         : {meta['binance_bars']}")
        print(f"dropped (cb only)    : {meta['dropped_coinbase_only']}")
        print(f"dropped (bn only)    : {meta['dropped_binance_only']}")
        print(f"dropped misaligned   : {meta['dropped_misaligned_total']} "
              f"({meta['misaligned_fraction']:.3%} of grid; stop-rule 1%)")
        print(f"missing on both      : {meta['missing_on_both']}")
    p = df["coinbase_premium"]
    print(f"premium: mean {p.mean():+.4f}%  std {p.std():.4f}  "
          f"min {p.min():+.4f}%  max {p.max():+.4f}%")
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))
    print("\nLast 5 rows:")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
