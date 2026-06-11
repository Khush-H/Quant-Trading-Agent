"""Dual-exchange Coinbase-premium fetcher (Experiments 8 & 9).

Fetches the Coinbase-side and Binance-side OHLCV closes for one asset over
the experiment range, aligns the two series on bar-open UTC timestamps, and
caches the aligned premium series as parquet. Parameterised by
``(asset, timeframe)``:

* ``("btc", "1h")``  — Experiment 8: BTC/USD 1h via ccxt Coinbase Advanced
  Trade vs BTC/USDT 1h via ccxt Binance, 2019-01-01 -> 2023-01-01.
* ``("link", "1d")`` — Experiment 9: LINK-USD 1d via direct GET requests to
  ``api.exchange.coinbase.com/products/LINK-USD/candles`` (granularity 86400,
  max 300 candles per request, paginated forward on ``start``) vs LINK/USDT
  1d via ccxt Binance, 2020-10-01 -> 2025-06-01.

Alignment rule (pre-registered): a bar enters the cache ONLY if BOTH
exchanges have a closed bar for it. Bars where either side is missing are
DROPPED — never forward-filled, never interpolated — and the drop counts are
recorded in a sidecar meta JSON next to the cache. If bars present on exactly
one exchange exceed 1% of the expected bar grid, the fetch raises instead of
caching (the experiment's misalignment stop-rule).

``coinbase_premium = (coinbase_close - binance_close) / binance_close * 100``
— both closes from the SAME closed bar, so the premium at T is fully known at
T's close by construction. The rolling-window features built on top of it are
PIT-proven separately by ``tests/test_coinbase_premium_pit_leakage.py`` (1h)
and ``tests/test_link_premium_pit_leakage.py`` (1d).

Each spec caches to its own parquet; fetching one asset can never touch
another asset's cache. Read-only public market data; no credentials, no
orders, mode-agnostic.

Usage:
    python -m src.onchain.coinbase_premium_fetcher                       # btc 1h
    python -m src.onchain.coinbase_premium_fetcher --asset link --timeframe 1d
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS

# --- Experiment 8 (btc, 1h) constants — unchanged from the original module.
START_TIME = "2019-01-01"   # inclusive, UTC
END_TIME = "2023-01-01"     # exclusive, UTC
COINBASE_SYMBOL = "BTC/USD"
BINANCE_SYMBOL = "BTC/USDT"
COINBASE_PAGE_LIMIT = 300   # Coinbase hard cap per request (both APIs)
BINANCE_PAGE_LIMIT = 1000

# Stop-rule: bars present on exactly one exchange, as a share of the
# expected bar grid, must not exceed this.
MAX_MISALIGNED_FRACTION = 0.01

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data" / "onchain"
CACHE_PATH = _DATA_DIR / "btc_coinbase_premium.parquet"
META_PATH = CACHE_PATH.with_suffix(".meta.json")

COINBASE_EXCHANGE_API = "https://api.exchange.coinbase.com"

_MAX_RETRIES = 5
_RETRY_BACKOFF_S = 2.0


@dataclass(frozen=True)
class PremiumSpec:
    """One pre-registered (asset, timeframe) premium configuration."""

    asset: str
    timeframe: str            # ccxt timeframe string ("1h" / "1d")
    bar_ms: int
    start: str                # inclusive, UTC
    end: str                  # exclusive, UTC
    coinbase_source: str      # "ccxt" | "exchange_rest"
    coinbase_symbol: str      # ccxt unified symbol, or REST product id
    binance_symbol: str
    cache_name: str

    @property
    def cache_path(self) -> Path:
        return _DATA_DIR / self.cache_name

    @property
    def meta_path(self) -> Path:
        return self.cache_path.with_suffix(".meta.json")


_SPECS: dict[tuple[str, str], PremiumSpec] = {
    ("btc", "1h"): PremiumSpec(
        asset="btc", timeframe="1h", bar_ms=HOUR_MS,
        start=START_TIME, end=END_TIME,
        coinbase_source="ccxt", coinbase_symbol=COINBASE_SYMBOL,
        binance_symbol=BINANCE_SYMBOL,
        cache_name="btc_coinbase_premium.parquet",
    ),
    ("link", "1d"): PremiumSpec(
        asset="link", timeframe="1d", bar_ms=DAY_MS,
        start="2020-10-01", end="2025-06-01",
        coinbase_source="exchange_rest", coinbase_symbol="LINK-USD",
        binance_symbol="LINK/USDT",
        cache_name="link_coinbase_premium.parquet",
    ),
}


def get_spec(asset: str, timeframe: str) -> PremiumSpec:
    key = (asset.lower(), timeframe)
    if key not in _SPECS:
        raise KeyError(
            f"No pre-registered premium spec for {key}; known: {sorted(_SPECS)}"
        )
    return _SPECS[key]


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


def _fetch_with_retries(exchange, symbol: str, timeframe: str, since_ms: int,
                        limit: int):
    """One paginated fetch_ohlcv call with backoff on transient errors."""
    import ccxt

    backoff = _RETRY_BACKOFF_S
    last_err: Optional[Exception] = None
    for _ in range(_MAX_RETRIES):
        try:
            return exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=limit
            )
        except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
            last_err = exc
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(
        f"{type(exchange).__name__} fetch_ohlcv failed after {_MAX_RETRIES} tries"
    ) from last_err


def _fetch_closes_ccxt(kind: str, symbol: str, timeframe: str, bar_ms: int,
                       start_ms: int, end_ms: int, limit: int) -> pd.Series:
    """All closes for ``symbol`` with bar-open ts in [start_ms, end_ms) via ccxt.

    Paginates FORWARD from ``start_ms``: each request resumes at the bar after
    the last one received; an empty page (exchange outage / delisting gap)
    advances the cursor by one full page of bars rather than stalling. Bars
    are deduped on ts (keep last) and clipped to the range.
    """
    exchange = _build_exchange(kind)
    closes: dict[int, float] = {}
    since = start_ms
    while since < end_ms:
        batch = _fetch_with_retries(exchange, symbol, timeframe, since, limit)
        if not batch:
            since += limit * bar_ms  # gap: skip forward, do not stall
            continue
        for row in batch:
            ts = int(row[0])
            if start_ms <= ts < end_ms:
                closes[ts] = float(row[4])
        last_ts = int(batch[-1][0])
        next_since = last_ts + bar_ms
        if next_since <= since:  # defensive: never loop on the same page
            next_since = since + limit * bar_ms
        since = next_since
    return pd.Series(closes, name=f"{kind}_close").sort_index()


def _coinbase_rest_get(product: str, params: dict) -> list:
    """One GET to the Coinbase Exchange candles endpoint, with backoff."""
    import requests  # deferred; ships with ccxt

    url = f"{COINBASE_EXCHANGE_API}/products/{product}/candles"
    backoff = _RETRY_BACKOFF_S
    last_err: Optional[Exception] = None
    for _ in range(_MAX_RETRIES):
        try:
            resp = requests.get(
                url, params=params, timeout=30,
                headers={"User-Agent": "premium-fetcher/1.0"},
            )
            if resp.status_code == 200:
                return resp.json()
            last_err = RuntimeError(
                f"HTTP {resp.status_code} from {url}: {resp.text[:200]}"
            )
        except requests.RequestException as exc:
            last_err = exc
        time.sleep(backoff)
        backoff *= 2
    raise RuntimeError(
        f"Coinbase Exchange GET {url} failed after {_MAX_RETRIES} tries"
    ) from last_err


def _fetch_closes_coinbase_rest(product: str, bar_ms: int, start_ms: int,
                                end_ms: int) -> pd.Series:
    """All closes for ``product`` with bar-open ts in [start_ms, end_ms) via
    the public Coinbase Exchange REST API.

    The endpoint returns at most 300 candles per request, so the range is
    walked in fixed 300-bar windows, moving ``start`` forward each request.
    ``start``/``end`` are both inclusive on the bucket OPEN time, hence the
    window's ``end`` is its last bucket open (``window_end - bar_ms``). The
    cursor advances by a fixed window regardless of payload, so an empty page
    (pre-listing / outage) can never stall the loop. Rows come back as
    ``[time_s, low, high, open, close, volume]``, newest first; closes are
    deduped on ts (keep last) and clipped to the range.
    """
    closes: dict[int, float] = {}
    page_span = COINBASE_PAGE_LIMIT * bar_ms
    cursor = start_ms
    while cursor < end_ms:
        window_end = min(cursor + page_span, end_ms)
        params = {
            "granularity": bar_ms // 1000,
            "start": pd.Timestamp(cursor, unit="ms", tz="UTC").isoformat(),
            "end": pd.Timestamp(window_end - bar_ms, unit="ms",
                                tz="UTC").isoformat(),
        }
        batch = _coinbase_rest_get(product, params)
        for row in batch or []:
            ts = int(row[0]) * 1000
            if start_ms <= ts < end_ms:
                closes[ts] = float(row[4])
        cursor = window_end
    return pd.Series(closes, name="coinbase_close").sort_index()


def fetch_premium(
    start: Optional[str] = None,
    end: Optional[str] = None,
    asset: str = "btc",
    timeframe: str = "1h",
) -> tuple[pd.DataFrame, dict]:
    """Fetch both exchanges, align on bar-open UTC ts, compute the premium.

    Returns ``(df, meta)`` where df has columns [timestamp_utc, coinbase_close,
    binance_close, coinbase_premium] and meta records the alignment audit.
    Raises RuntimeError if misaligned bars exceed the 1% stop-rule.
    """
    spec = get_spec(asset, timeframe)
    start = start or spec.start
    end = end or spec.end
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    if spec.coinbase_source == "exchange_rest":
        cb = _fetch_closes_coinbase_rest(
            spec.coinbase_symbol, spec.bar_ms, start_ms, end_ms
        )
    else:
        cb = _fetch_closes_ccxt(
            "coinbase", spec.coinbase_symbol, spec.timeframe, spec.bar_ms,
            start_ms, end_ms, COINBASE_PAGE_LIMIT,
        )
    bn = _fetch_closes_ccxt(
        "binance", spec.binance_symbol, spec.timeframe, spec.bar_ms,
        start_ms, end_ms, BINANCE_PAGE_LIMIT,
    )

    cb_idx, bn_idx = cb.index, bn.index
    common = cb_idx.intersection(bn_idx)
    cb_only = len(cb_idx.difference(bn_idx))
    bn_only = len(bn_idx.difference(cb_idx))
    misaligned = cb_only + bn_only
    expected_grid = (end_ms - start_ms) // spec.bar_ms
    missing_both = int(expected_grid - len(cb_idx.union(bn_idx)))
    misaligned_frac = misaligned / expected_grid

    meta = {
        "asset": spec.asset, "timeframe": spec.timeframe,
        "start": start, "end": end,
        "expected_grid_bars": int(expected_grid),
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
            f"STOP-RULE: {misaligned} misaligned bars = "
            f"{misaligned_frac:.2%} of the {expected_grid}-bar grid "
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


def load_premium(
    refresh: bool = False, asset: str = "btc", timeframe: str = "1h"
) -> pd.DataFrame:
    """Cached aligned premium series; fetches and caches on first use.

    Atomic write via temp file, same pattern as the AdrActCnt cache: a failed
    fetch can never corrupt an existing cache. Each spec owns its own cache
    file, so loading one asset never touches another's parquet.
    """
    spec = get_spec(asset, timeframe)
    if spec.cache_path.exists() and not refresh:
        return pd.read_parquet(spec.cache_path)

    df, meta = fetch_premium(asset=asset, timeframe=timeframe)
    spec.cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = spec.cache_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(spec.cache_path)
    spec.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Premium fetcher CLI.")
    parser.add_argument("--asset", default="btc")
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()

    spec = get_spec(args.asset, args.timeframe)
    df = load_premium(asset=args.asset, timeframe=args.timeframe)
    meta = (
        json.loads(spec.meta_path.read_text(encoding="utf-8"))
        if spec.meta_path.exists() else {}
    )

    print(f"[coinbase_premium_fetcher] cached at: {spec.cache_path}")
    print(f"rows: {len(df)}  range: {df['timestamp_utc'].iloc[0]} -> "
          f"{df['timestamp_utc'].iloc[-1]} UTC")
    if meta:
        grid = meta.get("expected_grid_bars", meta.get("expected_grid_hours"))
        print(f"expected bar grid    : {grid}")
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
