"""CoinMetrics Community API fetcher for on-chain metrics (Experiments 6/7).

Pulls the daily active-address count (``AdrActCnt``) for a given asset
(``btc`` by default; ``eth`` for Experiment 7) from the free CoinMetrics
Community v4 endpoint and caches it locally as parquet — one file per asset,
``data/onchain/{asset}_adr_act_cnt.parquet`` — so repeated pipeline runs never
re-hit the network and the assets' caches can never clobber each other.

Publication timing (the point-in-time rule downstream code must respect)
-------------------------------------------------------------------------
CoinMetrics computes ``AdrActCnt`` for calendar day D (UTC) over the FULL day,
so the value is only published after day D ends (end of day D UTC). At trading
open on day D the most recent on-chain value available is day D-1's. This
module only FETCHES and CACHES the raw series; enforcing the D-1 lag is the
feature layer's job (see ``ml.features``) and is proven by
``tests/test_onchain_pit_leakage.py``.

Read-only against a public API; no credentials, no orders, mode-agnostic.

Usage:
    python -m src.onchain.coinmetrics_fetcher                # BTC (default)
    python -m src.onchain.coinmetrics_fetcher --asset eth
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

API_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
DEFAULT_ASSET = "btc"
METRIC = "AdrActCnt"
START_TIME = "2018-01-01"

# Repo-root-relative cache location (repo root = two levels above this file).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO_ROOT / "data" / "onchain"
# Kept for backward compatibility with Experiment-6 call sites: the BTC path.
CACHE_PATH = _CACHE_DIR / "btc_adr_act_cnt.parquet"


def cache_path_for(asset: str) -> Path:
    """Per-asset cache file, e.g. ``data/onchain/eth_adr_act_cnt.parquet``."""
    return _CACHE_DIR / f"{asset.lower()}_adr_act_cnt.parquet"

_REQUEST_TIMEOUT_S = 30
_MAX_RETRIES = 5
_RETRY_BACKOFF_S = 2.0  # doubled on each retry (handles 429 rate limits)


def _get_with_retries(url: str, params: Optional[dict] = None) -> dict:
    """GET ``url`` returning parsed JSON, retrying on 429/5xx with backoff."""
    backoff = _RETRY_BACKOFF_S
    last_err: Optional[Exception] = None
    for _ in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_S)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(
                    f"HTTP {resp.status_code} from CoinMetrics: {resp.text[:200]}"
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:  # network hiccup: retry
            last_err = exc
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"CoinMetrics request failed after {_MAX_RETRIES} tries") \
        from last_err


def fetch_adr_act_cnt(
    start_time: str = START_TIME, asset: str = DEFAULT_ASSET
) -> pd.DataFrame:
    """Fetch the full daily ``AdrActCnt`` series for ``asset`` from CoinMetrics.

    Follows ``next_page_url`` pagination until exhausted. Returns a DataFrame
    with columns ``[date, AdrActCnt]`` sorted ascending and deduped on date:
    ``date`` is the UTC calendar day (datetime64, midnight, no tz) the metric
    was computed OVER, and ``AdrActCnt`` is float.
    """
    params: Optional[dict] = {
        "assets": asset.lower(),
        "metrics": METRIC,
        "frequency": "1d",
        "start_time": start_time,
        "page_size": 10_000,
    }
    url = API_URL
    records: list[dict] = []
    while True:
        payload = _get_with_retries(url, params=params)
        records.extend(payload.get("data", []))
        next_url = payload.get("next_page_url")
        if not next_url:
            break
        # next_page_url already encodes all query params, including the cursor.
        url, params = next_url, None

    if not records:
        raise RuntimeError(
            f"CoinMetrics returned no AdrActCnt data for asset {asset!r}."
        )

    df = pd.DataFrame.from_records(records)
    # ``time`` is the UTC day start, e.g. "2018-01-01T00:00:00.000000000Z".
    # Normalize to a tz-naive UTC date (midnight, no time component).
    df["date"] = (
        pd.to_datetime(df["time"], utc=True, format="ISO8601")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    df[METRIC] = df[METRIC].astype(float)
    df = (
        df[["date", METRIC]]
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


def load_adr_act_cnt(
    asset: str = DEFAULT_ASSET,
    cache_path: Optional[Path] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the cached AdrActCnt series for ``asset``, fetching on first use.

    ``cache_path`` defaults to the per-asset file from :func:`cache_path_for`.
    ``refresh=True`` forces a re-fetch (the cache is overwritten atomically via
    a temp file so a failed fetch never corrupts an existing cache).
    """
    if cache_path is None:
        cache_path = cache_path_for(asset)
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    df = fetch_adr_act_cnt(asset=asset)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(cache_path)
    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch/cache AdrActCnt.")
    parser.add_argument("--asset", default=DEFAULT_ASSET)
    args = parser.parse_args()

    df = load_adr_act_cnt(asset=args.asset)
    print(f"[coinmetrics_fetcher] cached at: {cache_path_for(args.asset)}")
    print(f"[coinmetrics_fetcher] asset: {args.asset.lower()}, rows: {len(df)}, "
          f"earliest: {df['date'].iloc[0].date()}, "
          f"latest: {df['date'].iloc[-1].date()}")
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))
    print("\nLast 5 rows:")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
