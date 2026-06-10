"""Coinbase-premium feature computation (Experiment 8) — T-inclusive PIT.

The pre-registered point-in-time rule this module enforces:

    A feature row for the 1h bar at T uses ONLY coinbase_premium values dated
    T or earlier. The premium at T is computed from the CLOSED bars of both
    exchanges at hour T, so it is fully known at T's close — no lag is needed
    (unlike the on-chain D-1 rule); the thing to verify is that the rolling
    windows never reach past T. Proven by perturbation in
    ``tests/test_coinbase_premium_pit_leakage.py``.

Exactly three features (the experiment's full scope):

* ``coinbase_premium``     — raw premium at the closed bar T,
                             ``(cb_close - bn_close) / bn_close * 100``.
                             Missing hours stay NaN (dropped by the shared
                             dropna in ``build_features`` — never invented).
* ``premium_zscore_168h``  — 168-bar rolling z-score of the premium, window
                             ending at T inclusive, ``min_periods=168``.
                             std==0 or NaN -> 0.
* ``premium_mom_24h``      — ``premium[T] / premium[T-24] - 1``.
                             Denominator 0 or NaN -> 0.

Lookups are by CALENDAR HOUR on a complete hourly grid (gap hours from the
alignment drop stay NaN; ``min_periods`` turns any gap-crossing window into
NaN, mapped to 0 by the fill rule) — windows are calendar-true and stay
correct even if the OHLCV index has gaps.

Deliberately free of imports from ``ml`` so ``ml.features`` can import it
without a cycle (same layout as the AdrActCnt module).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PREMIUM_COLUMN = "coinbase_premium"
PREMIUM_Z_WINDOW = 168
PREMIUM_MOM_LAG = 24

# Order matters: appended after FEATURE_COLUMNS in the recipe hash.
PREMIUM_FEATURE_COLUMNS = (
    "coinbase_premium",
    "premium_zscore_168h",
    "premium_mom_24h",
)

_HOUR = pd.Timedelta(hours=1)


def _candle_times(index: pd.Index) -> pd.DatetimeIndex:
    """UTC datetime of each 1h candle's OPEN ts (epoch ms)."""
    return pd.DatetimeIndex(
        pd.to_datetime(index.to_numpy(dtype="int64"), unit="ms")
    )


def _hourly_series(premium: pd.DataFrame) -> pd.Series:
    """coinbase_premium indexed by a COMPLETE hourly calendar (gaps -> NaN)."""
    s = (
        premium.sort_values("timestamp_utc")
        .drop_duplicates(subset="timestamp_utc", keep="last")
        .set_index("timestamp_utc")[PREMIUM_COLUMN]
        .astype(float)
    )
    full = pd.date_range(s.index[0], s.index[-1], freq="h")
    return s.reindex(full)


def add_premium_features(
    ohlcv: pd.DataFrame, premium: pd.DataFrame
) -> pd.DataFrame:
    """Return the three premium feature columns, indexed like ``ohlcv``.

    ``ohlcv``: 1h candles indexed by ``ts`` (bar OPEN time, epoch ms, UTC,
    ascending) — the frame ``ml.features.build_features`` consumes.
    ``premium``: the aligned frame from
    ``src.onchain.coinbase_premium_fetcher`` with at least
    ``[timestamp_utc, coinbase_premium]``.

    Causality: every lookup for the row at T reads the premium dated T or
    earlier; no rolling window extends past T.
    """
    if not ohlcv.index.is_monotonic_increasing:
        ohlcv = ohlcv.sort_index()

    s = _hourly_series(premium)
    t = _candle_times(ohlcv.index)

    # Feature 1: raw premium at the closed bar T. NaN where the hour was
    # dropped at alignment — propagates to build_features' dropna.
    raw = pd.Series(s.reindex(t).to_numpy(), index=ohlcv.index)

    # Feature 2: 168-bar rolling z-score, window ending at T inclusive.
    roll = s.rolling(window=PREMIUM_Z_WINDOW, min_periods=PREMIUM_Z_WINDOW)
    std = roll.std(ddof=0)
    z_hourly = (s - roll.mean()) / std.replace(0.0, np.nan)
    z = pd.Series(z_hourly.reindex(t).to_numpy(), index=ohlcv.index).fillna(0.0)

    # Feature 3: 24h momentum, premium[T] / premium[T-24] - 1.
    v0 = s.reindex(t).to_numpy()
    v24 = s.reindex(t - PREMIUM_MOM_LAG * _HOUR).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        mom = v0 / v24 - 1.0
    mom = np.where(np.isfinite(mom), mom, np.nan)  # denom 0/NaN -> NaN -> 0
    mom = pd.Series(mom, index=ohlcv.index).fillna(0.0)

    out = pd.DataFrame(index=ohlcv.index)
    out["coinbase_premium"] = raw
    out["premium_zscore_168h"] = z
    out["premium_mom_24h"] = mom
    return out[list(PREMIUM_FEATURE_COLUMNS)]
