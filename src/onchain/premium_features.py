"""Coinbase-premium feature computation (Experiments 8 & 9) — T-inclusive PIT.

The pre-registered point-in-time rule this module enforces:

    A feature row for the bar at T uses ONLY coinbase_premium values dated
    T or earlier. The premium at T is computed from the CLOSED bars of both
    exchanges at T, so it is fully known at T's close — no lag is needed
    (unlike the on-chain D-1 rule); the thing to verify is that the rolling
    windows never reach past T. Proven by perturbation in
    ``tests/test_coinbase_premium_pit_leakage.py`` (1h, Experiment 8) and
    ``tests/test_link_premium_pit_leakage.py`` (1d, Experiment 9).

Exactly three features (the experiments' full scope). Windows are counted in
BARS of the premium series' own resolution: at 1h the z-window spans 168
hours and the momentum lag 24 hours; at 1d they span 168 days and 24 days.
The column names are fixed (part of the pinned feature recipe) and do not
change with resolution:

* ``coinbase_premium``     — raw premium at the closed bar T,
                             ``(cb_close - bn_close) / bn_close * 100``.
                             Missing bars stay NaN (dropped by the shared
                             dropna in ``build_features`` — never invented).
* ``premium_zscore_168h``  — 168-bar rolling z-score of the premium, window
                             ending at T inclusive, ``min_periods=168``.
                             std==0 or NaN -> 0.
* ``premium_mom_24h``      — ``premium[T] / premium[T-24 bars] - 1``.
                             Denominator 0 or NaN -> 0.

The bar interval (1h or 1d) is INFERRED from the premium frame's timestamp
spacing and cross-checked against the OHLCV index spacing — handing an hourly
premium frame to daily candles (or vice versa) raises instead of silently
mixing resolutions.

Lookups are by CALENDAR position on a complete bar grid (gap bars from the
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
_DAY = pd.Timedelta(days=1)

# bar interval -> pandas date_range freq for the complete grid
_GRID_FREQ = {_HOUR: "h", _DAY: "D"}


def _candle_times(index: pd.Index) -> pd.DatetimeIndex:
    """UTC datetime of each candle's OPEN ts (epoch ms)."""
    return pd.DatetimeIndex(
        pd.to_datetime(index.to_numpy(dtype="int64"), unit="ms")
    )


def _infer_bar(times: pd.DatetimeIndex, what: str) -> pd.Timedelta:
    """The series' bar interval (1h or 1d), from median timestamp spacing.

    Median is robust to occasional gaps (alignment-dropped bars). Anything
    other than exactly 1h or 1d is refused — only the two pre-registered
    resolutions exist.
    """
    if len(times) < 2:
        raise ValueError(f"{what}: need >= 2 timestamps to infer the bar interval")
    step = pd.Series(times.sort_values()).diff().median()
    if step not in _GRID_FREQ:
        raise ValueError(
            f"{what}: unsupported bar interval {step!r}; expected 1h or 1d"
        )
    return step


def _grid_series(premium: pd.DataFrame, bar: pd.Timedelta) -> pd.Series:
    """coinbase_premium indexed by a COMPLETE bar calendar (gaps -> NaN)."""
    s = (
        premium.sort_values("timestamp_utc")
        .drop_duplicates(subset="timestamp_utc", keep="last")
        .set_index("timestamp_utc")[PREMIUM_COLUMN]
        .astype(float)
    )
    full = pd.date_range(s.index[0], s.index[-1], freq=_GRID_FREQ[bar])
    return s.reindex(full)


def add_premium_features(
    ohlcv: pd.DataFrame, premium: pd.DataFrame
) -> pd.DataFrame:
    """Return the three premium feature columns, indexed like ``ohlcv``.

    ``ohlcv``: candles indexed by ``ts`` (bar OPEN time, epoch ms, UTC,
    ascending) — the frame ``ml.features.build_features`` consumes.
    ``premium``: the aligned frame from
    ``src.onchain.coinbase_premium_fetcher`` with at least
    ``[timestamp_utc, coinbase_premium]``, at the SAME resolution as the
    candles (1h or 1d; inferred and cross-checked).

    Causality: every lookup for the row at T reads the premium dated T or
    earlier; no rolling window extends past T.
    """
    if not ohlcv.index.is_monotonic_increasing:
        ohlcv = ohlcv.sort_index()

    t = _candle_times(ohlcv.index)
    bar = _infer_bar(
        pd.DatetimeIndex(premium["timestamp_utc"]), "premium frame"
    )
    ohlcv_bar = _infer_bar(t, "ohlcv index")
    if bar != ohlcv_bar:
        raise ValueError(
            f"resolution mismatch: premium frame is {bar!r} but candles are "
            f"{ohlcv_bar!r} — refusing to mix"
        )

    s = _grid_series(premium, bar)

    # Feature 1: raw premium at the closed bar T. NaN where the bar was
    # dropped at alignment — propagates to build_features' dropna.
    raw = pd.Series(s.reindex(t).to_numpy(), index=ohlcv.index)

    # Feature 2: 168-bar rolling z-score, window ending at T inclusive.
    roll = s.rolling(window=PREMIUM_Z_WINDOW, min_periods=PREMIUM_Z_WINDOW)
    std = roll.std(ddof=0)
    z_grid = (s - roll.mean()) / std.replace(0.0, np.nan)
    z = pd.Series(z_grid.reindex(t).to_numpy(), index=ohlcv.index).fillna(0.0)

    # Feature 3: 24-bar momentum, premium[T] / premium[T-24 bars] - 1.
    v0 = s.reindex(t).to_numpy()
    v24 = s.reindex(t - PREMIUM_MOM_LAG * bar).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        mom = v0 / v24 - 1.0
    mom = np.where(np.isfinite(mom), mom, np.nan)  # denom 0/NaN -> NaN -> 0
    mom = pd.Series(mom, index=ohlcv.index).fillna(0.0)

    out = pd.DataFrame(index=ohlcv.index)
    out["coinbase_premium"] = raw
    out["premium_zscore_168h"] = z
    out["premium_mom_24h"] = mom
    return out[list(PREMIUM_FEATURE_COLUMNS)]
