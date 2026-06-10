"""On-chain feature computation with a hard D-1 point-in-time lag (Experiment 6).

The pre-registered point-in-time rule this module enforces:

    A feature row for trading day D may use ONLY AdrActCnt values dated D-1 or
    earlier. CoinMetrics publishes day D's value at end of day D UTC, so day D
    is NOT available at trading open on day D.

The lag is implemented by SAMPLING DATE, not by row shifting: every lookup into
the on-chain series for the candle dated D reads calendar date D-1 (and the
rolling windows END at D-1). This stays correct even if the OHLCV index has
gaps. Proven by perturbation in ``tests/test_onchain_pit_leakage.py``.

Exactly three features (the experiment's full scope):

* ``adr_zscore_28d``        — 28-day rolling z-score of AdrActCnt, window
                              ending at D-1. std==0 or NaN -> 0.
* ``adr_mom_7d``            — ``AdrActCnt[D-1] / AdrActCnt[D-8] - 1``.
                              Denominator 0 or NaN -> 0.
* ``adr_price_diverge_28d`` — ``adr_zscore_28d`` minus the 28-day rolling
                              z-score of close. The price z-score uses the same
                              window and the same trailing convention as the
                              OHLCV z-scores in ``ml.features`` (it ends at the
                              row's own CLOSED candle D; close_D is final at
                              end of day D, when the row is acted on at D+1
                              open).

This module is deliberately free of imports from ``ml`` so that
``ml.features`` can import it without a cycle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ADR_COLUMN = "AdrActCnt"
ADR_Z_WINDOW = 28
ADR_MOM_LAG = 7

# Order matters: appended after FEATURE_COLUMNS in the DB schema / recipe hash.
ONCHAIN_FEATURE_COLUMNS = (
    "adr_zscore_28d",
    "adr_mom_7d",
    "adr_price_diverge_28d",
)

_DAY = pd.Timedelta(days=1)


def _candle_dates(index: pd.Index) -> pd.DatetimeIndex:
    """UTC calendar day of each 1d candle's OPEN ts (epoch ms)."""
    return pd.DatetimeIndex(
        pd.to_datetime(index.to_numpy(dtype="int64"), unit="ms")
    ).normalize()


def _daily_series(adr: pd.DataFrame) -> pd.Series:
    """AdrActCnt indexed by a COMPLETE daily calendar (gaps become NaN).

    Reindexing onto the full calendar makes the 28-row rolling window a true
    28-DAY window; ``min_periods`` then turns any gap-crossing window into NaN
    (mapped to 0 by the spec'd fill rule) instead of silently stretching it.
    """
    s = (
        adr.sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .set_index("date")[ADR_COLUMN]
        .astype(float)
    )
    full = pd.date_range(s.index[0], s.index[-1], freq="D")
    return s.reindex(full)


def add_onchain_features(ohlcv: pd.DataFrame, adr: pd.DataFrame) -> pd.DataFrame:
    """Return the three on-chain feature columns, indexed like ``ohlcv``.

    ``ohlcv``: 1d candles indexed by ``ts`` (bar OPEN time, epoch ms, UTC,
    ascending) with at least a ``close`` column — the same frame
    ``ml.features.build_features`` consumes.
    ``adr``: the CoinMetrics frame ``[date, AdrActCnt]`` from
    ``src.onchain.coinmetrics_fetcher`` (date = tz-naive UTC midnight).

    Causality: every on-chain lookup for the row dated D reads date D-1 or
    earlier; the close-price z-score uses closes up to the row's own closed
    candle, exactly like the existing OHLCV features.
    """
    if not ohlcv.index.is_monotonic_increasing:
        ohlcv = ohlcv.sort_index()

    s = _daily_series(adr)
    dates = _candle_dates(ohlcv.index)
    sample = dates - _DAY  # the D-1 lag: row D reads the day-(D-1) value

    # Feature 1: 28-day rolling z-score of AdrActCnt, window ending at D-1.
    roll = s.rolling(window=ADR_Z_WINDOW, min_periods=ADR_Z_WINDOW)
    std = roll.std(ddof=0)
    z_daily = (s - roll.mean()) / std.replace(0.0, np.nan)
    adr_z = pd.Series(
        z_daily.reindex(sample).to_numpy(), index=ohlcv.index
    ).fillna(0.0)

    # Feature 2: 7-day momentum, AdrActCnt[D-1] / AdrActCnt[D-8] - 1.
    v_lag1 = s.reindex(sample).to_numpy()
    v_lag8 = s.reindex(dates - (ADR_MOM_LAG + 1) * _DAY).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        mom = v_lag1 / v_lag8 - 1.0
    mom = np.where(np.isfinite(mom), mom, np.nan)  # denom 0/NaN -> NaN -> 0
    adr_mom = pd.Series(mom, index=ohlcv.index).fillna(0.0)

    # Feature 3: on-chain z minus price z (same 28d window, pipeline trailing
    # convention for the price leg). Warmup rows stay NaN and are dropped by
    # build_features' dropna, same as every other warmup.
    close = ohlcv["close"].astype(float)
    croll = close.rolling(window=ADR_Z_WINDOW, min_periods=ADR_Z_WINDOW)
    price_z = (close - croll.mean()) / croll.std(ddof=0)

    out = pd.DataFrame(index=ohlcv.index)
    out["adr_zscore_28d"] = adr_z
    out["adr_mom_7d"] = adr_mom
    out["adr_price_diverge_28d"] = adr_z - price_z
    return out[list(ONCHAIN_FEATURE_COLUMNS)]
