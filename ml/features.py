"""Feature engineering (features/labels phase).

Transforms raw OHLCV into a strictly *causal* feature matrix: a feature row at
time ``t`` uses ONLY the closed candle at ``t`` and bars before it. No rolling
window reaches into ``t+1``, so nothing here can see the bar the label predicts.

``ts`` convention
-----------------
A feature row's ``ts`` is the OPEN time (epoch ms, UTC) of the CLOSED candle the
features were computed FROM — the same bar convention the data layer uses after
dropping the incomplete trailing bar (see :mod:`scripts.fetch_data`). It is NOT
the bar being predicted. The label (:mod:`ml.labels`) looks at the forward
return measured from candles strictly AFTER ``ts``; the windows never overlap.

Feature set (all stationary, all trailing)
-------------------------------------------
* ``log_ret_1h``  — 1-bar log return,  ``ln(close_t / close_{t-1})``.
* ``log_ret_4h``  — 4-bar log return,  ``ln(close_t / close_{t-4})``.
* ``gk_vol``      — Garman-Klass volatility from this bar's OHLC.
* ``gk_vol_ma24`` — 24-period trailing mean of ``gk_vol`` (ending at t).
* ``z_close_50``  — 50-period trailing z-score of close (ending at t).
* ``z_vol``       — 50-period trailing z-score of volume (ending at t).

The currently-forming candle is already dropped at ingest time, so every bar
read from the DB is closed; using bar ``t`` in row ``t`` is therefore causal.

On-chain features (Experiment 6, optional, 1d only)
---------------------------------------------------
When :func:`build_features` is given an ``onchain`` frame (CoinMetrics
``[date, AdrActCnt]`` from :mod:`src.onchain.coinmetrics_fetcher`), exactly
three more columns are APPENDED after the OHLCV set, computed by
:mod:`src.onchain.features` under a hard D-1 point-in-time lag (a row for
trading day D uses only AdrActCnt dated D-1 or earlier — proven by
``tests/test_onchain_pit_leakage.py``):

* ``adr_zscore_28d``        — 28-day rolling z-score of AdrActCnt, ending D-1.
* ``adr_mom_7d``            — ``AdrActCnt[D-1] / AdrActCnt[D-8] - 1``.
* ``adr_price_diverge_28d`` — ``adr_zscore_28d`` minus the 28-day trailing
  z-score of close (pipeline convention).

Without ``onchain`` the output is unchanged: exactly :data:`FEATURE_COLUMNS`.

Coinbase-premium features (Experiment 8, optional, 1h only)
-----------------------------------------------------------
When :func:`build_features` is given a ``premium`` frame (the aligned
dual-exchange series from :mod:`src.onchain.coinbase_premium_fetcher`),
exactly three more columns are APPENDED after every existing feature,
computed by :mod:`src.onchain.premium_features` under a T-inclusive PIT rule
(the premium at T comes from both exchanges' CLOSED bars at T; rolling
windows never reach past T — proven by
``tests/test_coinbase_premium_pit_leakage.py``):

* ``coinbase_premium``     — raw premium at the closed bar T.
* ``premium_zscore_168h``  — 168-bar rolling z-score ending at T.
* ``premium_mom_24h``      — ``premium[T] / premium[T-24] - 1``.

Without ``premium`` the output is unchanged.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd

from config import Settings, get_settings
from core.database import FEATURE_COLUMNS, Database
from src.onchain.features import (
    ADR_MOM_LAG,
    ADR_Z_WINDOW,
    ONCHAIN_FEATURE_COLUMNS,
    add_onchain_features,
)
from src.onchain.premium_features import (
    PREMIUM_FEATURE_COLUMNS,
    PREMIUM_MOM_LAG,
    PREMIUM_Z_WINDOW,
    add_premium_features,
)

# Lookback windows. Kept as named constants because they are part of the
# feature recipe that feature_hash pins.
GK_VOL_MA_WINDOW = 24
ZSCORE_WINDOW = 50
LOG_RET_LONG_LAG = 4

# Bump when the feature computation changes in a way that should invalidate
# previously-stored feature rows / trained models.
FEATURE_VERSION = 1


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Trailing z-score: (x_t - mean of last `window`) / std of last `window`.

    Both the mean and std are computed over the trailing window ENDING at t
    (inclusive), so the value at t never uses t+1. ``min_periods=window`` makes
    the warmup rows NaN rather than computing on a short, biased window; they
    are dropped by :func:`build_features`.
    """
    roll = series.rolling(window=window, min_periods=window)
    mean = roll.mean()
    std = roll.std(ddof=0)
    return (series - mean) / std


def _garman_klass_vol(df: pd.DataFrame) -> pd.Series:
    """Per-bar Garman-Klass volatility estimate from OHLC of the SAME bar.

    GK uses only the open/high/low/close of the (closed) bar at t, so it is
    causal. Returns the square root of the variance estimate (a volatility,
    not a variance). Guards against non-positive prices.
    """
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(h / low)
        log_co = np.log(c / o)
        var = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    var = np.where(var < 0.0, 0.0, var)  # numerical floor; GK var is >= 0
    return pd.Series(np.sqrt(var), index=df.index)


def compute_feature_hash(
    include_onchain: bool = False, include_premium: bool = False
) -> str:
    """Stable hash of the feature RECIPE (not of any data).

    Combines the ordered feature names, the lookback parameters, and the
    feature-code version. Identical for every row of one build; changes only
    when the recipe changes. The model phase can store this alongside a trained
    artifact and refuse to predict on features built under a different recipe.

    ``include_onchain=True`` extends the recipe with the three Experiment-6
    on-chain columns and their parameters, so a model trained WITH them can
    never be silently scored against features built WITHOUT them (or vice
    versa). The default hash is unchanged from before the experiment.
    """
    parts = [
        f"v{FEATURE_VERSION}",
        "cols=" + ",".join(FEATURE_COLUMNS),
        f"gk_ma={GK_VOL_MA_WINDOW}",
        f"z={ZSCORE_WINDOW}",
        f"ret_lag={LOG_RET_LONG_LAG}",
    ]
    if include_onchain:
        parts += [
            "onchain=" + ",".join(ONCHAIN_FEATURE_COLUMNS),
            f"adr_z={ADR_Z_WINDOW}",
            f"adr_mom={ADR_MOM_LAG}",
            "adr_lag=1",
        ]
    if include_premium:
        parts += [
            "premium=" + ",".join(PREMIUM_FEATURE_COLUMNS),
            f"prem_z={PREMIUM_Z_WINDOW}",
            f"prem_mom={PREMIUM_MOM_LAG}",
            "prem_lag=0",
        ]
    recipe = "|".join(parts)
    return hashlib.sha256(recipe.encode("utf-8")).hexdigest()[:16]


def build_features(
    ohlcv: pd.DataFrame,
    onchain: Optional[pd.DataFrame] = None,
    premium: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return the causal feature matrix for one symbol/timeframe.

    ``ohlcv`` must be indexed by ``ts`` (bar open time, ms) ascending, with
    columns open/high/low/close/volume — every bar already closed. The result
    is indexed by the same ``ts`` and has exactly the columns in
    :data:`FEATURE_COLUMNS`. Warmup rows (where any trailing window is short)
    are dropped via ``dropna`` so no row is computed on partial history.

    ``onchain`` (1d candles only): the CoinMetrics ``[date, AdrActCnt]`` frame.
    When given, the three Experiment-6 on-chain columns are appended AFTER
    every existing feature, computed under the D-1 point-in-time lag by
    :func:`src.onchain.features.add_onchain_features`. Existing columns are
    not touched.

    ``premium`` (1h candles only): the aligned Coinbase/Binance frame from
    :mod:`src.onchain.coinbase_premium_fetcher`. When given, the three
    Experiment-8 premium columns are appended after everything else under the
    T-inclusive PIT rule via
    :func:`src.onchain.premium_features.add_premium_features`.

    Causality: every feature at row t depends only on close/volume/OHLC at t
    and earlier (and, for the on-chain columns, AdrActCnt dated t-1 or
    earlier). No ``shift(-k)`` / forward window appears here.
    """
    if not ohlcv.index.is_monotonic_increasing:
        ohlcv = ohlcv.sort_index()

    close = ohlcv["close"]
    volume = ohlcv["volume"]

    feats = pd.DataFrame(index=ohlcv.index)
    # Log returns: ratio of current close to a PAST close (positive lag).
    feats["log_ret_1h"] = np.log(close / close.shift(1))
    feats["log_ret_4h"] = np.log(close / close.shift(LOG_RET_LONG_LAG))
    # Garman-Klass vol of the current bar, and its trailing mean.
    gk = _garman_klass_vol(ohlcv)
    feats["gk_vol"] = gk
    feats["gk_vol_ma24"] = gk.rolling(
        window=GK_VOL_MA_WINDOW, min_periods=GK_VOL_MA_WINDOW
    ).mean()
    # Trailing z-scores.
    feats["z_close_50"] = _zscore(close, ZSCORE_WINDOW)
    feats["z_vol"] = _zscore(volume, ZSCORE_WINDOW)

    feats = feats[list(FEATURE_COLUMNS)]
    if onchain is not None:
        # Appended strictly AFTER all existing features; same index, so the
        # shared dropna below also drops on-chain warmup rows.
        feats = feats.join(add_onchain_features(ohlcv, onchain))
    if premium is not None:
        # Appended last. NaN raw-premium rows (alignment-dropped hours) fall
        # to the shared dropna; z-warmup and momentum-gap rows are 0-filled
        # per the registered fill rules, not dropped.
        feats = feats.join(add_premium_features(ohlcv, premium))
    return feats.dropna()


def build_and_store(
    symbol: str,
    timeframe: str,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
) -> int:
    """Build features from stored candles and persist them.

    Reads candles via :meth:`Database.load_candles`, builds the causal matrix,
    tags every row with :func:`compute_feature_hash`, and upserts into the
    ``features`` table. Returns the number of feature rows written.
    """
    settings = settings or get_settings()
    db = db or Database(settings)
    db.init_schema()

    rows = db.load_candles(symbol, timeframe)
    if not rows:
        return 0
    ohlcv = pd.DataFrame(
        [dict(r) for r in rows]
    ).set_index("ts").sort_index()

    feats = build_features(ohlcv)
    if feats.empty:
        return 0

    fh = compute_feature_hash()
    payload = [
        {"ts": int(ts), **{c: float(row[c]) for c in FEATURE_COLUMNS}}
        for ts, row in feats.iterrows()
    ]
    return db.upsert_features(symbol, timeframe, payload, feature_hash=fh)
